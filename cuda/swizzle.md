## swizzle的理解和计算（以矩阵转置为例）
swizzle通过地址交错映射来避免bank cconflict  
以下代码思路是，读取的时候每个warp读一行，存的时候看每个warp也是存一行，这样保证内存合并访问，同时存的时候线程（ty,tx）所在位置要存的数据是对应线程（tx,ty）存入的数据。
```cuda
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define BLOCK_SIZE 32
#define M0 1024
#define N0 1024

// 检查 CUDA 错误
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// 1. Naive 版本：直接使用全局内存
__global__ void transpose_naive( float* input, float* output, int M, int N) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;  // 输出矩阵的行
    int col = blockDim.x * blockIdx.x + threadIdx.x;  // 输出矩阵的列
    if (row < N && col < M) {
        // input 是 M 行 N 列，output 是 N 行 M 列
        output[row * M + col] = __ldg(&input[col * N + row]);
    }
}

// 2. Padding 版本：使用共享内存，每行填充一个元素
template <int BS>
__global__ void transpose_padding( float* input, float* output, int M, int N) {
    __shared__ float s_mem[BS][BS + 1];  // 填充一行

    int bx = blockIdx.x * BS;
    int by = blockIdx.y * BS;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 加载到共享内存（按行存储）
    int x = bx + tx;
    int y = by + ty;
    if (x < N && y < M) {
        s_mem[ty][tx] = input[y * N + x];
    }
    __syncthreads();

    // 转置写回全局内存（按列读取）
    x = by + tx;  // 输出列
    y = bx + ty;  // 输出行
    if (x < M && y < N) {
        output[y * M + x] = s_mem[tx][ty];  // 注意：这里读取时行和列交换
    }
}

// 3. Swizzle 版本：使用共享内存 + 异或重映射
template <int BS>
__global__ void transpose_swizzle(float* input, float* output, int M, int N) {
    __shared__ float s_mem[BS][BS];  // 无填充

    int bx = blockIdx.x * BS;
    int by = blockIdx.y * BS;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 写入时使用 swizzle：物理列 = tx ^ ty
    int x = bx + tx;
    int y = by + ty;
    if (x < N && y < M) {
        s_mem[ty][tx ^ ty] = input[y * N + x];
    }
    __syncthreads();

    // 读取时同样使用 swizzle：从物理位置 (ty, tx ^ ty) 读取
    x = by + tx;  // 输出列
    y = bx + ty;  // 输出行
    if (x < M && y < N) {
        output[y * M + x] = s_mem[tx][tx ^ ty];
    }
}

// 用于验证的 CPU 转置函数
void cpu_transpose(const float* in, float* out, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            out[j * M + i] = in[i * N + j];
        }
    }
}

// 比较两个矩阵是否相等（误差容忍）
bool compare_matrices(const float* a, const float* b, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(a[i] - b[i]) > 1e-5) {
            printf("Mismatch at index %d: %f vs %f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

int main() {
    int M=M0;
    int N=N0;
    int size_input = M * N;
    int size_output = N * M;
    size_t bytes_input = size_input * sizeof(float);
    size_t bytes_output = size_output * sizeof(float);

    // 分配主机内存
    float *h_input = (float*)malloc(bytes_input);
    float *h_output_cpu = (float*)malloc(bytes_output);
    float *h_output_naive = (float*)malloc(bytes_output);
    float *h_output_padding = (float*)malloc(bytes_output);
    float *h_output_swizzle = (float*)malloc(bytes_output);

    // 初始化输入矩阵（随机数）
    for (int i = 0; i < size_input; i++) {
        h_input[i] = rand() / (float)RAND_MAX;
    }

    // 计算 CPU 转置结果作为参考
    cpu_transpose(h_input, h_output_cpu, M, N);

    // 分配设备内存
    float *d_input, *d_output_naive, *d_output_padding, *d_output_swizzle;
    CUDA_CHECK(cudaMalloc(&d_input, bytes_input));
    CUDA_CHECK(cudaMalloc(&d_output_naive, bytes_output));
    CUDA_CHECK(cudaMalloc(&d_output_padding, bytes_output));
    CUDA_CHECK(cudaMalloc(&d_output_swizzle, bytes_output));

    // 拷贝输入到设备
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes_input, cudaMemcpyHostToDevice));

    // 定义 kernel 启动配置
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(CEIL(M, BLOCK_SIZE), CEIL(N, BLOCK_SIZE));  // 注意：grid.x 对应输出列数 M，grid.y 对应输出行数 N

    // 创建 CUDA 事件用于计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms_naive, ms_padding, ms_swizzle;

    // 1. 运行 naive 版本
    CUDA_CHECK(cudaEventRecord(start));
    transpose_naive<<<grid, block>>>(d_input, d_output_naive, M0, N0);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));
    CUDA_CHECK(cudaMemcpy(h_output_naive, d_output_naive, bytes_output, cudaMemcpyDeviceToHost));
    if (!compare_matrices(h_output_naive, h_output_cpu, size_output)) {
        printf("Naive version failed correctness!\n");
    } else {
        printf("Naive version: %.3f ms\n", ms_naive);
    }

    // 2. 运行 padding 版本
    CUDA_CHECK(cudaEventRecord(start));
    transpose_padding<BLOCK_SIZE><<<grid, block>>>(d_input, d_output_padding, M0, N0);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_padding, start, stop));
    CUDA_CHECK(cudaMemcpy(h_output_padding, d_output_padding, bytes_output, cudaMemcpyDeviceToHost));
    if (!compare_matrices(h_output_padding, h_output_cpu, size_output)) {
        printf("Padding version failed correctness!\n");
    } else {
        printf("Padding version: %.3f ms\n", ms_padding);
    }

    // 3. 运行 swizzle 版本
    CUDA_CHECK(cudaEventRecord(start));
    transpose_swizzle<BLOCK_SIZE><<<grid, block>>>(d_input, d_output_swizzle, M0, N0);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_swizzle, start, stop));
    CUDA_CHECK(cudaMemcpy(h_output_swizzle, d_output_swizzle, bytes_output, cudaMemcpyDeviceToHost));
    if (!compare_matrices(h_output_swizzle, h_output_cpu, size_output)) {
        printf("Swizzle version failed correctness!\n");
    } else {
        printf("Swizzle version: %.3f ms\n", ms_swizzle);
    }

    // 释放资源
    free(h_input);
    free(h_output_cpu);
    free(h_output_naive);
    free(h_output_padding);
    free(h_output_swizzle);
    cudaFree(d_input);
    cudaFree(d_output_naive);
    cudaFree(d_output_padding);
    cudaFree(d_output_swizzle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```
