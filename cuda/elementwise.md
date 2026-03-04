### cuda add kernel
```
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void add_kernel(const float *input1, const float *input2, float *output, int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float4 x1,x2,o1;
  if (idx*4 < size){
    // output[idx] = input1[idx] + input2[idx];
    x1=((float4*)(input1))[idx];
    x2=((float4*)(input2))[idx];
    o1.x = x1.x+x2.x;
    o1.y = x1.y+x2.y;
    o1.z = x1.z+x2.z;
    o1.w = x1.w+x2.w;
    ((float4*)(output))[idx]=o1;
  }
}

__global__ void add_kernel1(const float *input1, const float *input2, float *output, int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size){
    output[idx] = __ldg(&input1[idx]) + __ldg(&input2[idx]);
  }
}

void add(const float* d_input1, const float* d_input2, float* d_output, int size) {
    if (size <= 0) return;

    if (d_input1 == nullptr || d_input2 == nullptr || d_output == nullptr) {
        fprintf(stderr, "Error: null pointer in add function\n");
        return;
    }

    int n_threads = 256;
    int n_blocks = (size + n_threads - 1) / n_threads /4;
    add_kernel<<<n_blocks, n_threads>>>(d_input1, d_input2, d_output, size);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after add_kernel: %s\n", cudaGetErrorString(err));
    }
}

int main() {
    const int N = 1024*10000;  
    size_t bytes = N * sizeof(float);

    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 3. 将数据从主机拷贝到设备
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // 4. 调用加法函数（纯 CUDA 版本）
    add(d_a, d_b, d_c, N);

    // 5. 将结果拷回主机
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // 6. 验证（可选）
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {
            correct = false;
            break;
        }
    }
    std::cout << "Result: " << (correct ? "PASS" : "FAIL") << std::endl;

    // 7. 清理资源
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```
