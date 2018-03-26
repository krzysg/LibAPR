
#include "ComputeGradientCudaRegs.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

constexpr int blockWidth = 3;
constexpr int numOfWorkers = 32;

template<typename T>
__global__ void superKernel(T *image, size_t x_num, size_t y_num, size_t z_num) {
    T localCache[numOfWorkers];
    const int localId = threadIdx.x;

    int yi = 0;
    // saturate cache with first two columns
    #pragma unroll
    for (int i = 0; i < blockWidth - 1; ++i) {
        localCache[i] = image[yi * y_num + localId];
        ++yi;
    }

    do {
        // load next data
        localCache[yi % blockWidth] = image[yi * y_num + localId];

        // do calculations and store
        localCache[yi % blockWidth] += localCache[(yi + 1)  % blockWidth] + localCache[(yi + 2) % blockWidth];
        image[yi * y_num + localId] = localCache[yi % blockWidth];

        ++yi;
    }
    while (yi < y_num);

//    for (int i = 0; i < numOfWorkers; ++i)
//    if (i == localId) {
//        for (int i = 0; i < numOfWorkers; ++i) {
//            printf("%.0f ", localCache[i]);
//        }
//        printf("\n");
//    }
}

void cudaFilterBsplineYdirectionRegs(MeshData<float> &input) {
    APRTimer timer;
    timer.verbose_flag=true;


    timer.start_timer("cuda: memory alloc + data transfer to device");
    size_t inputSize = input.mesh.size() * sizeof(float);
    float *cudaInput;
    cudaMalloc(&cudaInput, inputSize);
    cudaMemcpy(cudaInput, input.mesh.get(), inputSize, cudaMemcpyHostToDevice);
    timer.stop_timer();

//    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
//    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

    dim3 threadsPerBlock(numOfWorkers, 1, 1);
    dim3 numBlocks((input.x_num * input.z_num + threadsPerBlock.x - 1)/threadsPerBlock.x, 1, 1);
    std::cout << "Number of blocks  (x/y/z):  " << numBlocks.x << "/" << numBlocks.y << "/" << numBlocks.z << std::endl;
    std::cout << "Number of threads (x/y/z): " << threadsPerBlock.x << "/" << threadsPerBlock.y << "/" << threadsPerBlock.z << std::endl;

    timer.start_timer("cuda: calculations on device ============================================================================ ");
    superKernel<<<numBlocks, threadsPerBlock>>>(cudaInput, input.x_num, input.y_num, input.z_num);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)printf("Error: %s\n", cudaGetErrorString(err));
    timer.stop_timer();

    timer.start_timer("cuda: transfer data from device and freeing memory");
    cudaMemcpy((void*)input.mesh.get(), cudaInput, inputSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaInput);
    timer.stop_timer();
}