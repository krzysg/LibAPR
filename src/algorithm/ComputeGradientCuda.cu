#include "ComputeGradientCuda.hpp"
#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

__device__ size_t getMin(size_t a, size_t  b) { return (a < b) ? a : b; }
__device__ size_t getMax(size_t a, size_t  b) { return (a > b) ? a : b; }
__device__ float getMax(float a, float  b) { return (a > b) ? a : b; }
__device__ void swap(const float * &a, const float * &b) {const float *temp = b; b = a; a = temp;}

__global__ void gradient(float *input, size_t x_num, size_t y_num, size_t z_num, float *grad, size_t x_num_ds, size_t y_num_ds, float hx, float hy, float hz) {
//    float *temp = new float[y_num];
//    if (temp == nullptr) printf("NULL on CUDA\n");
    float temp[1024];
    for (size_t idx = 0; idx < y_num; ++idx) temp[idx] = 0.0;

    const size_t xnumynum = x_num * y_num;

    size_t xi = (blockIdx.x * blockDim.x) + threadIdx.x;
    size_t yi = (blockIdx.y * blockDim.y) + threadIdx.y;
    size_t zi = (blockIdx.z * blockDim.z) + threadIdx.z;
    if (xi*2 >= x_num || yi*2 >= y_num || zi*2 >= z_num) return;

    size_t xf = xi * 2;
    size_t xt = getMin(xf + 1, x_num - 1);
    size_t zf = zi * 2;
    size_t zt = getMin(zf + 1, z_num - 1);

    for (size_t z = zf; z <= zt; ++z) {
        // Belows pointers up, down... are forming stencil in X (left <-> right) and Z ( up <-> down) direction and
        // are pointing to whole Y column. If out of bounds then 'replicate' (nearest array border value) approach is used.
        //
        //                 up
        //   ...   left  center  right ...
        //                down

        size_t xl = xf > 0 ? xf - 1 : 0;
        const float *left = input + z * xnumynum + xl * y_num; // boundary value is chosen
        const float *center = input + z * xnumynum + xf * y_num;

        //LHS boundary condition is accounted for wiht this initialization
        const size_t zMinus = z > 0 ? z - 1 : 0 /* boundary */;
        const size_t zPlus = getMin(z + 1, z_num - 1 /* boundary */);

        for (size_t x = xf; x <= xt; ++x) {
            const float *up = input + zMinus * xnumynum + x * y_num;
            const float *down = input + zPlus * xnumynum + x * y_num;
            const size_t xPlus = getMin(x + 1, x_num - 1 /* boundary */);
            const float *right = input + z * xnumynum + xPlus * y_num;

            //compute the boundary values
            if (y_num >= 2) {
                temp[0] = sqrt(pow((right[0] - left[0]) / (2 * hx), 2.0) + pow((down[0] - up[0]) / (2 * hz), 2.0) + pow((center[1] - center[0 /* boundary */]) / (2 * hy), 2.0));
                temp[y_num - 1] = sqrt(pow((right[y_num - 1] - left[y_num - 1]) / (2 * hx), 2.0) + pow((down[y_num - 1] - up[y_num - 1]) / (2 * hz), 2.0) + pow((center[y_num - 1 /* boundary */] - center[y_num - 2]) / (2 * hy), 2.0));
            }
            else {
                temp[0] = 0; // same values minus same values in x/y/z
            }

            //do the y gradient in range 1..y_num-2
            for (size_t y = 1; y < y_num - 1; ++y) {
                temp[y] = sqrt(pow((right[y] - left[y]) / (2 * hx), 2.0) + pow((down[y] - up[y]) / (2 * hz), 2.0) + pow((center[y + 1] - center[y - 1]) / (2 * hy), 2.0));
            }

            // Set as a downsampled gradient maximum from 2x2x2 gradient cubes
            int64_t z_2 = z / 2;
            int64_t x_2 = x / 2;
            for (size_t k = 0; k < y_num_ds; ++k) {
                size_t k_s = getMin(2 * k + 1, y_num - 1);
                const size_t idx = z_2 * x_num_ds * y_num_ds + x_2 * y_num_ds + k;
                grad[idx] = getMax(temp[2 * k], getMax(temp[k_s], grad[idx]));
            }

            // move left, center to current center, right (both +1 to right)
            swap(left, center);
            swap(center, right);
        }
    }
//    delete temp;
}

void cudaDownsampledGradient(const MeshData<float> &input, MeshData<float> &grad, const float hx, const float hy,const float hz) {
    size_t inputSize = input.mesh.size() * sizeof(float);
    float *cudaInput;
    cudaMalloc(&cudaInput, inputSize);
    cudaMemcpy(cudaInput, input.mesh.get(), inputSize, cudaMemcpyHostToDevice);

    size_t gradSize = grad.mesh.size() * sizeof(float);
    float *cudaGrad;
    cudaMalloc(&cudaGrad, gradSize);
    cudaMemcpy(cudaGrad, grad.mesh.get(), gradSize, cudaMemcpyHostToDevice);

    std::cout << "INP: " << input << std::endl;
    std::cout << "GRA: " << grad << std::endl;

    dim3 threadsPerBlock(8, 1, 8);
    dim3 numBlocks((input.x_num + threadsPerBlock.x - 1)/threadsPerBlock.x,
                   1,
                   (input.z_num + threadsPerBlock.z - 1)/threadsPerBlock.z);
    std::cout << "NUM BLOCKS:" << numBlocks.x << "/" << numBlocks.y << "/" << numBlocks.z << std::endl;
    std::cout << "NUM THREAD:" << threadsPerBlock.x << "/" << threadsPerBlock.y << "/" << threadsPerBlock.z << std::endl;
    APRTimer timer;
    timer.verbose_flag=true;
    timer.start_timer("CUDA only");
    gradient<<<numBlocks,threadsPerBlock>>>(cudaInput, input.x_num, input.y_num, input.z_num, cudaGrad, grad.x_num, grad.y_num, hx, hy, hz);
    cudaDeviceSynchronize();
    timer.stop_timer();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)printf("Error: %s\n", cudaGetErrorString(err));
    cudaMemcpy((void*)input.mesh.get(), cudaInput, inputSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaInput);
    cudaMemcpy((void*)grad.mesh.get(), cudaGrad, gradSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaGrad);
}
