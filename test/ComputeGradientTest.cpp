/*
 * Created by Krzysztof Gonciarz 2018
 */
#include <array>
#include <gtest/gtest.h>
#include "data_structures/Mesh/MeshData.hpp"
#include "algorithm/ComputeGradient.hpp"
#include "algorithm/ComputeGradientCuda.hpp"
#include <random>

namespace {
    /**
     * Compares mesh with provided data
     * @param mesh
     * @param data - data with [Z][Y][X] structure
     * @return true if same
     */
    template <typename T>
    bool compare(MeshData<T> &mesh, const float *data, const float epsilon) {
        size_t dataIdx = 0;
        for (size_t z = 0; z < mesh.z_num; ++z) {
            for (size_t y = 0; y < mesh.y_num; ++y) {
                for (size_t x = 0; x < mesh.x_num; ++x) {
                    bool v = std::abs(mesh(y, x, z) - data[dataIdx]) < epsilon;
                    if (v == false) {
                        std::cerr << "Mesh and expected data differ. First place at (Y, X, Z) = " << y << ", " << x << ", " << z << ") " << mesh(y, x, z) << " vs " << data[dataIdx] << std::endl;
                        return false;
                    }
                    ++dataIdx;
                }
            }
        }
        return true;
    }

    TEST(ComputeGradientTest, 2D_XY) {
        {   // Corner points
            MeshData<float> m(6, 6, 1, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {1.41, 0, 4.24,
                              0, 0, 0,
                              2.82, 0, 5.65};
            // put values in corners
            m(0, 0, 0) = 2;
            m(5, 0, 0) = 4;
            m(0, 5, 0) = 6;
            m(5, 5, 0) = 8;
            MeshData<float> grad;
            grad.initDownsampled(m, 0);
            ComputeGradient cg;
            cg.calc_bspline_fd_ds_mag(m, grad, 1, 1, 1);
            ASSERT_TRUE(compare(grad, expect, 0.01));
        }
        {   // In the middle
            MeshData<float> m(6, 6, 1, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {1, 1, 0,
                              1, 0, 0,
                              0, 0, 0};
            // put values in corners
            m(1, 1, 0) = 2;
            MeshData<float> grad;
            grad.initDownsampled(m, 0);
            ComputeGradient cg;
            cg.calc_bspline_fd_ds_mag(m, grad, 1, 1, 1);
            ASSERT_TRUE(compare(grad, expect, 0.01));
        }
        {   // One pixel image 1x1x1
            MeshData<float> m(1, 1, 1, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {0};
            // put values in corners
            m(0, 0, 0) = 2;
            MeshData<float> grad;
            grad.initDownsampled(m, 0);
            ComputeGradient cg;
            cg.calc_bspline_fd_ds_mag(m, grad, 1, 1, 1);
            ASSERT_TRUE(compare(grad, expect, 0.01));
        }

    }

    TEST(ComputeGradientTest, Corners3D) {
        MeshData<float> m(6, 6, 4, 0);
        // expect gradient is 3x3x2 X/Y/Z plane
        float expect[] = { 1.73, 0, 5.19,
                           0,    0,    0,
                           3.46, 0, 6.92,

                           8.66, 0, 12.12,
                           0,    0,     0,
                          10.39, 0, 13.85 };
        // put values in corners
        m(0, 0, 0) = 2;
        m(5, 0, 0) = 4;
        m(0, 5, 0) = 6;
        m(5, 5, 0) = 8;
        m(0, 0, 3) = 10;
        m(5, 0, 3) = 12;
        m(0, 5, 3) = 14;
        m(5, 5, 3) = 16;

        MeshData<float> grad;
        grad.initDownsampled(m, 0);
        ComputeGradient cg;
        cg.calc_bspline_fd_ds_mag(m, grad, 1, 1, 1);
        ASSERT_TRUE(compare(grad, expect, 0.01));
    }

#ifdef APR_USE_CUDA
    TEST(ComputeGradientTest, 2D_XY_CUDA) {
        // Corner points
        MeshData<float> m(6, 6, 1, 0);
        // expect gradient is 3x3 X/Y plane
        float expect[] = {1.41, 0, 4.24,
                          0, 0, 0,
                          2.82, 0, 5.65};
        // put values in corners
        m(0, 0, 0) = 2;
        m(5, 0, 0) = 4;
        m(0, 5, 0) = 6;
        m(5, 5, 0) = 8;
        MeshData<float> grad;
        grad.initDownsampled(m, 0);
        cudaDownsampledGradient(m, grad, 1, 1, 1);
        ASSERT_TRUE(compare(grad, expect, 0.01));

    }


    TEST(ComputeGradientTest, Corners3DCuda) {
        MeshData<float> m(6, 6, 4, 0);
        // expect gradient is 3x3x2 X/Y/Z plane
        float expect[] = { 1.73, 0, 5.19,
                           0,    0,    0,
                           3.46, 0, 6.92,

                           8.66, 0, 12.12,
                           0,    0,     0,
                           10.39, 0, 13.85 };
        // put values in corners
        m(0, 0, 0) = 2;
        m(5, 0, 0) = 4;
        m(0, 5, 0) = 6;
        m(5, 5, 0) = 8;
        m(0, 0, 3) = 10;
        m(5, 0, 3) = 12;
        m(0, 5, 3) = 14;
        m(5, 5, 3) = 16;

        MeshData<float> grad;
        grad.initDownsampled(m, 0);
        cudaDownsampledGradient(m, grad, 1, 1, 1);
        ASSERT_TRUE(compare(grad, expect, 0.01));
    }

    TEST(ComputeGradientTest, Corners3DCudaBigCompareToOld) {
        MeshData<float> m(512, 512, 512, 0);
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (size_t i = 0; i < m.mesh.size(); ++i) {
            m.mesh[i] =dist(mt);
        }
        APRTimer timer;
        timer.verbose_flag=true;

        MeshData<float> grad;
        grad.initDownsampled(m, 0);

        timer.start_timer("Old gradient");
        ComputeGradient cg;
        cg.calc_bspline_fd_ds_mag(m, grad, 1, 1, 1);
        timer.stop_timer();

        MeshData<float> gradCuda;
        gradCuda.initDownsampled(m, 0);

        timer.start_timer("CUDA gradient");
        cudaDownsampledGradient(m, gradCuda, 1, 1, 1);
        timer.stop_timer();

        bool once = true;
        int cnt=0;
        for (size_t i = 0; i < grad.mesh.size(); ++i) {
            if (std::abs(grad.mesh[i] - gradCuda.mesh[i]) > 0.0001) {
                if (once) { std::cout << "ERR " << grad.mesh[i] << " vs " <<  gradCuda.mesh[i] << std::endl; once = false; }
                cnt++;
            }
        }
        std::cout << cnt << " / " << grad.mesh.size() << std::endl;
    }

#endif
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
