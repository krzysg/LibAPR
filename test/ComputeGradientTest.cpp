/*
 * Created by Krzysztof Gonciarz 2018
 */
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include "data_structures/Mesh/PixelData.hpp"
#include "algorithm/ComputeGradient.hpp"
#include "algorithm/ComputeGradientCuda.hpp"
#include "algorithm/ComputeBsplineRecursiveFilterCuda.h"
#include "algorithm/ComputeInverseCubicBsplineCuda.h"
#include <random>
#include "algorithm/APRConverter.hpp"

namespace {
    /**
     * Compares mesh with provided data
     * @param mesh
     * @param data - data with [Z][Y][X] structure
     * @return true if same
     */
    template<typename T>
    bool compare(PixelData<T> &mesh, const float *data, const float epsilon) {
        size_t dataIdx = 0;
        for (size_t z = 0; z < mesh.z_num; ++z) {
            for (size_t y = 0; y < mesh.y_num; ++y) {
                for (size_t x = 0; x < mesh.x_num; ++x) {
                    bool v = std::abs(mesh(y, x, z) - data[dataIdx]) < epsilon;
                    if (v == false) {
                        std::cerr << "Mesh and expected data differ. First place at (Y, X, Z) = " << y << ", " << x
                                  << ", " << z << ") " << mesh(y, x, z) << " vs " << data[dataIdx] << std::endl;
                        return false;
                    }
                    ++dataIdx;
                }
            }
        }
        return true;
    }

    /**
     * Compares two meshes
     * @param expected
     * @param tested
     * @param maxNumOfErrPrinted - how many error values should be printed (-1 for all)
     * @return number of errors detected
     */
    template <typename T>
    int compareMeshes(const PixelData<T> &expected, const PixelData<T> &tested, double maxError = 0.0001, int maxNumOfErrPrinted = 3) {
        int cnt = 0;
        for (size_t i = 0; i < expected.mesh.size(); ++i) {
            if (std::abs(expected.mesh[i] - tested.mesh[i]) > maxError || std::isnan(expected.mesh[i]) ||
                std::isnan(tested.mesh[i])) {
                if (cnt < maxNumOfErrPrinted || maxNumOfErrPrinted == -1) {
                    std::cout << "ERROR expected vs tested mesh: " << expected.mesh[i] << " vs " << tested.mesh[i] << " IDX:" << tested.getStrIndex(i) << std::endl;
                }
                cnt++;
            }
        }
        std::cout << "Number of errors / all points: " << cnt << " / " << expected.mesh.size() << std::endl;
        return cnt;
    }

    /**
     * Generates mesh with provided dims with random values in range [0, 1] * multiplier
     * @param y
     * @param x
     * @param z
     * @param multiplier
     * @return
     */
    template <typename T>
    PixelData<T> getRandInitializedMesh(int y, int x, int z, float multiplier = 2.0f, bool useIdxNumbers = false) {
        PixelData<T> m(y, x, z);
        std::cout << "Mesh info: " << m << std::endl;
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (size_t i = 0; i < m.mesh.size(); ++i) {
            m.mesh[i] = useIdxNumbers ? i : dist(mt) * multiplier;
        }
        return m;
    }

    template<typename T>
    bool initFromZYXarray(PixelData<T> &mesh, const float *data) {
        size_t dataIdx = 0;
        for (size_t z = 0; z < mesh.z_num; ++z) {
            for (size_t y = 0; y < mesh.y_num; ++y) {
                for (size_t x = 0; x < mesh.x_num; ++x) {
                    mesh(y, x, z) = data[dataIdx];
                    ++dataIdx;
                }
            }
        }
        return true;
    }

    TEST(ComputeGradientTest, 2D_XY) {
        {   // Corner points
            PixelData<float> m(6, 6, 1, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {1.41, 0, 4.24,
                              0, 0, 0,
                              2.82, 0, 5.65};
            // put values in corners
            m(0, 0, 0) = 2;
            m(5, 0, 0) = 4;
            m(0, 5, 0) = 6;
            m(5, 5, 0) = 8;
            PixelData<float> grad;
            grad.initDownsampled(m, 0);
            ComputeGradient cg;
            cg.calc_bspline_fd_ds_mag(m, grad, 1, 1, 1);
            ASSERT_TRUE(compare(grad, expect, 0.05));
        }
        {   // In the middle
            PixelData<float> m(6, 6, 1, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {1, 1, 0,
                              1, 0, 0,
                              0, 0, 0};
            // put values in corners
            m(1, 1, 0) = 2;
            PixelData<float> grad;
            grad.initDownsampled(m, 0);
            ComputeGradient cg;
            cg.calc_bspline_fd_ds_mag(m, grad, 1, 1, 1);
            ASSERT_TRUE(compare(grad, expect, 0.01));
        }
        {   // One pixel image 1x1x1
            PixelData<float> m(1, 1, 1, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {0};
            // put values in corners
            m(0, 0, 0) = 2;
            PixelData<float> grad;
            grad.initDownsampled(m, 0);
            ComputeGradient cg;
            cg.calc_bspline_fd_ds_mag(m, grad, 1, 1, 1);
            ASSERT_TRUE(compare(grad, expect, 0.01));
        }

    }

    TEST(ComputeGradientTest, Corners3D) {
        PixelData<float> m(6, 6, 4, 0);
        // expect gradient is 3x3x2 X/Y/Z plane
        float expect[] = {1.73, 0, 5.19,
                          0, 0, 0,
                          3.46, 0, 6.92,

                          8.66, 0, 12.12,
                          0, 0, 0,
                          10.39, 0, 13.85};
        // put values in corners
        m(0, 0, 0) = 2;
        m(5, 0, 0) = 4;
        m(0, 5, 0) = 6;
        m(5, 5, 0) = 8;
        m(0, 0, 3) = 10;
        m(5, 0, 3) = 12;
        m(0, 5, 3) = 14;
        m(5, 5, 3) = 16;

        PixelData<float> grad;
        grad.initDownsampled(m, 0);
        ComputeGradient cg;
        cg.calc_bspline_fd_ds_mag(m, grad, 1, 1, 1);
        ASSERT_TRUE(compare(grad, expect, 0.05));
    }

    TEST(ComputeGradientTest, 2D_XY_BSPLINE_Y_DIR) {
        {   // values in corners and in middle
            PixelData<float> m(5, 7, 1, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {0.58, 0.00, 0.00, 0.08, 0.00, 0.00, 1.71,
                              0.56, 0.00, 0.00, 0.11, 0.00, 0.00, 1.51,
                              0.63, 0.00, 0.00, 0.11, 0.00, 0.00, 1.42,
                              0.88, 0.00, 0.00, 0.00, 0.00, 0.00, 1.75,
                              1.17, 0.00, 0.00, 0.00, 0.00, 0.00, 2.34};
            // put values in corners
            m(2, 3, 0) = 1;
            m(0, 0, 0) = 2;
            m(4, 0, 0) = 4;
            m(0, 6, 0) = 6;
            m(4, 6, 0) = 8;

            // Calculate bspline on CPU
            PixelData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_y(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect, 0.01));
        }
        {   // single point set in the middle
            PixelData<float> m(9, 3, 1, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {0.00, 0.01, 0.00,
                              0.00, 0.05, 0.00,
                              0.00, 0.12, 0.00,
                              0.00, 0.22, 0.00,
                              0.00, 0.28, 0.00,
                              0.00, 0.19, 0.00,
                              0.00, 0.08, 0.00,
                              0.00, 0.00, 0.00,
                              0.00, 0.00, 0.00};
            // put values in corners
            m(4, 1, 0) = 1;

            // Calculate bspline on CPU
            PixelData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_y(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect, 0.01));
        }
        {   // two pixel image 1x2x1
            PixelData<float> m(2, 1, 1, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {0,
                              0};
            // put values in corners
            m(0, 0, 0) = 1;

            // Calculate bspline on CPU
            PixelData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_y(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect, 0.01));
        }
    }

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_Y) {
        using ImgType = float;

        ImgType init[] =   {1.00, 0.00, 0.00,
                            1.00, 0.00, 6.00,
                            0.00, 6.00, 0.00,
                            6.00, 0.00, 0.00};

        ImgType expect[] = {1.00, 0.00, 2.00,
                            0.83, 1.00, 4.00,
                            1.17, 4.00, 1.00,
                            4.00, 2.00, 0.00};

        PixelData<ImgType> m(4, 3, 1);
        initFromZYXarray(m, init);

        // Calculate and compare
        ComputeGradient().calc_inv_bspline_y(m);
        ASSERT_TRUE(compare(m, expect, 0.01));
    }

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_X) {
        using ImgType = float;

        ImgType init[] =   {0.00, 6.00, 0.00,
                            1.00, 0.00, 0.00,
                            0.00, 0.00, 1.00};

        ImgType expect[] = {2.00, 4.00, 2.00,
                            0.67, 0.16, 0.00,
                            0.00, 0.16, 0.67};

        PixelData<ImgType> m(3, 3, 1);
        initFromZYXarray(m, init);

        // Calculate and compare
        ComputeGradient().calc_inv_bspline_x(m);
        ASSERT_TRUE(compare(m, expect, 0.01));
    }

    // ======================= CUDA =======================================
    // ======================= CUDA =======================================
    // ======================= CUDA =======================================

#ifdef APR_USE_CUDA

    TEST(ComputeGradientTest, 2D_XY_CUDA) {
        // Corner points
        PixelData<float> m(6, 6, 1, 0);
        // expect gradient is 3x3 X/Y plane
        float expect[] = {1.41, 0, 4.24,
                          0, 0, 0,
                          2.82, 0, 5.65};
        // put values in corners
        m(0, 0, 0) = 2;
        m(5, 0, 0) = 4;
        m(0, 5, 0) = 6;
        m(5, 5, 0) = 8;
        PixelData<float> grad;
        grad.initDownsampled(m, 0);
        cudaDownsampledGradient(m, grad, 1, 1, 1);
        ASSERT_TRUE(compare(grad, expect, 0.01));
    }

    TEST(ComputeGradientTest, Corners3D_CUDA) {
        PixelData<float> m(6, 6, 4, 0);
        // expect gradient is 3x3x2 X/Y/Z plane
        float expect[] = {1.73, 0, 5.19,
                          0, 0, 0,
                          3.46, 0, 6.92,

                          8.66, 0, 12.12,
                          0, 0, 0,
                          10.39, 0, 13.85};
        // put values in corners
        m(0, 0, 0) = 2;
        m(5, 0, 0) = 4;
        m(0, 5, 0) = 6;
        m(5, 5, 0) = 8;
        m(0, 0, 3) = 10;
        m(5, 0, 3) = 12;
        m(0, 5, 3) = 14;
        m(5, 5, 3) = 16;

        PixelData<float> grad;
        grad.initDownsampled(m, 0);
        cudaDownsampledGradient(m, grad, 1, 1, 1);
        ASSERT_TRUE(compare(grad, expect, 0.01));
    }

    TEST(ComputeGradientTest, GPU_VS_CPU_ON_RANDOM_VALUES) {
        // Generate random mesh
        // Generate random mesh
        using ImgType = float;
        PixelData<ImgType> m = getRandInitializedMesh<ImgType>(33, 31, 3);

        APRTimer timer(true);

        // Calculate gradient on CPU
        PixelData<ImgType> grad;
        grad.initDownsampled(m, 0);
        timer.start_timer("CPU gradient");
        ComputeGradient().calc_bspline_fd_ds_mag(m, grad, 1, 1, 1);
        timer.stop_timer();

        // Calculate gradient on GPU
        PixelData<ImgType> gradCuda;
        gradCuda.initDownsampled(m, 0);
        timer.start_timer("GPU gradient");
        cudaDownsampledGradient(m, gradCuda, 1, 1, 1);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(grad, gradCuda), 0);
    }

    TEST(ComputeBspineTest, BSPLINE_Y_DIR_CUDA) {
        APRTimer timer(true);

        // Generate random mesh
        using ImgType = float;
        PixelData<ImgType> m = getRandInitializedMesh<ImgType>(129,127,128);

        // Filter parameters
        const float lambda = 3;
        const float tolerance = 0.0001;

        // Calculate bspline on CPU
        PixelData<ImgType> mCpu(m, true);
        timer.start_timer("CPU bspline");
        ComputeGradient().bspline_filt_rec_y(mCpu, lambda, tolerance);
        timer.stop_timer();

        // Calculate bspline on GPU
        PixelData<ImgType> mGpu(m, true);
        timer.start_timer("GPU bspline");
        cudaFilterBsplineFull(mGpu, lambda, tolerance, BSPLINE_Y_DIR);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }

    TEST(ComputeBspineTest, BSPLINE_X_DIR_CUDA) {
        APRTimer timer(true);

        // Generate random mesh
        using ImgType = float;
        PixelData<ImgType> m = getRandInitializedMesh<ImgType>(129,127,128);

        // Filter parameters
        const float lambda = 3;
        const float tolerance = 0.0001;

        // Calculate bspline on CPU
        PixelData<ImgType> mCpu(m, true);
        timer.start_timer("CPU bspline");
        ComputeGradient().bspline_filt_rec_x(mCpu, lambda, tolerance);
        timer.stop_timer();

        // Calculate bspline on GPU
        PixelData<ImgType> mGpu(m, true);
        timer.start_timer("GPU bspline");
        cudaFilterBsplineFull(mGpu, lambda, tolerance, BSPLINE_X_DIR);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }

    TEST(ComputeBspineTest, BSPLINE_Z_DIR_CUDA) {
        APRTimer timer(true);

        // Generate random mesh
        using ImgType = float;
        PixelData<ImgType> m = getRandInitializedMesh<ImgType>(129,127,128);

        // Filter parameters
        const float lambda = 3;
        const float tolerance = 0.0001;

        // Calculate bspline on CPU
        PixelData<ImgType> mCpu(m, true);
        timer.start_timer("CPU bspline");
        ComputeGradient().bspline_filt_rec_z(mCpu, lambda, tolerance);
        timer.stop_timer();

        // Calculate bspline on GPU
        PixelData<ImgType> mGpu(m, true);
        timer.start_timer("GPU bspline");
        cudaFilterBsplineFull(mGpu, lambda, tolerance, BSPLINE_Z_DIR);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }

    TEST(ComputeBspineTest, BSPLINE_FULL_XYZ_DIR_CUDA) {
        APRTimer timer(true);

        // Generate random mesh
        using ImgType = float;
        PixelData<ImgType> m = getRandInitializedMesh<ImgType>(127, 128, 129);

        // Filter parameters
        const float lambda = 3;
        const float tolerance = 0.0001; // as defined in get_smooth_bspline_3D

        // Calculate bspline on CPU
        PixelData<ImgType> mCpu(m, true);
        timer.start_timer("CPU bspline");
        ComputeGradient().get_smooth_bspline_3D(mCpu, lambda);
        timer.stop_timer();

        // Calculate bspline on GPU
        PixelData<ImgType> mGpu(m, true);
        timer.start_timer("GPU bspline");
        cudaFilterBsplineFull(mGpu, lambda, tolerance, BSPLINE_ALL_DIR);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_Y_CUDA) {
        using ImgType = float;

        ImgType init[] =   {1.00, 0.00, 0.00,
                            1.00, 0.00, 6.00,
                            0.00, 6.00, 0.00,
                            6.00, 0.00, 0.00};

        ImgType expect[] = {1.00, 0.00, 2.00,
                            0.83, 1.00, 4.00,
                            1.17, 4.00, 1.00,
                            4.00, 2.00, 0.00};

        PixelData<ImgType> m(4, 3, 1);
        initFromZYXarray(m, init);

        // Calculate and compare
        m.printMesh(4,2);
        cudaInverseBspline(m, INV_BSPLINE_Y_DIR);
        m.printMesh(4,2);
        ASSERT_TRUE(compare(m, expect, 0.01));
    }

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_Y_RND_CUDA) {
        APRTimer timer(true);

        // Generate random mesh
        using ImgType = float;
        PixelData<ImgType> m = getRandInitializedMesh<ImgType>(127, 33, 31);

        // Calculate bspline on CPU
        PixelData<ImgType> mCpu(m, true);
        timer.start_timer("CPU inv bspline");
        ComputeGradient().calc_inv_bspline_y(mCpu);
        timer.stop_timer();

        // Calculate bspline on GPU
        PixelData<ImgType> mGpu(m, true);
        timer.start_timer("GPU inv bspline");
        cudaInverseBspline(mGpu,  INV_BSPLINE_Y_DIR);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_X_CUDA) {
        using ImgType = float;

        ImgType init[] =   {0.00, 6.00, 0.00,
                            1.00, 0.00, 0.00,
                            0.00, 0.00, 1.00};

        ImgType expect[] = {2.00, 4.00, 2.00,
                            0.67, 0.16, 0.00,
                            0.00, 0.16, 0.67};

        PixelData<ImgType> m(3, 3, 1);
        initFromZYXarray(m, init);

        // Calculate and compare
        m.printMesh(4,2);
        cudaInverseBspline(m, INV_BSPLINE_X_DIR);
        m.printMesh(4,2);
        ASSERT_TRUE(compare(m, expect, 0.01));
    }

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_X_RND_CUDA) {
        APRTimer timer(true);

        // Generate random mesh
        using ImgType = float;
        PixelData<ImgType> m = getRandInitializedMesh<ImgType>(127, 61, 66);

        // Calculate bspline on CPU
        PixelData<ImgType> mCpu(m, true);
        timer.start_timer("CPU inv bspline");
        ComputeGradient().calc_inv_bspline_x(mCpu);
        timer.stop_timer();

        // Calculate bspline on GPU
        PixelData<ImgType> mGpu(m, true);
        timer.start_timer("GPU inv bspline");
        cudaInverseBspline(mGpu,  INV_BSPLINE_X_DIR);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_Z_RND_CUDA) {
        APRTimer timer(true);

        // Generate random mesh
        using ImgType = float;
        PixelData<ImgType> m = getRandInitializedMesh<ImgType>(127, 61, 66);

        // Calculate bspline on CPU
        PixelData<ImgType> mCpu(m, true);
        timer.start_timer("CPU inv bspline");
        ComputeGradient().calc_inv_bspline_z(mCpu);
        timer.stop_timer();

        // Calculate bspline on GPU
        PixelData<ImgType> mGpu(m, true);
        timer.start_timer("GPU inv bspline");
        cudaInverseBspline(mGpu,  INV_BSPLINE_Z_DIR);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_FULL_XYZ_DIR_RND_CUDA) {
        APRTimer timer(true);

        // Generate random mesh
        using ImgType = float;
        PixelData<ImgType> m = getRandInitializedMesh<ImgType>(3,3,3,100);

        // Calculate bspline on CPU
        PixelData<ImgType> mCpu(m, true);
        timer.start_timer("CPU inv bspline");
        ComputeGradient().calc_inv_bspline_y(mCpu);
        ComputeGradient().calc_inv_bspline_x(mCpu);
        ComputeGradient().calc_inv_bspline_z(mCpu);
        timer.stop_timer();

        // Calculate bspline on GPU
        PixelData<ImgType> mGpu(m, true);
        timer.start_timer("GPU inv bspline");
        cudaInverseBspline(mGpu,  INV_BSPLINE_ALL_DIR);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }

    TEST(ComputeThreshold, CALC_THRESHOLD_RND_CUDA) {
        APRTimer timer(true);

        // Generate random mesh
        using ImgType = float;
        PixelData<ImgType> m = getRandInitializedMesh<ImgType>(31, 33, 13);
        PixelData<ImgType> g = getRandInitializedMesh<ImgType>(31, 33, 13);
        float thresholdLevel = 1;

        // Calculate bspline on CPU
        PixelData<ImgType> mCpu(g, true);
        timer.start_timer("CPU threshold");
        ComputeGradient().threshold_gradient(mCpu, m, thresholdLevel);

        timer.stop_timer();

        // Calculate bspline on GPU
        PixelData<ImgType> mGpu(g, true);
        timer.start_timer("GPU threshold");
        thresholdGradient(mGpu, m, thresholdLevel);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }

    TEST(ComputeThreshold, CALC_THRESHOLD_IMG_RND_CUDA) {
        APRTimer timer(true);

        // Generate random mesh
        using ImgType = float;
        PixelData<ImgType> g = getRandInitializedMesh<ImgType>(31, 33, 13, 1, true);

        float thresholdLevel = 10;

        // Calculate bspline on CPU
        PixelData<ImgType> mCpu(g, true);
        timer.start_timer("CPU threshold");
        for (size_t i = 0; i < mCpu.mesh.size(); ++i) {
            if (mCpu.mesh[i] <= (thresholdLevel)) { mCpu.mesh[i] = thresholdLevel; }
        }
        timer.stop_timer();

        // Calculate bspline on GPU
        PixelData<ImgType> mGpu(g, true);
        timer.start_timer("GPU threshold");
        thresholdImg(mGpu, thresholdLevel);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }

    TEST(ComputeThreshold, FULL_GRADIENT_TEST) {
        APRTimer timer(true);

        // Generate random mesh
        using ImageType = float;
        PixelData<ImageType> input_image = getRandInitializedMesh<ImageType>(310, 330, 13, 25);
        PixelData<ImageType> &image_temp = input_image;

        PixelData<ImageType> grad_temp; // should be a down-sampled image
        grad_temp.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, 0);
        PixelData<float> local_scale_temp; // Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors
        local_scale_temp.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num);
        PixelData<float> local_scale_temp2;
        local_scale_temp2.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num);

        PixelData<ImageType> grad_temp_GPU; // should be a down-sampled image
        grad_temp_GPU.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, 0);
        PixelData<float> local_scale_temp_GPU; // Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors
        local_scale_temp_GPU.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num);
        PixelData<float> local_scale_temp2_GPU;
        local_scale_temp2_GPU.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num);


        APRParameters par;
        par.lambda = 3;
        par.Ip_th = 10;
        par.dx = 1;
        par.dy = 1;
        par.dz = 1;

        // Calculate bspline on CPU
        PixelData<ImageType> mCpuImage(image_temp, true);
        timer.start_timer(">>>>>>>>>>>>>>>>> CPU gradient");
        APRConverter<float>().get_gradient(mCpuImage, grad_temp, local_scale_temp, local_scale_temp2, 0, par);
        timer.stop_timer();

        // Calculate bspline on GPU
        PixelData<ImageType> mGpuImage(image_temp, true);
        timer.start_timer(">>>>>>>>>>>>>>>>> GPU gradient");
        getGradient(mGpuImage, grad_temp_GPU, local_scale_temp_GPU, local_scale_temp2_GPU, 0, par);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpuImage, mGpuImage), 0);
        EXPECT_EQ(compareMeshes(grad_temp, grad_temp_GPU), 0);
        EXPECT_EQ(compareMeshes(local_scale_temp, local_scale_temp_GPU), 0);
    }

    TEST(ComputeThreshold, FULL_PIPELINE_TEST) {
        APRTimer timer(true);

        // Generate random mesh
        using ImageType = float;
        PixelData<ImageType> input_image = getRandInitializedMesh<ImageType>(310, 330, 32, 25);
        PixelData<ImageType> &image_temp = input_image;

        PixelData<ImageType> grad_temp; // should be a down-sampled image
        grad_temp.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, 0);
        PixelData<float> local_scale_temp; // Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors
        local_scale_temp.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num);
        PixelData<float> local_scale_temp2;
        local_scale_temp2.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num);

        PixelData<ImageType> grad_temp_GPU; // should be a down-sampled image
        grad_temp_GPU.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, 0);
        PixelData<float> local_scale_temp_GPU; // Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors
        local_scale_temp_GPU.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num);
        PixelData<float> local_scale_temp2_GPU;
        local_scale_temp2_GPU.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num);


        APRParameters par;
        par.lambda = 3;
        par.Ip_th = 10;
        par.sigma_th = 0;
        par.sigma_th_max = 0;
        par.dx = 1;
        par.dy = 1;
        par.dz = 1;

        // Calculate bspline on CPU
        PixelData<ImageType> mCpuImage(image_temp, true);
        timer.start_timer(">>>>>>>>>>>>>>>>> CPU PIPELINE");
        APRConverter<float>().get_gradient(mCpuImage, grad_temp, local_scale_temp, local_scale_temp2, 0, par);
        APRConverter<float>().get_local_intensity_scale(local_scale_temp, local_scale_temp2, par);
        timer.stop_timer();

        // Calculate bspline on GPU
        PixelData<ImageType> mGpuImage(image_temp, true);
        timer.start_timer(">>>>>>>>>>>>>>>>> GPU PIPELINE");
        getFullPipeline(mGpuImage, grad_temp_GPU, local_scale_temp_GPU, local_scale_temp2_GPU, 0, par);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpuImage, mGpuImage), 0);
        EXPECT_EQ(compareMeshes(grad_temp, grad_temp_GPU), 0);
        EXPECT_EQ(compareMeshes(local_scale_temp2, local_scale_temp2_GPU, 0.01), 0);
        EXPECT_EQ(compareMeshes(local_scale_temp, local_scale_temp_GPU, 0.01), 0);
    }


#endif // APR_USE_CUDA



    namespace {
        typedef struct {
            std::vector<float> bc1;
            std::vector<float> bc2;
            std::vector<float> bc3;
            std::vector<float> bc4;
            size_t k0;
            float b1;
            float b2;
            float norm_factor;
        } BsplineParams;

        float impulse_resp(float k, float rho, float omg) {
            //  Impulse Response Function
            return (pow(rho, (std::abs(k))) * sin((std::abs(k) + 1) * omg)) / sin(omg);
        }

        float impulse_resp_back(float k, float rho, float omg, float gamma, float c0) {
            //  Impulse Response Function (nominator eq. 4.8, denominator from eq. 4.7)
            return c0 * pow(rho, std::abs(k)) * (cos(omg * std::abs(k)) + gamma * sin(omg * std::abs(k))) *
                   (1.0 / (pow((1 - 2.0 * rho * cos(omg) + pow(rho, 2)), 2)));
        }

        template<typename T>
        BsplineParams prepareBsplineStuff(PixelData<T> &image, float lambda, float tol, int k0Len = -1) {
            // Recursive Filter Implimentation for Smoothing BSplines
            // B-Spline Signal Processing: Part II - Efficient Design and Applications, Unser 1993

            float xi = 1 - 96 * lambda + 24 * lambda * sqrt(3 + 144 * lambda); // eq 4.6
            float rho = (24 * lambda - 1 - sqrt(xi)) / (24 * lambda) *
                        sqrt((1 / xi) * (48 * lambda + 24 * lambda * sqrt(3 + 144 * lambda))); // eq 4.5
            float omg = atan(sqrt((1 / xi) * (144 * lambda - 1))); // eq 4.6

            float c0 = (1 + pow(rho, 2)) / (1 - pow(rho, 2)) * (1 - 2 * rho * cos(omg) + pow(rho, 2)) /
                       (1 + 2 * rho * cos(omg) + pow(rho, 2)); // eq 4.8
            float gamma = (1 - pow(rho, 2)) / (1 + pow(rho, 2)) * (1 / tan(omg)); // eq 4.8

            const float b1 = 2 * rho * cos(omg);
            const float b2 = -pow(rho, 2.0);

            const size_t idealK0Len = ceil(std::abs(log(tol) / log(rho)));
            const size_t minDimension = image.y_num;
            const size_t k0 = k0Len > 0 ? k0Len : std::min(idealK0Len, minDimension);

            const float norm_factor = pow((1 - 2.0 * rho * cos(omg) + pow(rho, 2)), 2);
            std::cout << "GPU: xi=" << xi << " rho=" << rho << " omg=" << omg << " gamma=" << gamma << " c0=" << c0 << " b1=" << b1 << " b2=" << b2 << " k0=" << k0 << " norm_factor=" << norm_factor << std::endl;

            // ------- Calculating boundary conditions

            // forward boundaries
            std::vector<float> impulse_resp_vec_f(k0 + 1);
            for (size_t k = 0; k < impulse_resp_vec_f.size(); ++k) impulse_resp_vec_f[k] = impulse_resp(k, rho, omg);

            //y(0) init
            std::vector<float> bc1(k0, 0);
            for (size_t k = 0; k < k0; ++k) bc1[k] = impulse_resp_vec_f[k];
            //y(1) init
            std::vector<float> bc2(k0, 0);
            bc2[1] = impulse_resp_vec_f[0];
            for (size_t k = 0; k < k0; ++k) bc2[k] += impulse_resp_vec_f[k + 1];

            // backward boundaries
            std::vector<float> impulse_resp_vec_b(k0 + 1);
            for (size_t k = 0; k < impulse_resp_vec_b.size(); ++k)
                impulse_resp_vec_b[k] = impulse_resp_back(k, rho, omg, gamma, c0);

            //y(N-1) init
            std::vector<float> bc3(k0, 0);
            bc3[0] = impulse_resp_vec_b[1];
            for (size_t k = 0; k < (k0 - 1); ++k) bc3[k + 1] += impulse_resp_vec_b[k] + impulse_resp_vec_b[k + 2];
            //y(N) init
            std::vector<float> bc4(k0, 0);
            bc4[0] = impulse_resp_vec_b[0];
            for (size_t k = 1; k < k0; ++k) bc4[k] += 2 * impulse_resp_vec_b[k];


            return BsplineParams{
                    bc1,
                    bc2,
                    bc3,
                    bc4,
                    k0,
                    b1,
                    b2,
                    norm_factor
            };
        }
    }

    template<typename T>
    void bspline_filt_rec_y2(PixelData<T>& image, BsplineParams p) {
        const size_t y_num = image.y_num;
        ArrayWrapper<T> &data = image.mesh;

        float temp1 = 0, temp2 = 0, temp3 = 0, temp4 = 0;
        for (size_t k = 0; k < p.k0; ++k) {
            temp1 += p.bc1[k] * data[k];
            temp2 += p.bc2[k] * data[k];
            temp3 += p.bc3[k] * data[y_num - 1 - k];
            temp4 += p.bc4[k] * data[y_num - 1 - k];
        }

        //initialize the sequence
        data[0] = temp1;
        data[1] = temp2;

        for (auto it = (data.begin() + 2); it != (data.begin() + y_num); ++it) {
            float  temp = temp2*p.b1 + temp1*p.b2 + *it;
            *it = temp;
            temp1 = temp2;
            temp2 = temp;
        }

        data[y_num - 2] = temp3 * p.norm_factor;
        data[y_num - 1] = temp4 * p.norm_factor;

        temp2 = temp3;
        temp1 = temp4;

        for (auto it = (data.begin() + y_num - 3); it != (data.begin() - 1); --it) {
            float temp = temp2*p.b1 + temp1*p.b2 + *it;
            *it = temp*p.norm_factor;
            temp1 = temp2;
            temp2 = temp;
        }
    }


//
//    namespace {
//        typedef struct {
//            std::vector<float> bc1;
//            std::vector<float> bc2;
//            std::vector<float> bc3;
//            std::vector<float> bc4;
//            size_t k0;
//            float b1;
//            float b2;
//            float norm_factor;
//        } BsplineParams2;
//
//        float impulse_resp2(float k, float rho, float omg) {
//            //  Impulse Response Function
//            return (pow(rho, (std::abs(k))) * sin((std::abs(k) + 1) * omg)) / sin(omg);
//        }
//
//        float impulse_resp_back2(float k, float rho, float omg, float gamma, float c0) {
//            //  Impulse Response Function (nominator eq. 4.8, denominator from eq. 4.7)
//            return c0 * pow(rho, std::abs(k)) * (cos(omg * std::abs(k)) + gamma * sin(omg * std::abs(k))) *
//                   (1.0 / (pow((1 - 2.0 * rho * cos(omg) + pow(rho, 2)), 2)));
//        }
//
//        template<typename T>
//        BsplineParams2 prepareBsplineStuff2(PixelData<T> &image, float lambda, float tol, int k0Len = -1) {
//            // Recursive Filter Implimentation for Smoothing BSplines
//            // B-Spline Signal Processing: Part II - Efficient Design and Applications, Unser 1993
//
//            float xi = 1 - 96 * lambda + 24 * lambda * sqrt(3 + 144 * lambda); // eq 4.6
//            float rho = (24 * lambda - 1 - sqrt(xi)) / (24 * lambda) *
//                        sqrt((1 / xi) * (48 * lambda + 24 * lambda * sqrt(3 + 144 * lambda))); // eq 4.5
//            float omg = atan(sqrt((1 / xi) * (144 * lambda - 1))); // eq 4.6
//
//            float c0 = (1 + pow(rho, 2)) / (1 - pow(rho, 2)) * (1 - 2 * rho * cos(omg) + pow(rho, 2)) /
//                       (1 + 2 * rho * cos(omg) + pow(rho, 2)); // eq 4.8
//            float gamma = (1 - pow(rho, 2)) / (1 + pow(rho, 2)) * (1 / tan(omg)); // eq 4.8
//
//            const float b1 = 2 * rho * cos(omg);
//            const float b2 = -pow(rho, 2.0);
//
//            const size_t idealK0Len = ceil(std::abs(log(tol) / log(rho)));
//            const size_t minDimension = image.y_num;
//            const size_t k0 = k0Len > 0 ? k0Len : std::min(idealK0Len, minDimension);
//
//            const float norm_factor = pow((1 - 2.0 * rho * cos(omg) + pow(rho, 2)), 2);
//            std::cout << "GPU: xi=" << xi << " rho=" << rho << " omg=" << omg << " gamma=" << gamma << " b1=" << b1 << " b2=" << b2 << " k0=" << k0 << " norm_factor=" << norm_factor << std::endl;
//
//            // ------- Calculating boundary conditions
//
//            // forward boundaries
//            std::vector<float> impulse_resp_vec_f(k0 + 1);
//            for (size_t k = 0; k < impulse_resp_vec_f.size(); ++k) impulse_resp_vec_f[k] = impulse_resp2(k, rho, omg);
//
//            //y(0) init
//            std::vector<float> bc1(k0, 0);
//            for (size_t k = 0; k < k0; ++k) bc1[k] = impulse_resp_vec_f[k];
//            //y(1) init
//            std::vector<float> bc2(k0, 0);
//            bc2[1] = impulse_resp_vec_f[0];
//            for (size_t k = 0; k < k0; ++k) bc2[k] += impulse_resp_vec_f[k + 1];
//
//            // backward boundaries
//            std::vector<float> impulse_resp_vec_b(k0 + 1);
//            for (size_t k = 0; k < impulse_resp_vec_b.size(); ++k)
//                impulse_resp_vec_b[k] = impulse_resp_back2(k, rho, omg, gamma, c0);
//
//            //y(N-1) init
//            std::vector<float> bc3(k0, 0);
//            bc3[0] = impulse_resp_vec_b[1];
//            for (size_t k = 0; k < (k0 - 1); ++k) bc3[k + 1] += impulse_resp_vec_b[k] + impulse_resp_vec_b[k + 2];
//            //y(N) init
//            std::vector<float> bc4(k0, 0);
//            bc4[0] = impulse_resp_vec_b[0];
//            for (size_t k = 1; k < k0; ++k) bc4[k] += 2 * impulse_resp_vec_b[k];
//
//
//            return BsplineParams2{
//                    bc1,
//                    bc2,
//                    bc3,
//                    bc4,
//                    k0,
//                    b1,
//                    b2,
//                    norm_factor
//            };
//        }
//    }
//
//    template<typename T>
//    void bspline_filt_rec_y3(PixelData<T>& image, BsplineParams2 p) {
//        const size_t y_num = image.y_num;
//        ArrayWrapper<T> &data = image.mesh;
//
//        float temp1 = 0, temp2 = 0, temp3 = 0, temp4 = 0;
//        for (size_t k = 0; k < p.k0; ++k) {
//            temp1 += p.bc1[k] * data[k];
//            temp2 += p.bc2[k] * data[k];
//            temp3 += p.bc3[k] * data[y_num - 1 - k];
//            temp4 += p.bc4[k] * data[y_num - 1 - k];
//        }
//
//        //initialize the sequence
//        data[0] = temp1;
//        data[1] = temp2;
//
//        for (auto it = (data.begin() + 2); it != (data.begin() + y_num); ++it) {
//            float  temp = temp2*p.b1 + temp1*p.b2 + *it;
//            *it = temp;
//            temp1 = temp2;
//            temp2 = temp;
//        }
//
//        std::cout << "[y_num - 2] = " << data[y_num - 2] << " vs " << temp3 << std::endl;
//        std::cout << "[y_num - 1] = " << data[y_num - 1] << " vs " << temp4 << std::endl;
//
//        data[y_num - 2] = temp3 * p.norm_factor;
//        data[y_num - 1] = temp4 * p.norm_factor;
//
//        temp2 = temp3;
//        temp1 = temp4;
//
//        for (auto it = (data.begin() + y_num - 3); it != (data.begin() - 1); --it) {
//            float temp = temp2*p.b1 + temp1*p.b2 + *it;
//            *it = temp*p.norm_factor;
//            temp1 = temp2;
//            temp2 = temp;
//        }
//    }


    namespace {
        typedef struct {
            std::vector<float> bc1;
            std::vector<float> bc2;
            std::vector<float> bc3;
            std::vector<float> bc4;
            size_t k0;
            float b1;
            float b2;
            float norm_factor;
        } BsplineParams2;

        float impulse_resp2(float k, float rho, float omg) {
            //  Impulse Response Function
            return (pow(rho, (std::abs(k))) * sin((std::abs(k) + 1) * omg)) / sin(omg);
        }

        float impulse_resp_back2(float k, float rho, float omg, float gamma, float c0) {
            //  Impulse Response Function (nominator eq. 4.8, denominator from eq. 4.7)
            return c0 * pow(rho, std::abs(k)) * (cos(omg * std::abs(k)) + gamma * sin(omg * std::abs(k))) *
                   (1.0 / (pow((1 - 2.0 * rho * cos(omg) + pow(rho, 2)), 2)));
        }

        template<typename T>
        BsplineParams2 prepareBsplineStuff2(PixelData<T> &image, float lambda, float tol, int k0Len = -1) {
            // Recursive Filter Implimentation for Smoothing BSplines
            // B-Spline Signal Processing: Part II - Efficient Design and Applications, Unser 1993

            float xi = 1 - 96 * lambda + 24 * lambda * sqrt(3 + 144 * lambda); // eq 4.6
            float rho = (24 * lambda - 1 - sqrt(xi)) / (24 * lambda) *
                        sqrt((1 / xi) * (48 * lambda + 24 * lambda * sqrt(3 + 144 * lambda))); // eq 4.5
            float omg = atan(sqrt((1 / xi) * (144 * lambda - 1))); // eq 4.6

            float c0 = (1 + pow(rho, 2)) / (1 - pow(rho, 2)) * (1 - 2 * rho * cos(omg) + pow(rho, 2)) /
                       (1 + 2 * rho * cos(omg) + pow(rho, 2)); // eq 4.8
            float gamma = (1 - pow(rho, 2)) / (1 + pow(rho, 2)) * (1 / tan(omg)); // eq 4.8

            const float b1 =  2 * rho * cos(omg);
            const float b2 = -pow(rho, 2.0);

            const size_t idealK0Len = ceil(std::abs(log(tol) / log(rho)));
            const size_t minDimension = image.y_num;
            const size_t k0 = k0Len > 0 ? k0Len : std::min(idealK0Len, minDimension);

            const float norm_factor = (1 - 2.0 * rho * cos(omg) + pow(rho, 2));
            std::cout << "GPU: xi=" << xi << " rho=" << rho << " omg=" << omg << " gamma=" << gamma << " b1=" << b1 << " b2=" << b2 << " k0=" << k0 << " norm_factor=" << norm_factor << std::endl;

            // ------- Calculating boundary conditions

            // forward boundaries
            std::vector<float> impulse_resp_vec_f(k0 + 1);
            for (size_t k = 0; k < impulse_resp_vec_f.size(); ++k) impulse_resp_vec_f[k] = impulse_resp2(k, rho, omg);

            //y(0) init
            std::vector<float> bc1(k0, 0);
            for (size_t k = 0; k < k0; ++k) bc1[k] = impulse_resp_vec_f[k];
            //y(1) init
            std::vector<float> bc2(k0, 0);
            bc2[1] = impulse_resp_vec_f[0];
            for (size_t k = 0; k < k0; ++k) bc2[k] += impulse_resp_vec_f[k + 1];

            // backward boundaries
            std::vector<float> impulse_resp_vec_b(k0 + 1);
            for (size_t k = 0; k < impulse_resp_vec_b.size(); ++k)
                impulse_resp_vec_b[k] = impulse_resp_back2(k, rho, omg, gamma, c0);

            //y(N-1) init
            std::vector<float> bc3(k0, 0);
            bc3[0] = impulse_resp_vec_b[1];
            for (size_t k = 0; k < (k0 - 1); ++k) bc3[k + 1] += impulse_resp_vec_b[k] + impulse_resp_vec_b[k + 2];
            //y(N) init
            std::vector<float> bc4(k0, 0);
            bc4[0] = impulse_resp_vec_b[0];
            for (size_t k = 1; k < k0; ++k) bc4[k] += 2 * impulse_resp_vec_b[k];


            return BsplineParams2{
                    bc1,
                    bc2,
                    bc3,
                    bc4,
                    k0,
                    b1,
                    b2,
                    norm_factor
            };
        }
    }

    template <typename T>
    void matNumbers(const PixelData<T> &pd, int aColumnWidth, int aFloatPrecision) {
        for (size_t y = 0; y < pd.y_num; ++y) {
            std::__1::cout << std::__1::setw(aColumnWidth) << std::__1::setprecision(aFloatPrecision) << std::__1::fixed << pd.at(y, 0, 0) << " ";
        }
    }

    template <typename T>
    void matprint(std::string n, PixelData<T> &pd) {
        int aColumnWidth = 5;
        int aFloatPrecision = 3;
        std::cout << n << " = [";
        matNumbers(pd, aColumnWidth, aFloatPrecision);
        std::cout << "];\n";

    }


    template <typename T>
    void matprint(std::string n, std::vector<T> &pd) {
        int aColumnWidth = 5;
        int aFloatPrecision = 4;
        std::cout << n << " = [";
        for (size_t y = 0; y < pd.size(); ++y) {
            std::cout << std::setw(aColumnWidth) << std::setprecision(aFloatPrecision) << std::fixed << pd[y] << " ";
        }
        std::cout << "];\n";
    }

    template<typename T>
    void bspline_filt_rec_y3(PixelData<T>& image, BsplineParams2 p) {

        auto &ext = image;
        const size_t y_num = ext.y_num;

        ArrayWrapper<T> &data = ext.mesh;
        {
            float z1 = data[0], z2 = data[0];
            for (auto it = (data.begin() + 0); it != (data.begin() + y_num); ++it) {
                float temp = *it * p.norm_factor + p.b1 * z1 + p.b2 * z2;
                *it = temp;
                z2 = z1;
                z1 = temp;
            }
        }
        {
            float z1 = data[y_num - 1], z2 = data[y_num - 1];
            for (auto it = (data.begin() + y_num - 1); it != (data.begin() - 1); --it) {
                float temp = *it * p.norm_factor + p.b1 * z1 + p.b2 * z2;
                *it = temp;
                z2 = z1;
                z1 = temp;
            }
        }
    }

    template<typename T>
    void bspline_filt_rec_y3(PixelData<T>& image, BsplineParams2 p, int ghost) {

        PixelData<T> ext(image.y_num + 2 * ghost, 1, 1, 3);
        for (int64_t i = -ghost; i < (int64_t)image.y_num + ghost; ++i) {
            if (i < 0)
                ext.mesh[i + ghost] = image.mesh[0 - i];
            else if (i < image.y_num)
                ext.mesh[i + ghost] = image.mesh[i];
            else
                ext.mesh[i + ghost] = image.mesh[image.y_num - 1 - (i - (image.y_num - 1))];
        }

        const size_t y_num = ext.y_num;
        ArrayWrapper<T> &data = ext.mesh;
        {
            float z1 = data[0], z2 = data[0];
            for (auto it = (data.begin() + 0); it != (data.begin() + y_num); ++it) {
                float temp = *it * p.norm_factor + p.b1 * z1 + p.b2 * z2;
                *it = temp;
                z2 = z1;
                z1 = temp;
            }
        }
        {
            float z1 = data[y_num - 1], z2 = data[y_num - 1];
            for (auto it = (data.begin() + y_num - 1); it != (data.begin() - 1); --it) {
                float temp = *it * p.norm_factor + p.b1 * z1 + p.b2 * z2;
                *it = temp;
                z2 = z1;
                z1 = temp;
            }
        }

        for (int i = 0; i < image.y_num; ++i) {
            image.mesh[i] = ext.mesh[i + ghost];
        }
    }



    TEST(ComputeGradientTest, TESTTEST) {
        {   // values in corners and in middle
            int size = 155;
//            PixelData<float> m(size, 1, 1, 0);
            PixelData<float> m = getRandInitializedMesh<float>(size, 1, 1, 20);

            // put values in corners
//            for (int i = 0; i < size; ++i) {
//                m.mesh[i] = 3*sin(i/5 + 4) * ((float)size - i/1.1) / size / 2 + 3;
//                m.mesh[i] += 2* ((i ) % 12 == 0) + 1 + (float)i / size;
//            }



//            for (int i = 0; i < 44 ; ++i) m.mesh[i] = 8 * ((i /9 ) % 2 == 0) -2;
//
//            for (int i = 0; i < 12; ++i) m.mesh[i] = 4* std::abs(7-i);
//            for (int i = 44; i < size; ++i) m.mesh[i] += 15 * sin(i/8);
//            for (int i = 0; i < size; ++i) m.mesh[i] += 5;
//            m.mesh[0] = 5;
//            m.mesh[1] = 5;
//



//            for (int i = 0 ; i < size ; ++i) m.mesh[i] = i > 75 ? 1 : -1;
//            m.mesh[0] = 1;
//            m.mesh[size-1] = -1;

            float lambda = 3;
            float tol = 0.000001;

            // Calculate bspline on CPU
            PixelData<float> mCpu1(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_y(mCpu1, lambda, tol);

            PixelData<float> mCpu2(m, true);
            auto p = prepareBsplineStuff(mCpu2, lambda, tol);
            bspline_filt_rec_y2(mCpu2, p);

            PixelData<float> mCpu3(m, true);
            auto p2 = prepareBsplineStuff2(mCpu3, lambda, tol);
            bspline_filt_rec_y3(mCpu3, p2);
            std::cout << "\n\nallGhost=[";
            for (int i = 0; i < 20; ++i) {
                PixelData<float> mCpu4(m, true);
                bspline_filt_rec_y3(mCpu4, p2, i);
                matNumbers(mCpu4, 5, 3);
                std::cout << ";\n";
            }
            std::cout << "];\n";

            matprint("y", m);
            matprint("p", mCpu3);
            matprint("n", mCpu2);

            EXPECT_EQ(compareMeshes(mCpu2, mCpu1, 2), 0);
        }
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
