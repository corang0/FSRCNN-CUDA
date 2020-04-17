#pragma once

#include <iostream>
#include <cudnn.h>
#include <cuda.h>
#include <array>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class FSRCNN
{
private:
    FSRCNN();
    ~FSRCNN();
    void checkCUDNN(cudnnStatus_t status);
    void checkCUDA(cudaError_t error);
    void feature_extraction(int h, int w);
    void shrinking(int h, int w);
    void mapping1(int h, int w);
    void mapping2(int h, int w);
    void expanding(int h, int w);
    void deconvolution(int h, int w);
    void addWithCuda(float* dst, const float* src, int size);
    void conv2D(float* src, float* filter, float* dst, const array<int, 4>& src_dim, const array<int, 4>& filter_dim, int padding, cudnnTensorFormat_t format);
    void biasAdd(float* src, float* bias, const array<int, 4>& src_dim, int bias_dim);
    void pRelu(float* dst, const float* alpha, int size, int unit);
    void depth2Space(float* dst, const float* src, const array<int, 4>& src_dim);
    void paddingImg(cuda::GpuMat& src, cuda::GpuMat& dst, int top, int bottom, int left, int right);
    void preprocessImg(const Mat& src_img);
    Mat postProcessImg(const Mat& src_img);
    void fromMat2Tenser4D(const Mat& src_img, float* dst);

    static FSRCNN* instance;
    bool isInit;

    // CUDNN 배열
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t inTensorDesc, outTensorDesc, biasTensorDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;

    // 컨볼루션1 (feature extraction)
    float feature_extraction_block_w[32][1][5][5];
    float* feature_extraction_block_w_d;
    float feature_extraction_block_b[32];
    float* feature_extraction_block_b_d;
    float feature_extraction_block_alpha[32];
    float* feature_extraction_block_alpha_d;

    // 컨볼루션2 (shrinking)
    float shrinking_block_w[5][32][1][1];
    float* shrinking_block_w_d;
    float shrinking_block_b[5];
    float* shrinking_block_b_d;

    // 컨볼루션3 (mapping1)
    float mapping_block_w1[5][5][3][3];
    float* mapping_block_w1_d;
    float mapping_block_b1[5];
    float* mapping_block_b1_d;
    float mapping_block_alpha1[5];
    float* mapping_block_alpha1_d;

    // 컨볼루션4 (mapping2)
    float mapping_block_w2[5][5][1][1];
    float* mapping_block_w2_d;
    float mapping_block_b2[5];
    float* mapping_block_b2_d;
    float mapping_block_alpha2[5];
    float* mapping_block_alpha2_d;

    // 컨볼루션5 (expanding)
    float expanding_block_w[32][5][1][1];
    float* expanding_block_w_d;
    float expanding_block_b[32];
    float* expanding_block_b_d;
    float expanding_block_alpha[32];
    float* expanding_block_alpha_d;

    // 컨볼루션6 (deconvolution)
    float deconvolution_block_w[4][32][3][3];
    float* deconvolution_block_w_d;
    float deconvolution_block_b[4];
    float* deconvolution_block_b_d;

    // 작업 결과
    cuda::GpuMat src_img_d;
    float* input;        // 입력 (host)
    float* input_d;     // 입력 (device)
    float* conv1;       // feature_extraction 결과
    float* conv2;       // shrinking 결과
    float* conv3;       // mapping1 결과
    float* conv4;       // mapping2 결과
    float* conv5;       // expanding 결과
    float* conv6;       // deconvolution 결과
    float* output_d;   // Depth2Space 결과

public:
    static FSRCNN* getInstance() {
        if (instance == nullptr) instance = new FSRCNN();
        return instance;
    }
    Mat sr(const Mat& img);
    void init(int h, int w);
    void finish();
};