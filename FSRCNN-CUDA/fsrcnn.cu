#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "fsrcnn.h"
#include "fsrcnn_params.h"

FSRCNN::FSRCNN() {
    isInit = false;

    Mat mat;
    cuda::GpuMat mat_d(mat);

    checkCUDNN(cudnnCreate(&cudnnHandle));
    checkCUDNN(cudnnCreateTensorDescriptor(&inTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&outTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 5; k++) {
                feature_extraction_block_w[i][0][j][k] = feature_extraction_block_feature_extraction_w_x2[32 * (5 * j + k) + i];
            }
        }
    }
    checkCUDA(cudaMalloc((void**)&feature_extraction_block_w_d, sizeof(feature_extraction_block_w)));
    checkCUDA(cudaMemcpy(feature_extraction_block_w_d, feature_extraction_block_w, sizeof(feature_extraction_block_w), cudaMemcpyHostToDevice));

    for (int i = 0; i < 32; i++) {
        feature_extraction_block_b[i] = feature_extraction_block_feature_extraction_b_x2[i];
    }
    checkCUDA(cudaMalloc((void**)&feature_extraction_block_b_d, sizeof(feature_extraction_block_b)));
    checkCUDA(cudaMemcpy(feature_extraction_block_b_d, feature_extraction_block_b, sizeof(feature_extraction_block_b), cudaMemcpyHostToDevice));

    for (int i = 0; i < 32; i++) {
        feature_extraction_block_alpha[i] = shrinking_block_alpha1_x2[i];
    }
    checkCUDA(cudaMalloc((void**)&feature_extraction_block_alpha_d, sizeof(feature_extraction_block_alpha)));
    checkCUDA(cudaMemcpy(feature_extraction_block_alpha_d, feature_extraction_block_alpha, sizeof(feature_extraction_block_alpha), cudaMemcpyHostToDevice));

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 32; j++) {
            shrinking_block_w[i][j][0][0] = shrinking_block_shrinking_w_x2[5 * j + i];          
        }
    }
    checkCUDA(cudaMalloc((void**)&shrinking_block_w_d, sizeof(shrinking_block_w)));
    checkCUDA(cudaMemcpy(shrinking_block_w_d, shrinking_block_w, sizeof(shrinking_block_w), cudaMemcpyHostToDevice));

    for (int i = 0; i < 5; i++) {
        shrinking_block_b[i] = shrinking_block_shrinking_b_x2[i];
    }
    checkCUDA(cudaMalloc((void**)&shrinking_block_b_d, sizeof(shrinking_block_b)));
    checkCUDA(cudaMemcpy(shrinking_block_b_d, shrinking_block_b, sizeof(shrinking_block_b), cudaMemcpyHostToDevice));  

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    mapping_block_w1[i][j][k][l] = mapping_block_w3_x2[25 * (3 * k + l) + 5 * j + i];                   
                }              
            }
        }
    }
    checkCUDA(cudaMalloc((void**)&mapping_block_w1_d, sizeof(mapping_block_w1)));
    checkCUDA(cudaMemcpy(mapping_block_w1_d, mapping_block_w1, sizeof(mapping_block_w1), cudaMemcpyHostToDevice));

    for (int i = 0; i < 5; i++) {
        mapping_block_b1[i] = mapping_block_b3_x2[i];
    }
    checkCUDA(cudaMalloc((void**)&mapping_block_b1_d, sizeof(mapping_block_b1)));
    checkCUDA(cudaMemcpy(mapping_block_b1_d, mapping_block_b1, sizeof(mapping_block_b1), cudaMemcpyHostToDevice));

    for (int i = 0; i < 5; i++) {
        mapping_block_alpha1[i] = mapping_block_alpha4_x2[i];
    }
    checkCUDA(cudaMalloc((void**)&mapping_block_alpha1_d, sizeof(mapping_block_alpha1)));
    checkCUDA(cudaMemcpy(mapping_block_alpha1_d, mapping_block_alpha1, sizeof(mapping_block_alpha1), cudaMemcpyHostToDevice));

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            mapping_block_w2[i][j][0][0] = mapping_block_w4_x2[5 * j + i];
        }
    }
    checkCUDA(cudaMalloc((void**)&mapping_block_w2_d, sizeof(mapping_block_w2)));
    checkCUDA(cudaMemcpy(mapping_block_w2_d, mapping_block_w2, sizeof(mapping_block_w2), cudaMemcpyHostToDevice));

    for (int i = 0; i < 5; i++) {
        mapping_block_b2[i] = mapping_block_b4_x2[i];
    }
    checkCUDA(cudaMalloc((void**)&mapping_block_b2_d, sizeof(mapping_block_b2)));
    checkCUDA(cudaMemcpy(mapping_block_b2_d, mapping_block_b2, sizeof(mapping_block_b2), cudaMemcpyHostToDevice));

    for (int i = 0; i < 5; i++) {
        mapping_block_alpha2[i] = alpha2_x2[i];
    }
    checkCUDA(cudaMalloc((void**)&mapping_block_alpha2_d, sizeof(mapping_block_alpha2)));
    checkCUDA(cudaMemcpy(mapping_block_alpha2_d, mapping_block_alpha2, sizeof(mapping_block_alpha2), cudaMemcpyHostToDevice));

    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 5; j++) {
            expanding_block_w[i][j][0][0] = expanding_block_w5_x2[32 * j + i];
        }
    }
    checkCUDA(cudaMalloc((void**)&expanding_block_w_d, sizeof(expanding_block_w)));
    checkCUDA(cudaMemcpy(expanding_block_w_d, expanding_block_w, sizeof(expanding_block_w), cudaMemcpyHostToDevice));

    for (int i = 0; i < 32; i++) {
        expanding_block_b[i] = expanding_block_b5_x2[i];
    }
    checkCUDA(cudaMalloc((void**)&expanding_block_b_d, sizeof(expanding_block_b)));
    checkCUDA(cudaMemcpy(expanding_block_b_d, expanding_block_b, sizeof(expanding_block_b), cudaMemcpyHostToDevice));

    for (int i = 0; i < 32; i++) {
        expanding_block_alpha[i] = expanding_block_alpha5_x2[i];
    }
    checkCUDA(cudaMalloc((void**)&expanding_block_alpha_d, sizeof(expanding_block_alpha)));
    checkCUDA(cudaMemcpy(expanding_block_alpha_d, expanding_block_alpha, sizeof(expanding_block_alpha), cudaMemcpyHostToDevice));

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 32; j++) {
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    deconvolution_block_w[i][j][k][l] = deconvolution_block_deconv_w_x2[128 * (3 * k + l) + 4 * j + i];
                }
            }
        }
    }
    checkCUDA(cudaMalloc((void**)&deconvolution_block_w_d, sizeof(deconvolution_block_w)));
    checkCUDA(cudaMemcpy(deconvolution_block_w_d, deconvolution_block_w, sizeof(deconvolution_block_w), cudaMemcpyHostToDevice));

    for (int i = 0; i < 4; i++) {
        deconvolution_block_b[i] = deconvolution_block_deconv_b_x2[i];
    }
    checkCUDA(cudaMalloc((void**)&deconvolution_block_b_d, sizeof(deconvolution_block_b)));
    checkCUDA(cudaMemcpy(deconvolution_block_b_d, deconvolution_block_b, sizeof(deconvolution_block_b), cudaMemcpyHostToDevice));
};

FSRCNN::~FSRCNN() {
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(inTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(outTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDesc));
    checkCUDNN(cudnnDestroy(cudnnHandle));

    checkCUDA(cudaFree(feature_extraction_block_w_d));
    checkCUDA(cudaFree(feature_extraction_block_b_d));
    checkCUDA(cudaFree(feature_extraction_block_alpha_d));
    checkCUDA(cudaFree(shrinking_block_w_d));
    checkCUDA(cudaFree(shrinking_block_b_d));
    checkCUDA(cudaFree(mapping_block_w1_d));
    checkCUDA(cudaFree(mapping_block_b1_d));
    checkCUDA(cudaFree(mapping_block_alpha1_d));
    checkCUDA(cudaFree(mapping_block_w2_d));
    checkCUDA(cudaFree(mapping_block_b2_d));
    checkCUDA(cudaFree(mapping_block_alpha2_d));
    checkCUDA(cudaFree(expanding_block_w_d));
    checkCUDA(cudaFree(expanding_block_b_d));
    checkCUDA(cudaFree(expanding_block_alpha_d));
    checkCUDA(cudaFree(deconvolution_block_w_d));
    checkCUDA(cudaFree(deconvolution_block_b_d));

    finish();
};

FSRCNN* FSRCNN::instance = nullptr;

void FSRCNN::checkCUDNN(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS)
        cout << "[ERROR] CUDNN " << status << endl;
}

void FSRCNN::checkCUDA(cudaError_t error) {
    if (error != CUDA_SUCCESS)
        cout << "[ERROR] CUDA " << error << endl;
}

Mat FSRCNN::sr(const Mat& img) {
    int h = img.rows;
    int w = img.cols;

    preprocessImg(img);
    feature_extraction(h, w);
    shrinking(h, w);
    mapping1(h, w);
    mapping2(h, w);
    expanding(h, w);
    deconvolution(h, w);
    return postProcessImg(img);
}

void FSRCNN::init(int h, int w) {
    input = new float[(h + 4) * (w + 4)];   

    checkCUDA(cudaMalloc((void**)&input_d, _msize(input)));
    checkCUDA(cudaMalloc((void**)&output_d, sizeof(float) * h * w * 4));
    checkCUDA(cudaMalloc((void**)&conv1, sizeof(float) * h * w * 32));
    checkCUDA(cudaMalloc((void**)&conv2, sizeof(float) * h * w * 5));
    checkCUDA(cudaMalloc((void**)&conv3, sizeof(float) * h * w * 5));
    checkCUDA(cudaMalloc((void**)&conv4, sizeof(float) * h * w * 5));
    checkCUDA(cudaMalloc((void**)&conv5, sizeof(float) * h * w * 32));
    checkCUDA(cudaMalloc((void**)&conv6, sizeof(float) * h * w * 4));

    isInit = true;
}

void FSRCNN::finish() {
    if (isInit == false) return;

    delete[] input;

    checkCUDA(cudaFree(input_d));
    checkCUDA(cudaFree(output_d));
    checkCUDA(cudaFree(conv1));
    checkCUDA(cudaFree(conv2));
    checkCUDA(cudaFree(conv3));
    checkCUDA(cudaFree(conv4));
    checkCUDA(cudaFree(conv5));
    checkCUDA(cudaFree(conv6));

    isInit = false;
}

void FSRCNN::feature_extraction(int h, int w) {  
    conv2D(input_d, feature_extraction_block_w_d, conv1, { 1, 1, h + 4, w + 4 }, { 32, 1, 5, 5 }, 0, CUDNN_TENSOR_NHWC);
    biasAdd(conv1, feature_extraction_block_b_d, { 1, 32, h, w }, 32);
    pRelu(conv1, feature_extraction_block_alpha_d, h * w * 32, 32);
}

void FSRCNN::shrinking(int h, int w) {
    conv2D(conv1, shrinking_block_w_d, conv2, { 1, 32, h, w }, { 5, 32, 1, 1 }, 0, CUDNN_TENSOR_NCHW);
    biasAdd(conv2, shrinking_block_b_d, { 1, 5, h, w }, 5);
}

void FSRCNN::mapping1(int h, int w) {
    conv2D(conv2, mapping_block_w1_d, conv3, { 1, 5, h, w }, { 5, 5, 3, 3 }, 1, CUDNN_TENSOR_NCHW);
    biasAdd(conv3, mapping_block_b1_d, { 1, 5, h, w }, 5);
    pRelu(conv3, mapping_block_alpha1_d, h * w * 5, 5);
}

void FSRCNN::mapping2(int h, int w) {
    conv2D(conv3, mapping_block_w2_d, conv4, { 1, 5, h, w }, { 5, 5, 1, 1 }, 0, CUDNN_TENSOR_NCHW);
    biasAdd(conv4, mapping_block_b2_d, { 1, 5, h, w }, 5);
    addWithCuda(conv4, conv2, h * w * 5);
    pRelu(conv4, mapping_block_alpha2_d, h * w * 5, 5);
}

void FSRCNN::expanding(int h, int w) {
    conv2D(conv4, expanding_block_w_d, conv5, { 1, 5, h, w }, { 32, 5, 1, 1 }, 0, CUDNN_TENSOR_NCHW);
    biasAdd(conv5, expanding_block_b_d, { 1, 32, h, w }, 32);  
    pRelu(conv5, expanding_block_alpha_d, h * w * 32, 32);
}

void FSRCNN::deconvolution(int h, int w) {
    conv2D(conv5, deconvolution_block_w_d, conv6, { 1, 32, h, w }, { 4, 32, 3, 3 }, 1, CUDNN_TENSOR_NCHW);
    biasAdd(conv6, deconvolution_block_b_d, { 1, 4, h, w }, 4);
    depth2Space(output_d, conv6, { 1, 4, h, w }); 
}

__global__ void addKernel(float* dst, const float* src, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        dst[i] += src[i];
    }
}

void FSRCNN::addWithCuda(float* dst, const float* src, int size) {  
    addKernel << <(size + 1) / 1024, 1024 >> > (dst, src, size);
    cudaDeviceSynchronize();
}

void FSRCNN::conv2D(float* src, float* filter, float* dst, const array<int, 4>& src_dim, const array<int, 4>& filter_dim, int padding, cudnnTensorFormat_t format) {
    //초기화
    checkCUDNN(cudnnSetTensor4dDescriptor(inTensorDesc, format, CUDNN_DATA_FLOAT, src_dim[0], src_dim[1], src_dim[2], src_dim[3]));
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filter_dim[0], filter_dim[1], filter_dim[2], filter_dim[3]));

    //컨볼루션의 패딩, 스트라이드, 컨볼루션 모드 등을 셋팅
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding, padding, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT)); // 패딩, 패딩, 스트라이드, 스트라이드, 1, 1

    int out_n, out_c, out_h, out_w;
    //입력데이터를 위에서 셋팅한 대로 컨볼루션 했을때 출력 데이터의 구조 알아내기
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, inTensorDesc, filterDesc, &out_n, &out_c, &out_h, &out_w));

    //출력 데이터의 자료형, 구조를 셋팅
    checkCUDNN(cudnnSetTensor4dDescriptor(outTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));

    //입력과 필터, 컨볼루션 패딩, 스트라이드가 위와 같이 주어졌을때 가장 빠르게 계산할 수 있는 알고리즘이 무엇인지를 알아내기
    cudnnConvolutionFwdAlgo_t algo;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
        inTensorDesc,
        filterDesc,
        convDesc,
        outTensorDesc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &algo
    ));

    //위에서 알아낸 가장 빠른 알고리즘을 사용할 경우 계산과정에서 필요한 버퍼 데이터의 크기를 알아내기
    size_t sizeInBytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
        inTensorDesc,
        filterDesc,
        convDesc,
        outTensorDesc,
        algo,
        &sizeInBytes));

    //계산과정에서 버퍼 데이터가 필요하다면 메모리 할당
    void* workSpace;//CUDNN이 작업 중에 사용할 버퍼 메모리
    checkCUDA(cudaMalloc(&workSpace, sizeInBytes));

    float alpha = 1;
    float beta = 0;

    //컨볼루션 시작
    checkCUDNN(cudnnConvolutionForward(cudnnHandle,
        &alpha,
        inTensorDesc,
        src,
        filterDesc,
        filter,
        convDesc,
        algo,
        workSpace,
        sizeInBytes,
        &beta,
        outTensorDesc,
        dst));

    //메모리 해제
    checkCUDA(cudaFree(workSpace));
}

void FSRCNN::biasAdd(float* src, float* bias, const array<int, 4>& src_dim, int bias_dim) {
    //초기화
    checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, bias_dim, 1, 1));

    //출력 데이터의 자료형, 구조를 셋팅
    checkCUDNN(cudnnSetTensor4dDescriptor(outTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, src_dim[0], src_dim[1], src_dim[2], src_dim[3]));

    float alpha = 1;
    float beta = 1;

    checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, biasTensorDesc, bias, &beta, outTensorDesc, src));
}

__global__ void pReluKernel(float* dst, const float* alpha, int size, int unit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size && dst[i] < 0) {
        dst[i] = dst[i] * alpha[i / (size / unit)];
    }
}

void FSRCNN::pRelu(float* dst, const float* alpha, int size, int unit) {
    pReluKernel << <(size + 1) / 1024, 1024 >> > (dst, alpha, size, unit);
    cudaDeviceSynchronize();
}

__global__ void depth2SpaceKernal(float* dst, const float* src) {
    int i = threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.x;
    int h = gridDim.y;
    int w = gridDim.x;

    dst[(2 * w) * (2 * j + i / 2) + (2 * k + i % 2)] = src[h * w * i + w * j + k] * 255;
}

void FSRCNN::depth2Space(float* dst, const float* src, const array<int, 4>& src_dim) {
    dim3 blocks(src_dim[3], src_dim[2]);
    depth2SpaceKernal << <blocks, src_dim[1] >> > (dst, src);
    cudaDeviceSynchronize();
}

void FSRCNN::paddingImg(cuda::GpuMat& src, cuda::GpuMat& dst, int top, int bottom, int left, int right) {
    left = (left > 0 ? left : 0);
    right = (right > 0 ? right : 0);
    top = (top > 0 ? top : 0);
    bottom = (bottom > 0 ? bottom : 0);
    cuda::copyMakeBorder(src, dst, top, bottom, left, right, BORDER_REFLECT);
}

void FSRCNN::preprocessImg(const Mat& src_img) {
    src_img_d.upload(src_img);

    cuda::GpuMat img_y_cb_cr, y_img_d1, y_img_d2;
    cuda::cvtColor(src_img_d, img_y_cb_cr, COLOR_BGR2YCrCb);
    vector<cuda::GpuMat> img_y_cb_cr_channels(3);
    cuda::split(img_y_cb_cr, img_y_cb_cr_channels);
    paddingImg(img_y_cb_cr_channels[0], y_img_d1, 2, 2, 2, 2);
    //transform datatype from uchar to float
    y_img_d1.convertTo(y_img_d2, CV_32FC1);

    Mat y_img;
    y_img_d2.download(y_img);

    fromMat2Tenser4D(y_img, input);
    checkCUDA(cudaMemcpy(input_d, input, _msize(input), cudaMemcpyHostToDevice));
}

Mat FSRCNN::postProcessImg(const Mat& src_img) {    
    cuda::GpuMat f_img_d1(2 * src_img.rows, 2 * src_img.cols, CV_32F, output_d);
    cuda::GpuMat f_img_d2;
    f_img_d1.convertTo(f_img_d2, CV_8U);

    cuda::GpuMat img_y_cb_cr, dest_d;
    cuda::cvtColor(src_img_d, img_y_cb_cr, COLOR_BGR2YCrCb);

    vector<cuda::GpuMat> img_y_cb_cr_channels(3);
    cuda::split(img_y_cb_cr, img_y_cb_cr_channels);

    cuda::resize(img_y_cb_cr_channels[1], img_y_cb_cr_channels[1], { src_img_d.cols * 2, src_img_d.rows * 2 }, 0, 0, INTER_CUBIC);
    cuda::resize(img_y_cb_cr_channels[2], img_y_cb_cr_channels[2], { src_img_d.cols * 2, src_img_d.rows * 2 }, 0, 0, INTER_CUBIC);

    vector<cuda::GpuMat> mv = { f_img_d2, img_y_cb_cr_channels[1], img_y_cb_cr_channels[2] };
    cuda::merge(mv, dest_d);
    
    cuda::cvtColor(dest_d, dest_d, COLOR_YCrCb2BGR);

    Mat dest;
    dest_d.download(dest);

    return dest;
}

void FSRCNN::fromMat2Tenser4D(const Mat& src_img, float* dst) {
    for (int i = 0; i < src_img.rows; i++) {      
        for (int j = 0; j < src_img.cols; j++) {
            dst[src_img.cols * i + j] = src_img.at<float>(i, j) / 255.0;
        }
    }
}

extern "C" {
    __declspec(dllexport) FSRCNN* FSRCNN_construct();
    __declspec(dllexport) void FSRCNN_init(FSRCNN* ptr, int h , int w);
    __declspec(dllexport) Mat* FSRCNN_sr(FSRCNN* ptr, int row, int col, int* data);
    __declspec(dllexport) void FSRCNN_finish(FSRCNN* ptr);
}

FSRCNN* FSRCNN_construct() {
    return FSRCNN::getInstance();
}

void FSRCNN_init(FSRCNN* ptr, int h, int w) {
    ptr->init(h, w);
}

Mat* FSRCNN_sr(FSRCNN* ptr, int row, int col, int* data) {
    Mat src_img(row, col, CV_8UC3, data);
    return new Mat(ptr->sr(src_img));
}

void FSRCNN_finish(FSRCNN* ptr) {
    ptr->finish();
}