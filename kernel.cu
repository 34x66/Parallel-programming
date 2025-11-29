#include <iostream>
#include <omp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <array>
#include <vector>
#include <opencv2/opencv.hpp>

#define CUDA_CHECK_RETURN(value)                                  \
    {                                                             \
        cudaError_t err = value;                                  \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "Error %s at line %d in file %s\n",   \
                    cudaGetErrorString(err), __LINE__, __FILE__); \
            exit(-1);                                             \
        }                                                         \
    }

__global__ void processImg(unsigned char* out, unsigned char* in,
    size_t pitch, unsigned int width, unsigned int height,
    unsigned int filter_size, unsigned int tile_size, unsigned int block_size) {

    int x_o = (tile_size * blockIdx.x) + threadIdx.x;
    int y_o = (tile_size * blockIdx.y) + threadIdx.y;

    int x_i = x_o - (filter_size / 2);
    int y_i = y_o - (filter_size / 2);

    extern __shared__ unsigned char sBuffer[];

    if ((x_i >= 0) && (x_i < width) && (y_i >= 0) && (y_i < height))
        sBuffer[threadIdx.y * block_size + threadIdx.x] = in[y_i * pitch + x_i];
    else
        sBuffer[threadIdx.y * block_size + threadIdx.x] = 0;

    __syncthreads();

    int sum = 0;
    if ((threadIdx.x < tile_size) && (threadIdx.y < tile_size)) {
        for (int r = 0; r < filter_size; ++r)
            for (int c = 0; c < filter_size; ++c)
                sum += sBuffer[(threadIdx.y + r) * block_size + (threadIdx.x + c)];

        sum = sum / (filter_size * filter_size);
        if (x_o < width && y_o < height)
            out[y_o * width + x_o] = sum;
    }
}

void processImageChannel(unsigned char* d_input, unsigned char* d_output,
    unsigned char* host_input, unsigned char* host_output,
    size_t width, size_t height, size_t host_step,
    unsigned int block_size, unsigned int filter_size, unsigned int tile_size) {

    size_t size = width * height * sizeof(unsigned char);
    size_t pitch;

    CUDA_CHECK_RETURN(cudaMallocPitch(&d_input, &pitch, width, height));
    CUDA_CHECK_RETURN(cudaMalloc(&d_output, size));

    CUDA_CHECK_RETURN(cudaMemcpy2D(d_input, pitch, host_input, host_step,
        width, height, cudaMemcpyHostToDevice));

    dim3 grid_size((width + tile_size - 1) / tile_size,
        (height + tile_size - 1) / tile_size);
    dim3 block_size_dim(block_size, block_size);

    size_t shared_mem_size = block_size * block_size * sizeof(unsigned char);

    processImg << <grid_size, block_size_dim, shared_mem_size >> >
        (d_output, d_input, pitch, width, height, filter_size, tile_size, block_size);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    CUDA_CHECK_RETURN(cudaMemcpy(host_output, d_output, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(d_input));
    CUDA_CHECK_RETURN(cudaFree(d_output));
}

double applyBlurFilter(const cv::Mat& input, cv::Mat& output,
    unsigned int block_size, unsigned int filter_size, unsigned int tile_size) {
    double start_time = omp_get_wtime();

    unsigned int width = input.cols;
    unsigned int height = input.rows;
    unsigned int size = width * height * sizeof(unsigned char);

    unsigned char* host_red = (unsigned char*)malloc(size);
    unsigned char* host_green = (unsigned char*)malloc(size);
    unsigned char* host_blue = (unsigned char*)malloc(size);

    cv::Mat channels[3];
    cv::split(input, channels);

    unsigned char* d_r = NULL;
    unsigned char* d_g = NULL;
    unsigned char* d_b = NULL;
    unsigned char* d_r_n = NULL;
    unsigned char* d_g_n = NULL;
    unsigned char* d_b_n = NULL;

    processImageChannel(d_r, d_r_n, channels[2].data, host_red,
        width, height, channels[2].step, block_size, filter_size, tile_size);
    processImageChannel(d_g, d_g_n, channels[1].data, host_green,
        width, height, channels[1].step, block_size, filter_size, tile_size);
    processImageChannel(d_b, d_b_n, channels[0].data, host_blue,
        width, height, channels[0].step, block_size, filter_size, tile_size);

    output.create(height, width, CV_8UC3);
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            output.at<cv::Vec3b>(i, j)[0] = host_blue[i * width + j];
            output.at<cv::Vec3b>(i, j)[1] = host_green[i * width + j];
            output.at<cv::Vec3b>(i, j)[2] = host_red[i * width + j];
        }
    }

    free(host_blue);
    free(host_green);
    free(host_red);

    double end_time = omp_get_wtime();
    return end_time - start_time;
}

int main() {
    std::cout << "Loading image" << std::endl;

    cv::Mat img = cv::imread("test_image.png", cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    std::cout << "Processing image" << std::endl;

    cv::Mat output_img;

    std::vector<unsigned int> block_sizes = { 8, 16, 32, 64, 128 };
    unsigned int filter_size = 5;
    for (unsigned int block_size : block_sizes) {
        unsigned int tile_size = block_size - 2 * (filter_size / 2);
        std::cout << "block_size :" << block_size << " " << "tile_size: " << tile_size << std::endl;
        double execution_time = applyBlurFilter(img, output_img, block_size, filter_size, tile_size);
        std::cout << "Execution time: " << execution_time << " seconds" << std::endl;
        // std::string filename = "filtred_image_block_" + std::to_string(block_size) + ".png";
        // cv::imwrite(filename, output_img);
    }
    return 0;
}