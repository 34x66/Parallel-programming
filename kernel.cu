#include <iostream>
#include <omp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <array>
#include <opencv2/opencv.hpp>

#define BLOCK_SIZE (16u)
#define FILTER_SIZE (5u)
#define TILE_SIZE (12u) // BLOCK_SIZE - 2 * (FILTER_SIZE / 2)
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
    size_t pitch, unsigned int width, unsigned int height) {
    int x_o = (TILE_SIZE * blockIdx.x) + threadIdx.x;
    int y_o = (TILE_SIZE * blockIdx.y) + threadIdx.y;

    int x_i = x_o - (FILTER_SIZE / 2);
    int y_i = y_o - (FILTER_SIZE / 2);

    __shared__ unsigned char sBuffer[BLOCK_SIZE][BLOCK_SIZE];

    if ((x_i >= 0) && (x_i < width) && (y_i >= 0) && (y_i < height))
        sBuffer[threadIdx.y][threadIdx.x] = in[y_i * pitch + x_i];
    else
        sBuffer[threadIdx.y][threadIdx.x] = 0;

    __syncthreads();

    int sum = 0;
    if ((threadIdx.x < TILE_SIZE) && (threadIdx.y < TILE_SIZE)) {
        for (int r = 0; r < FILTER_SIZE; ++r)
            for (int c = 0; c < FILTER_SIZE; ++c)
                sum += sBuffer[threadIdx.y + r][threadIdx.x + c];


        sum = sum / (FILTER_SIZE * FILTER_SIZE);  
        if (x_o < width && y_o < height)
            out[y_o * width + x_o] = sum;
    }

}

void processImageChannel(unsigned char* d_input, unsigned char* d_output,
    unsigned char* host_input, unsigned char* host_output,
    size_t width, size_t height, size_t host_step) {
    size_t size = width * height * sizeof(unsigned char);
    size_t pitch;

    CUDA_CHECK_RETURN(cudaMallocPitch(&d_input, &pitch, width, height));
    CUDA_CHECK_RETURN(cudaMalloc(&d_output, size));

    CUDA_CHECK_RETURN(cudaMemcpy2D(d_input, pitch, host_input, host_step,
        width, height, cudaMemcpyHostToDevice));

    dim3 grid_size((width + TILE_SIZE - 1) / TILE_SIZE,
        (height + TILE_SIZE - 1) / TILE_SIZE);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    processImg << <grid_size, blockSize >> > (d_output, d_input, pitch, width, height);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    CUDA_CHECK_RETURN(cudaMemcpy(host_output, d_output, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(d_input));
    CUDA_CHECK_RETURN(cudaFree(d_output));
}

int main() {
	std::cout << "Loading image..." << std::endl;

    cv::Mat img = cv::imread("test_image.png", cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    unsigned int width = img.cols;
    unsigned int height = img.rows;
    unsigned int size = width * height * sizeof(unsigned char);

    unsigned char* host_red = (unsigned char *) malloc (size);
    unsigned char* host_green = (unsigned char *) malloc (size);
    unsigned char* host_blue = (unsigned char *) malloc (size);

    cv::Mat channels[3];
    cv::split(img, channels);

    unsigned char* d_r = NULL;
    unsigned char* d_g = NULL;
    unsigned char* d_b = NULL;
    unsigned char* d_r_n = NULL;
    unsigned char* d_g_n = NULL;
    unsigned char* d_b_n = NULL;

    processImageChannel(d_r, d_r_n, channels[2].data, host_red,
        width, height, channels[2].step);
    processImageChannel(d_g, d_g_n, channels[1].data, host_green,
        width, height, channels[1].step);
    processImageChannel(d_b, d_b_n, channels[0].data, host_blue,
        width, height, channels[0].step);
    
    cv::Mat output_img(height, width, CV_8UC3);
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            output_img.at<cv::Vec3b>(i, j)[0] = host_blue[i * width + j];
            output_img.at<cv::Vec3b>(i, j)[1] = host_green[i * width + j];
            output_img.at<cv::Vec3b>(i, j)[2] = host_red[i * width + j];
        }
    }
    
    cv::imwrite("filtred_image.png", output_img);
    
    free(host_blue);
    free(host_green);
    free(host_red);

	return 0;
}