
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <math.h>

using namespace std;
using namespace cv;

static const bool ENABLE_GRAYSCALE_CONVERSION = true;
static const bool ENABLE_GAUSSIAN_BLUR_CONVERSION = true; 
// Radius gaussian filter
static const int SIGMA = 1;
// Half floating window gaussian blur
static const int HALF_WIN = 3 * SIGMA; 

__device__ int indexByteFromCol(int baseCol, int row, const int width)
{
    return (baseCol + row * width) * 3;
}

__device__ int indexByteFromRow(int baseRow, int col)
{
    return (baseRow + col) * 3;
}

// Constants (Always)
// [0] - Width image
// [1] - Height image
// [2] - Half_floating_win 

__global__ void ConversionToGrayscale(char * image, const int * constants)
{
    uchar * bytes = (uchar *)image;
    // Get row
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int widthImage = constants[0];
    const int heightImage = constants[1];
    int offsetRow = row * widthImage;
    
    // Convert to grayscale 0.21 R + 0.72 G + 0.07 B. OpenCV R <-> B
    if(row < heightImage)
    for(int col = 0; col < widthImage; col++)
    {  
            uchar res = bytes[indexByteFromRow(offsetRow, col)] * 0.07f + bytes[indexByteFromRow(offsetRow, col) + 1] * 0.72f + bytes[indexByteFromRow(offsetRow, col) + 2] * 0.21f;
            bytes[indexByteFromRow(offsetRow, col)] = res;
            bytes[indexByteFromRow(offsetRow, col) + 1] = res;
            bytes[indexByteFromRow(offsetRow, col) + 2] = res;
    }
}

__global__ void ConversionGaussianHor(char * image, const int * constants, const float * floating_window)
{
    uchar * bytes = (uchar *)image;
    // Get row
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int widthImage = constants[0];
    const int heightImage = constants[1];
    int offsetRow = row * widthImage;

    // Gaussian filter
    int half_win = constants[2];
    uchar temp_buf[(HALF_WIN + 1) * 3];
    float sum = 0.0f; // for normalize 
    uint r = 0, g = 0, b = 0;

    // Floating window string processing algorithm
    // TODO? (In the case of grayscale, one color component is sufficient)
    if(row < heightImage)
    {
        for(int col = 0; col < widthImage; col++)
        {
            sum = 0.0f;
            r = g = b = 0;
            for(int i = -half_win; i < half_win + 1 ; i++ )
            {
                int index_pix = col + i;
                if(index_pix >= 0 && index_pix < widthImage) 
                {
                    b += bytes[indexByteFromRow(offsetRow, index_pix)] * floating_window[i + half_win];
                    g += bytes[indexByteFromRow(offsetRow, index_pix) + 1] * floating_window[i + half_win];
                    r += bytes[indexByteFromRow(offsetRow, index_pix) + 2] * floating_window[i + half_win];
                    sum += floating_window[i + half_win];
                }
            }

            // If the window leaves the reading zone of the first pixels of the row, 
            // we write the oldest result to the global memory
            if(col > HALF_WIN) 
            {
                bytes[indexByteFromRow(offsetRow, col - HALF_WIN - 1)] = temp_buf[0];
                bytes[indexByteFromRow(offsetRow, col - HALF_WIN - 1) + 1] = temp_buf[1];
                bytes[indexByteFromRow(offsetRow, col - HALF_WIN - 1) + 2] = temp_buf[2];       
            }
                    
            // Shift the queue
            for(int j = 0; j < HALF_WIN; j++)
            {
                temp_buf[j * 3] = temp_buf[(j + 1) * 3];
                temp_buf[j * 3 + 1] = temp_buf[(j + 1) * 3 + 1];
                temp_buf[j * 3 + 2] = temp_buf[(j + 1) * 3 + 2];
            } 

            // At the end write the result
            temp_buf[HALF_WIN * 3] = b / sum;
            temp_buf[HALF_WIN * 3 + 1] = g / sum;
            temp_buf[HALF_WIN * 3 + 2] = r / sum;     
        }

            // Record end
        for(int j = 0; j < HALF_WIN + 1; j++)
        {
            bytes[indexByteFromRow(offsetRow, widthImage - HALF_WIN - 1 + j)] = temp_buf[j * 3];
            bytes[indexByteFromRow(offsetRow, widthImage - HALF_WIN - 1 + j) + 1] = temp_buf[j * 3 + 1];
            bytes[indexByteFromRow(offsetRow, widthImage - HALF_WIN - 1 + j) + 2] = temp_buf[j * 3 + 2];
        }
    }
}

__global__ void ConversionGaussianVer(char * image, const int * constants, const float * floating_window)
{
    uchar * bytes = (uchar *)image;
    // Get col
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int widthImage = constants[0];
    const int heightImage = constants[1];
    int offsetCol = col;

    // Gaussian filter
    int half_win = constants[2];  
    uchar temp_buf[(HALF_WIN + 1) * 3];
    float sum = 0.0f; // for normalize 
    uint r = 0, g = 0, b = 0;

    // Floating window column processing algorithm
    // TODO? (In the case of grayscale, one color component is sufficient)
    if(col < widthImage)
    {
        for(int row = 0; row < heightImage; row++)
        {
            sum = 0.0f;
            r = g = b = 0;
            for(int i = -half_win; i < half_win + 1 ; i++ )
            {
                int index_pix = row + i;
                if(index_pix >= 0 && index_pix < heightImage) 
                {
                    b += bytes[indexByteFromCol(offsetCol, index_pix, widthImage)] * floating_window[i + half_win];
                    g += bytes[indexByteFromCol(offsetCol, index_pix, widthImage) + 1] * floating_window[i + half_win];
                    r += bytes[indexByteFromCol(offsetCol, index_pix, widthImage) + 2] * floating_window[i + half_win];
                    sum += floating_window[i + half_win];
                }
            }

            // If the window leaves the reading zone of the first pixels of the row, 
            // we write the oldest result to the global memory
            if(row > HALF_WIN) 
            {
                bytes[indexByteFromCol(offsetCol, row - HALF_WIN - 1, widthImage)] = temp_buf[0];
                bytes[indexByteFromCol(offsetCol, row - HALF_WIN - 1, widthImage) + 1] = temp_buf[1];
                bytes[indexByteFromCol(offsetCol, row - HALF_WIN - 1, widthImage) + 2] = temp_buf[2];                
            }
                    
            // Shift the queue
            for(int j = 0; j < HALF_WIN; j++)
            {
                temp_buf[j * 3] = temp_buf[(j + 1) * 3];
                temp_buf[j * 3 + 1] = temp_buf[(j + 1) * 3 + 1];
                temp_buf[j * 3 + 2] = temp_buf[(j + 1) * 3 + 2];
            } 

            // At the end write the result
            temp_buf[HALF_WIN * 3] = b / sum;
            temp_buf[HALF_WIN * 3 + 1] = g / sum;
            temp_buf[HALF_WIN * 3 + 2] = r / sum;        
        }

        // Record end
        for(int j = 0; j < HALF_WIN + 1; j++)
        {
            bytes[indexByteFromCol(offsetCol, constants[1] - HALF_WIN - 1 + j, widthImage)] = temp_buf[j * 3];
            bytes[indexByteFromCol(offsetCol, constants[1] - HALF_WIN - 1 + j, widthImage) + 1] = temp_buf[j * 3 + 1];
            bytes[indexByteFromCol(offsetCol, constants[1] - HALF_WIN - 1 + j, widthImage) + 2] = temp_buf[j * 3 + 2];
        }
    }
}

Mat Conversion (Mat &image)
{
    int constWidth[] {image.cols, image.rows, HALF_WIN};
    char * out_Image = 0;
    char * dev_Image = 0;
    int * dev_Constant = 0; // width height half_floating_win
    float * dev_FloatWindow = 0; //    
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) 
    {
        cout << "cudaSetDevice error" << endl;     
    }

    // depth = 8 bit, channels = 3
    int sizeDataImage = image.rows * image.cols * 3;

    // --------- floating window gaussian filter ----------
    int s2 = 2.0f * SIGMA * SIGMA;

    float float_win[2 * HALF_WIN + 1];
    float_win[HALF_WIN] = 1.0f; // Center = 1f

    // window calculation
    for(int i = 1; i <= HALF_WIN; i++)
    {
        float_win[HALF_WIN + i] = exp(-1.0f * i * i / s2);
        float_win[HALF_WIN - i] = float_win[HALF_WIN + i];
    }
    // ---------------------------------------------------
    
    cudaMalloc((void**)&dev_Image, sizeDataImage * sizeof(char));
    cudaMalloc((void**)&dev_Constant, sizeof(constWidth));
    cudaMalloc((void**)&dev_FloatWindow, sizeof(float_win));
 
    cudaMemcpy(dev_Image, image.ptr(), sizeDataImage * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Constant, constWidth, sizeof(constWidth), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_FloatWindow, float_win, sizeof(float_win), cudaMemcpyHostToDevice);

    // Count thread = Max side image
    if(ENABLE_GRAYSCALE_CONVERSION)
    {
        ConversionToGrayscale  <<<image.rows / 512 + 1, 512>>> (dev_Image, dev_Constant);
        cudaStatus = cudaDeviceSynchronize();
    }

    if(ENABLE_GAUSSIAN_BLUR_CONVERSION)
    {
        ConversionGaussianHor  <<<image.rows / 512 + 1, 512>>> (dev_Image, dev_Constant, dev_FloatWindow);
        cudaStatus = cudaDeviceSynchronize();
        ConversionGaussianVer  <<<image.cols / 512 + 1, 512>>> (dev_Image, dev_Constant, dev_FloatWindow);
        cudaStatus = cudaDeviceSynchronize();
    }

    // malloc without deleting, move to Mat
    out_Image = (char*)std::malloc(sizeDataImage * sizeof(char));
    cudaStatus = cudaMemcpy(out_Image, dev_Image, sizeDataImage * sizeof(char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
    {
        cout << "cudaMemcpy failed!" << endl; 
    }

    cudaFree(dev_Image);
    cudaFree(dev_Constant);
    cudaFree(dev_FloatWindow);
    return Mat(image.rows, image.cols, CV_8UC3, out_Image);  
} 

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << "Error. Need the path to the image file" << endl;
		return -1;
	}

	// Read image
    Mat image;
    Mat resImage;
	image = imread(argv[1], IMREAD_COLOR); 
	if (image.empty()) 
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
    }
    
    if(image.rows < (HALF_WIN * 2 + 2) || image.cols < (HALF_WIN * 2 + 2))
	{
		cout << "Image Too Small" << std::endl;
		return -1;
    }

	cout << "Load completed" << endl;
	cout << "Conversion..." << endl;

    resImage = Conversion(image);

	cout << "Conversion completed" << endl;
	cout << "Save..." << endl;

	// Save image
	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	try
	{
		imwrite("out.png", resImage, compression_params);
	}
	catch (const cv::Exception & ex)
	{
        cout << "Exception converting image to PNG format: " << ex.what() << endl;
        return -1;
    }
    
	cout << "Save completed" << endl;
    return 0;
}
