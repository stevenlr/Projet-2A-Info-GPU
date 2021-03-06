#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <image/image.h>
#include <image/tga.h>

#include "../../CudaBench.h"

#define PARTSIZE 4

__global__ void invert(uint8_t *data)
{
	int thread = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;
	uint8_t *ptr = data + thread * PARTSIZE;
	uint8_t values[PARTSIZE];

#pragma unroll PARTSIZE
	for (int i = 0; i < PARTSIZE; ++i) {
		values[i] = ptr[i];
	}

#pragma unroll PARTSIZE
	for (int i = 0; i < PARTSIZE; ++i) {
		values[i] = 255 - values[i];
	}

#pragma unroll PARTSIZE
	for (int i = 0; i < PARTSIZE; ++i) {
		ptr[i] = values[i];
	}
}

__constant__ __device__ unsigned int full = 0xffffffff;

__global__ void invertSIMD(unsigned int *data)
{
	int thread = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;
	unsigned int *ptr = data + thread;

	*ptr = __vsubss4(full, *ptr);
}

int main(int argc, char *argv[])
{
	if (argc != 3) {
		printf("Invalid number of arguments.\n");
		return 1;
	}

	Image *input_image;
	Image *output_image;
	int error;

	if ((error = TGA_readImage(argv[1], &input_image)) != 0) {
		printf("Error when opening image: %d\n", error);
		return 1;
	}

	if ((error = Image_copy(input_image, &output_image)) != 0) {
		printf("Error when copying image: %d\n", error);
		Image_delete(input_image);
		return 1;
	}

	CudaBench allBench, sendBench, retrieveBench, kernelBench;
	allBench = CudaBench_new();
	sendBench = CudaBench_new();
	retrieveBench = CudaBench_new();
	kernelBench = CudaBench_new();

	int c, size, sizeDevice, sizePadding;
	uint8_t *c_data;
	
	size = input_image->width * input_image->height * sizeof(uint8_t);

	int threadsPerBlock = 512;
	dim3 blocks(input_image->width / 64, input_image->height / 32, 1);

	sizePadding = threadsPerBlock * PARTSIZE;
	
	if (size % sizePadding == 0)
		sizeDevice = size;
	else
		sizeDevice = size + sizePadding - (size % sizePadding);

	CudaBench_start(allBench);
	cudaMalloc(&c_data, sizeDevice);

	for (c = 0; c < input_image->channels; ++c) {
		CudaBench_start(sendBench);
		cudaMemcpy(c_data, input_image->data[c], size, cudaMemcpyHostToDevice);
		CudaBench_end(sendBench);

		CudaBench_start(kernelBench);
		//invert<<<blocks, threadsPerBlock>>>(c_data);
		invertSIMD<<<blocks, threadsPerBlock>>>((unsigned int *) c_data);
		CudaBench_end(kernelBench);

		CudaBench_start(retrieveBench);
		cudaMemcpy(output_image->data[c], c_data, size, cudaMemcpyDeviceToHost);
		CudaBench_end(retrieveBench);
	}

	cudaFree(c_data);
	CudaBench_end(allBench);

	cudaEventSynchronize(allBench.end);

	float timeAll, timeSend, timeKernel, timeRetrieve;

	timeAll = CudaBench_elapsedTime(allBench);
	timeSend = CudaBench_elapsedTime(sendBench);
	timeRetrieve = CudaBench_elapsedTime(retrieveBench);
	timeKernel = CudaBench_elapsedTime(kernelBench);

	printf("All: %f ms\nSend: %f ms\nRetrieve: %f ms\nKernel: %f ms\n", timeAll, timeSend, timeRetrieve, timeKernel);

	CubaBench_delete(allBench);
	CubaBench_delete(sendBench);
	CubaBench_delete(retrieveBench);
	CubaBench_delete(kernelBench);

	if ((error = TGA_writeImage(argv[2], output_image)) != 0) {
		printf("Error when writing image: %d\n", error);
	}

	Image_delete(input_image);
	Image_delete(output_image);

	cudaDeviceReset();

	return 0;
}