#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <image/image.h>
#include <image/tga.h>

#include "../../CudaBench.h"

#define PARTSIZE 4

__global__ void add(uint8_t *data1, uint8_t *data2)
{
	int thread = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;
	uint8_t *ptr1 = data1 + thread * PARTSIZE;
	uint8_t *ptr2 = data2 + thread * PARTSIZE;
	uint16_t value[2 * PARTSIZE];

#pragma unroll PARTSIZE
	for (int i = 0; i < PARTSIZE; ++i) {
		value[i] = ptr1[i];
		value[PARTSIZE + i] = ptr2[i];
	}

#pragma unroll PARTSIZE
	for (int i = 0; i < PARTSIZE; ++i) {
		value[i] += value[PARTSIZE + i];
	}

#pragma unroll PARTSIZE
	for (int i = 0; i < PARTSIZE; ++i) {
		value[i] = min(value[i], 255);
	}

#pragma unroll PARTSIZE
	for (int i = 0; i < PARTSIZE; ++i) {
		ptr1[i] = value[i];
	}
}

__global__ void addSIMD(unsigned int *data1, unsigned int *data2)
{
	int thread = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;
	unsigned int *ptr1 = data1 + thread;
	unsigned int *ptr2 = data2 + thread;

	*ptr1 = __vaddus4(*ptr1, *ptr2);
}

int main(int argc, char *argv[])
{
	if (argc != 4) {
		printf("Invalid number of arguments.\n");
		return 1;
	}

	Image *input_image1;
	Image *input_image2;
	Image *output_image;
	int error;

	if ((error = TGA_readImage(argv[1], &input_image1)) != 0) {
		printf("Error when opening image: %d\n", error);
		return 1;
	}

	if ((error = TGA_readImage(argv[2], &input_image2)) != 0) {
		printf("Error when opening image: %d\n", error);
		return 1;
	}

	if (input_image1->width != input_image2->width
		|| input_image1->height != input_image2->height
		|| input_image1->channels != input_image2->channels) {
		printf("Input images dimensions differ\n");
		Image_delete(input_image1);
		Image_delete(input_image2);
		return 1;
	}

	if ((error = Image_copy(input_image1, &output_image)) != 0) {
		printf("Error when copying image: %d\n", error);
		Image_delete(input_image1);
		Image_delete(input_image2);
		return 1;
	}

	CudaBench allBench, sendBench, retrieveBench, kernelBench;
	allBench = CudaBench_new();
	sendBench = CudaBench_new();
	retrieveBench = CudaBench_new();
	kernelBench = CudaBench_new();

	int c, size, sizeDevice, sizePadding;
	uint8_t *c_data1, *c_data2;
	
	size = input_image1->width * input_image1->height * sizeof(uint8_t);

	int threadsPerBlock = 256;
	dim3 blocks(input_image1->width / 32, input_image1->height / 32, 1);

	sizePadding = threadsPerBlock * PARTSIZE;
	
	if (size % sizePadding == 0)
		sizeDevice = size;
	else
		sizeDevice = size + sizePadding - (size % sizePadding);

	CudaBench_start(allBench);
	cudaMalloc(&c_data1, sizeDevice);
	cudaMalloc(&c_data2, sizeDevice);

	for (c = 0; c < input_image1->channels; ++c) {
		CudaBench_start(sendBench);
		cudaMemcpy(c_data1, input_image1->data[c], size, cudaMemcpyHostToDevice);
		cudaMemcpy(c_data2, input_image2->data[c], size, cudaMemcpyHostToDevice);
		CudaBench_end(sendBench);

		CudaBench_start(kernelBench);
		//add<<<blocks, threadsPerBlock>>>(c_data1, c_data2);
		addSIMD<<<blocks, threadsPerBlock>>>((unsigned int *) c_data1, (unsigned int *) c_data2);
		CudaBench_end(kernelBench);

		CudaBench_start(retrieveBench);
		cudaMemcpy(output_image->data[c], c_data1, size, cudaMemcpyDeviceToHost);
		CudaBench_end(retrieveBench);
	}

	cudaFree(c_data1);
	cudaFree(c_data2);
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

	if ((error = TGA_writeImage(argv[3], output_image)) != 0) {
		printf("Error when writing image: %d\n", error);
	}

	Image_delete(input_image1);
	Image_delete(input_image2);
	Image_delete(output_image);

	cudaDeviceReset();

	return 0;
}