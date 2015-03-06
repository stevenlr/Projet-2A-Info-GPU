#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <image/image.h>
#include <image/tga.h>

#include "../../CudaBench.h"

__constant__ __device__ unsigned int full = 0xffffffff;

__global__ void threshold(uint8_t *data, uint8_t threshold, int size, int partSize)
{
	int thread = blockDim.x * blockIdx.x + threadIdx.x;
	uint8_t *ptr = data + thread * partSize;
	uint8_t *end = data + min(size, thread * partSize + partSize);

	for (; ptr < end; ++ptr) {
		if (*ptr < threshold) {
			*ptr = 0;
		} else {
			*ptr = 255;
		}
	}
}

__global__ void thresholdSIMD(unsigned int *data, unsigned int threshold, int size)
{
	unsigned int *ptr = data + blockDim.x * blockIdx.x + threadIdx.x;

	*ptr = __vcmpgeu4(*ptr, threshold);
}

int main(int argc, char *argv[])
{
	if (argc != 4) {
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

	uint8_t thresholdValue = atoi(argv[2]);
	unsigned int thresholdValue32 = thresholdValue | (thresholdValue << 8) | (thresholdValue << 16) | (thresholdValue << 24);

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

	int c, size;
	uint8_t *c_data;
	int partSize = 4;
	int threadsPerBlock = 512;
	int blocks = input_image->width * input_image->height / threadsPerBlock / partSize;

	size = input_image->width * input_image->height * sizeof(uint8_t);

	CudaBench_start(allBench);
	cudaMalloc(&c_data, size);

	for (c = 0; c < input_image->channels; ++c) {
		CudaBench_start(sendBench);
		cudaMemcpy(c_data, input_image->data[c], size, cudaMemcpyHostToDevice);
		CudaBench_end(sendBench);

		CudaBench_start(kernelBench);
		thresholdSIMD<<<blocks, threadsPerBlock>>>((unsigned int *) c_data, thresholdValue32, size);
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

	if ((error = TGA_writeImage(argv[3], output_image)) != 0) {
		printf("Error when writing image: %d\n", error);
	}

	Image_delete(input_image);
	Image_delete(output_image);

	cudaDeviceReset();

	return 0;
}