#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <image/image.h>
#include <image/tga.h>

#include "../../CudaBench.h"

__global__ void invert(uint8_t *data, int size, int partSize)
{
	int thread = blockDim.x * blockIdx.x + threadIdx.x;
	int start = thread * partSize;
	int end = min(start + partSize, size);
	int i;

	for (i = start; i < end; ++i) {
		data[i] = data[i] ^ 0xff;
	}
}

__constant__ __device__ unsigned int full = 0xffffffff;

__global__ void invertSIMD(unsigned int *data, int size)
{
	int thread = blockDim.x * blockIdx.x + threadIdx.x;
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

	int partSizes[] = {1, 4, 16, 64, 256, 1024};
	int threadsPerBlocks[] = {32, 96, 192, 384};

	int partSizeIndex = 0;
	int threadsPerBlockIndex = 0;

	CudaBench allBench, channelBench, kernelBench;
	allBench = CudaBench_new();
	channelBench = CudaBench_new();
	kernelBench = CudaBench_new();

	for (partSizeIndex = 0; partSizeIndex < 6; ++partSizeIndex) {
		for (threadsPerBlockIndex = 0; threadsPerBlockIndex < 4; ++threadsPerBlockIndex) {
			int c, size;
			uint8_t *c_data;
			int partSize = partSizes[partSizeIndex];
			int threadsPerBlock = threadsPerBlocks[threadsPerBlockIndex];
			int blocks = input_image->width * input_image->height / threadsPerBlock / partSize;

			printf("%d %d\n", partSize, threadsPerBlock);

			size = input_image->width * input_image->height * sizeof(uint8_t);

			CudaBench_start(allBench);
			cudaMalloc(&c_data, size);

			for (c = 0; c < input_image->channels; ++c) {
				CudaBench_start(channelBench);
				cudaMemcpy(c_data, input_image->data[c], size, cudaMemcpyHostToDevice);
				CudaBench_start(kernelBench);
				invert<<<blocks, threadsPerBlock>>>(c_data, input_image->width * input_image->height, partSize);
				CudaBench_end(kernelBench);
				cudaMemcpy(output_image->data[c], c_data, size, cudaMemcpyDeviceToHost);
				CudaBench_end(channelBench);
			}

			cudaFree(c_data);
			CudaBench_end(allBench);

			cudaEventSynchronize(allBench.end);

			float timeAll, timeChannel, timeKernel;

			timeAll = CudaBench_elapsedTime(allBench);
			timeChannel = CudaBench_elapsedTime(channelBench);
			timeKernel = CudaBench_elapsedTime(kernelBench);

			//printf("All: %fms\nChannel: %fms\nKernel: %fms\n", timeAll, timeChannel, timeKernel);
			printf("%f\n\n", timeKernel);
		}
	}

	for (threadsPerBlockIndex = 0; threadsPerBlockIndex < 4; ++threadsPerBlockIndex) {
		int c, size;
		uint8_t *c_data;
		int partSize = 4;
		int threadsPerBlock = threadsPerBlocks[threadsPerBlockIndex];
		int blocks = input_image->width * input_image->height / threadsPerBlock / partSize;

		printf("%d %d\n", partSize, threadsPerBlock);

		size = input_image->width * input_image->height * sizeof(uint8_t);

		CudaBench_start(allBench);
		cudaMalloc(&c_data, size);

		for (c = 0; c < input_image->channels; ++c) {
			CudaBench_start(channelBench);
			cudaMemcpy(c_data, input_image->data[c], size, cudaMemcpyHostToDevice);
			CudaBench_start(kernelBench);
			invertSIMD<<<blocks, threadsPerBlock>>>((unsigned int *) c_data, input_image->width * input_image->height);
			CudaBench_end(kernelBench);
			cudaMemcpy(output_image->data[c], c_data, size, cudaMemcpyDeviceToHost);
			CudaBench_end(channelBench);
		}

		cudaFree(c_data);
		CudaBench_end(allBench);

		cudaEventSynchronize(allBench.end);

		float timeAll, timeChannel, timeKernel;

		timeAll = CudaBench_elapsedTime(allBench);
		timeChannel = CudaBench_elapsedTime(channelBench);
		timeKernel = CudaBench_elapsedTime(kernelBench);

		//printf("All: %fms\nChannel: %fms\nKernel: %fms\n", timeAll, timeChannel, timeKernel);
		printf("%f\n\n", timeKernel);
	}

	CubaBench_delete(allBench);
	CubaBench_delete(channelBench);
	CubaBench_delete(kernelBench);

	if ((error = TGA_writeImage(argv[2], output_image)) != 0) {
		printf("Error when writing image: %d\n", error);
	}

	Image_delete(input_image);
	Image_delete(output_image);

	return 0;
}