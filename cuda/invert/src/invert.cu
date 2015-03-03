#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <image/image.h>
#include <image/tga.h>

__global__ void invert(uint8_t *data, int size, int partSize)
{
	int thread = blockDim.x * blockIdx.x + threadIdx.x;
	int start = thread * partSize;
	int end = min(start + partSize, size);
	int i;

	for (i = start; i < end; ++i) {
		data[i] = 255 - data[i];
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

	int partSizes[] = {4, 16, 64, 256, 1024};
	int threadsPerBlocks[] = {32, 96, 192, 384};

	int partSizeIndex = 0;
	int threadsPerBlockIndex = 0;

	for (partSizeIndex = 0; partSizeIndex < 5; ++partSizeIndex) {
		for (threadsPerBlockIndex = 0; threadsPerBlockIndex < 4; ++threadsPerBlockIndex) {
			int c, size;
			uint8_t *c_data;
			int partSize = partSizes[partSizeIndex];
			int threadsPerBlock = threadsPerBlocks[threadsPerBlockIndex];
			int blocks = input_image->width * input_image->height / threadsPerBlock / partSize;

			printf("%d %d\n", partSize, threadsPerBlock);

			size = input_image->width * input_image->height * sizeof(uint8_t);

			cudaEvent_t startAll, startChannel, startKernel;
			cudaEvent_t stopAll, stopChannel, stopKernel;

			cudaEventCreate(&startAll);
			cudaEventCreate(&startChannel);
			cudaEventCreate(&startKernel);
			cudaEventCreate(&stopAll);
			cudaEventCreate(&stopChannel);
			cudaEventCreate(&stopKernel);

			cudaEventRecord(startAll);
			cudaMalloc(&c_data, size);

			for (c = 0; c < input_image->channels; ++c) {
				cudaEventRecord(startChannel);
				cudaMemcpy(c_data, input_image->data[c], size, cudaMemcpyHostToDevice);
				cudaEventRecord(startKernel);
				invert<<<blocks, threadsPerBlock>>>(c_data, input_image->width * input_image->height, partSize);
				cudaEventRecord(stopKernel);
				cudaMemcpy(output_image->data[c], c_data, size, cudaMemcpyDeviceToHost);
				cudaEventRecord(stopChannel);
			}

			cudaFree(c_data);
			cudaEventRecord(stopAll);

			cudaEventSynchronize(stopAll);

			float timeAll, timeChannel, timeKernel;

			cudaEventElapsedTime(&timeAll, startAll, stopAll);
			cudaEventElapsedTime(&timeChannel, startChannel, stopChannel);
			cudaEventElapsedTime(&timeKernel, startKernel, stopKernel);

			cudaEventDestroy(startAll);
			cudaEventDestroy(startChannel);
			cudaEventDestroy(startKernel);
			cudaEventDestroy(stopAll);
			cudaEventDestroy(stopChannel);
			cudaEventDestroy(stopKernel);

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

		cudaEvent_t startAll, startChannel, startKernel;
		cudaEvent_t stopAll, stopChannel, stopKernel;

		cudaEventCreate(&startAll);
		cudaEventCreate(&startChannel);
		cudaEventCreate(&startKernel);
		cudaEventCreate(&stopAll);
		cudaEventCreate(&stopChannel);
		cudaEventCreate(&stopKernel);

		cudaEventRecord(startAll);
		cudaMalloc(&c_data, size);

		for (c = 0; c < input_image->channels; ++c) {
			cudaEventRecord(startChannel);
			cudaMemcpy(c_data, input_image->data[c], size, cudaMemcpyHostToDevice);
			cudaEventRecord(startKernel);
			invertSIMD<<<blocks, threadsPerBlock>>>((unsigned int *) c_data, input_image->width * input_image->height);
			cudaEventRecord(stopKernel);
			cudaMemcpy(output_image->data[c], c_data, size, cudaMemcpyDeviceToHost);
			cudaEventRecord(stopChannel);
		}

		cudaFree(c_data);
		cudaEventRecord(stopAll);

		cudaEventSynchronize(stopAll);

		float timeAll, timeChannel, timeKernel;

		cudaEventElapsedTime(&timeAll, startAll, stopAll);
		cudaEventElapsedTime(&timeChannel, startChannel, stopChannel);
		cudaEventElapsedTime(&timeKernel, startKernel, stopKernel);

		cudaEventDestroy(startAll);
		cudaEventDestroy(startChannel);
		cudaEventDestroy(startKernel);
		cudaEventDestroy(stopAll);
		cudaEventDestroy(stopChannel);
		cudaEventDestroy(stopKernel);

		printf("All: %fms\nChannel: %fms\nKernel: %fms\n", timeAll, timeChannel, timeKernel);
		printf("%f\n\n", timeKernel);
	}

	if ((error = TGA_writeImage(argv[2], output_image)) != 0) {
		printf("Error when writing image: %d\n", error);
	}

	Image_delete(input_image);
	Image_delete(output_image);

	return 0;
}