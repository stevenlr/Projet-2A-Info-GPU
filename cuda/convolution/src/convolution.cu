#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>

#include <image/image.h>
#include <image/tga.h>

#include "../../CudaBench.h"

#define TILESIZE 32

typedef struct {
	int width;
	int height;
	float sum;
	float *data;
} Kernel;

int get_kernel(const char *file_name, Kernel *kernel)
{
	FILE *fp;

	if (kernel == NULL) {
		return 2;
	}

	kernel->data = NULL;
	kernel->sum = 0;

	if ((fp = fopen(file_name, "r")) == NULL) {
		return 2;
	}

	fscanf(fp, "%d*%d", &(kernel->width), &(kernel->height));

	if (kernel->width % 2 != 1 || kernel->height % 2 != 1
		|| kernel->width < 0 || kernel->height < 0) {
		fclose(fp);
		return 3;
	}

	size_t size = kernel->width * kernel->height;
	unsigned int i;

	kernel->data = (float *) malloc(sizeof(float) * size);

	if (kernel->data == NULL) {
		fclose(fp);
		return 1;
	}

	for (i = 0; i < size; ++i) {
		fscanf(fp, "%f", kernel->data + i);
		kernel->sum += kernel->data[i];
	}

	if (kernel->sum >= -FLT_EPSILON && kernel->sum <= FLT_EPSILON) {
		kernel->sum = 1;
	}

	fclose(fp);

	return 0;
}

void kernel_delete(Kernel kernel)
{
	if (kernel.data != NULL) {
		free(kernel.data);
	}
}

__global__ void convolution(uint8_t *inData, uint8_t *outData, int width, int height, float *kernel, int kwidth, int kheight, float ksum)
{
	int gx = blockIdx.x * blockDim.x + threadIdx.x;
	int gy = blockIdx.y * blockDim.y + threadIdx.y;

	if (gx < width && gy < height) {
		int rx = (kwidth - 1) / 2;
		int ry = (kheight - 1) / 2;

		float sum = 0.0;

		for (int y = 0; y < kheight; ++y) {
			int cy = max(0, min(height - 1, gy + y - ry));

			for (int x = 0; x < kwidth; ++x) {
				int cx = max(0, min(width - 1, gx + x - rx));
				sum = fmaf((float) inData[cx + cy * width], kernel[x + y * kwidth], sum);
			}
		}

		sum = max(0.0, min(255.0, sum));
		outData[gx + gy * width] = (uint8_t) fdividef(sum, ksum);
	}
}

int main(int argc, char *argv[])
{
	if (argc != 4) {
		printf("Invalid number of arguments.\n");
		return 1;
	}

	Image *input_image;
	Image *output_image;
	Kernel kernel;
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

	if ((error = get_kernel(argv[2], &kernel)) != 0) {
		printf("Error when opening kernel : %d\n", error);
		Image_delete(input_image);
		Image_delete(output_image);
		return 1;
	}

	if (kernel.width > 127 || kernel.height > 127) {
		printf("Kernel size too big.\n");
		Image_delete(input_image);
		Image_delete(output_image);
		kernel_delete(kernel);
		return 1;
	}

	CudaBench allBench, sendBench, retrieveBench, kernelBench;
	allBench = CudaBench_new();
	sendBench = CudaBench_new();
	retrieveBench = CudaBench_new();
	kernelBench = CudaBench_new();

	int c, size;
	uint8_t *c_inData, *c_outData;
	float *c_kernel;
	
	size = input_image->width * input_image->height * sizeof(uint8_t);

	dim3 threadsPerBlock(TILESIZE, TILESIZE, 1);
	dim3 blocks(ceil(input_image->width / TILESIZE), ceil(input_image->height / TILESIZE), 1);

	CudaBench_start(allBench);
	cudaMalloc(&c_inData, size);
	cudaMalloc(&c_outData, size);
	cudaMalloc(&c_kernel, sizeof(float) * kernel.width * kernel.height);

	cudaMemcpy(c_kernel, kernel.data, sizeof(float) * kernel.width * kernel.height, cudaMemcpyHostToDevice);

	for (c = 0; c < input_image->channels; ++c) {
		CudaBench_start(sendBench);
		cudaMemcpy(c_inData, input_image->data[c], size, cudaMemcpyHostToDevice);
		CudaBench_end(sendBench);

		CudaBench_start(kernelBench);
		convolution<<<blocks, threadsPerBlock>>>(c_inData, c_outData, input_image->width, input_image->height,
			c_kernel, kernel.width, kernel.height, kernel.sum);
		CudaBench_end(kernelBench);

		CudaBench_start(retrieveBench);
		cudaMemcpy(output_image->data[c], c_outData, size, cudaMemcpyDeviceToHost);
		CudaBench_end(retrieveBench);
	}

	cudaFree(c_inData);
	cudaFree(c_outData);
	cudaFree(c_kernel);
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