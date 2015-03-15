#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <image/image.h>
#include <image/tga.h>

#include "../../CudaBench.h"

#define TILE_SIZE 128
#define PROCESSED_SIZE 32
#define MAX_RADIUS ((TILE_SIZE - PROCESSED_SIZE - 1) / 2)

__global__ void erosion2(uint8_t *inData, uint8_t *outData, int radiusX, int radiusY, int width, int height)
{
	__shared__ uint8_t localData[TILE_SIZE * TILE_SIZE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int gx = blockIdx.x * blockDim.x;
	int gy = blockIdx.y * blockDim.y;

	localData[TILE_SIZE * (radiusY + ty) + radiusX + tx] = inData[width * (gy + ty) + gx + tx];

	int x1 = tx, y1 = ty, x2 = tx, y2 = ty;

	if (tx == 0) {
		x1 = max(0, gx - radiusX) - gx;
	} else if (tx == blockDim.x - 1) {
		x2 = min(width - 1 - gx, blockDim.x + radiusX - 1);
	}

	if (ty == 0) {
		y1 = max(0, gy - radiusY) - gy;
	} else if (ty == blockDim.y - 1) {
		y2 = min(height - 1 - gy, blockDim.y + radiusY - 1);
	}

	__syncthreads();

	for (int y = y1; y <= y2; ++y) {
		for (int x = x1; x <= x2; ++x) {
			localData[TILE_SIZE * (radiusY + y) + radiusX + x] = inData[width * (gy + y) + gx + x];
		}
	}

	__syncthreads();

	x1 = tx - radiusX;
	x2 = tx + radiusX;
	y1 = ty - radiusY;
	y2 = ty + radiusY;

	if (gx + x1 < 0) {
		x1 = 0;
	} else if (gx + x2 >= width) {
		x2 = width - gx - 1;
	}

	if (gy + y1 < 0) {
		y1 = 0;
	} else if (gy + y2 >= height) {
		y2 = height - gy - 1;
	}

	uint8_t minimum = 255;

	for (int y = y1; y <= y2; ++y) {
		for (int x = x1; x <= x2; ++x) {
			minimum = min(minimum, localData[TILE_SIZE * (radiusY + y) + radiusX + x]);
		}
	}

	outData[width * (gy + ty) + gx + tx] = minimum;
}

__global__ void erosion(uint8_t *inData, uint8_t *outData, int radiusX, int radiusY, int width, int height)
{
	int gx = blockIdx.x * blockDim.x + threadIdx.x;
	int gy = blockIdx.y * blockDim.y + threadIdx.y;

	int x1 = gx - radiusX;
	int x2 = gx + radiusX;
	int y1 = gy - radiusY;
	int y2 = gy + radiusY;

	if (x1 < 0) {
		x1 = 0;
	} else if (x2 >= width) {
		x2 = width - 1;
	}

	if (y1 < 0) {
		y1 = 0;
	} else if (y2 >= height) {
		y2 = height - 1;
	}

	uint8_t minimum = 255;

	for (int y = y1; y <= y2; ++y) {
		for (int x = x1; x <= x2; ++x) {
			minimum = min(minimum, inData[width * y + x]);
		}
	}

	outData[width * gy + gx] = minimum;
}

int main(int argc, char *argv[])
{
	if (argc != 5) {
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

	int rx = atoi(argv[2]);
	int ry = atoi(argv[3]);

	if (rx < 0 || rx >= MAX_RADIUS || ry < 0 || ry >= MAX_RADIUS) {
		printf("Invalid radius value.\n");
		Image_delete(input_image);
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

	int c, size;
	uint8_t *c_inData, *c_outData;
	
	size = input_image->width * input_image->height * sizeof(uint8_t);

	dim3 threadsPerBlock(PROCESSED_SIZE, PROCESSED_SIZE, 1);
	dim3 blocks(ceil(input_image->width / PROCESSED_SIZE), ceil(input_image->height / PROCESSED_SIZE), 1);

	CudaBench_start(allBench);
	cudaMalloc(&c_inData, size);
	cudaMalloc(&c_outData, size);

	for (c = 0; c < input_image->channels; ++c) {
		CudaBench_start(sendBench);
		cudaMemcpy(c_inData, input_image->data[c], size, cudaMemcpyHostToDevice);
		CudaBench_end(sendBench);

		CudaBench_start(kernelBench);
		erosion<<<blocks, threadsPerBlock>>>(c_inData, c_outData, rx, 0, input_image->width, input_image->height);
		erosion<<<blocks, threadsPerBlock>>>(c_outData, c_inData, 0, ry, input_image->width, input_image->height);
		CudaBench_end(kernelBench);

		CudaBench_start(retrieveBench);
		cudaMemcpy(output_image->data[c], c_inData, size, cudaMemcpyDeviceToHost);
		CudaBench_end(retrieveBench);
	}

	cudaFree(c_inData);
	cudaFree(c_outData);
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

	if ((error = TGA_writeImage(argv[4], output_image)) != 0) {
		printf("Error when writing image: %d\n", error);
	}

	Image_delete(input_image);
	Image_delete(output_image);

	cudaDeviceReset();

	return 0;
}