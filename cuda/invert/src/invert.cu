#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <image/image.h>
#include <image/tga.h>

__global__ void invert(uint8_t *data, int width, int height)
{
	int y = blockDim.x * blockIdx.x + threadIdx.x;
	int start = y * width;
	int end = start + width;
	int i;

	if (y < height) {
		for (i = start; i < end; ++i) {
			data[i] = 255 - data[i];
		}
	}
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

	int c, size;
	uint8_t *c_data;
	int threadsPerBlock = 512;
	int blocks = input_image->height / 512;

	size = input_image->width * input_image->height * sizeof(uint8_t);
	cudaMalloc(&c_data, size);

	for (c = 0; c < input_image->channels; ++c) {
		cudaMemcpy(c_data, input_image->data[c], size, cudaMemcpyHostToDevice);
		invert<<<blocks, threadsPerBlock>>>(c_data, input_image->width, input_image->height);
		cudaMemcpy(output_image->data[c], c_data, size, cudaMemcpyDeviceToHost);
	}

	cudaFree(c_data);

	if ((error = TGA_writeImage(argv[2], output_image)) != 0) {
		printf("Error when writing image: %d\n", error);
	}

	Image_delete(input_image);
	Image_delete(output_image);

	return 0;
}