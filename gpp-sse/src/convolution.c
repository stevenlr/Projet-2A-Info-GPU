/**
 * @file convolution.c
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <float.h>
#include <image/image.h>
#include <image/tga.h>
#include <tmmintrin.h>

#include "convolution.h"
#include "benchmark.h"

#define min(x, y) ((x < y) ? x : y)
#define max(y, x) ((y < x) ? x : y)
#define clip(x, a, b) (min(b, max(a, x)))

typedef struct {
	int width;
	int height;
	float sum;
	float *data;
} Kernel;

/**
 * Opens a convolution kernel from a file.
 * @param file_name Path to the kernel file.
 * @param kernel Kernel to write data in.
 * @return 0 if the kernel was retrieved successfully,
 *	   1 if there was an error when allocating memory,
 *	   2 if arguments were invalid,
 *	   3 if the matrix dimensions are invalid. (Has to be odd)
 */
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

void convolution(int argc, char *argv[])
{
	if (argc != 3) {
		printf("Invalid number of arguments.\n");
		return;
	}

	Image *input_image;
	Image *output_image;
	int error;

	if ((error = TGA_readImage(argv[0], &input_image)) != 0) {
		printf("Error when opening image: %d\n", error);
		return;
	}

	if ((error = Image_new(input_image->width,
			       input_image->height,
			       input_image->channels,
			       &output_image)) != 0) {
		printf("Error when creating output image : %d\n", error);
		Image_delete(input_image);
		return;
	}

	Kernel kernel;

	if ((error = get_kernel(argv[1], &kernel)) != 0) {
		printf("Error when opening kernel : %d\n", error);
		Image_delete(input_image);
		Image_delete(output_image);
		return;
	}

	int radius_x, radius_y;
	int x, y, c;

	radius_x = (kernel.width - 1) / 2;
	radius_y = (kernel.height - 1) / 2;

	Benchmark bench;
	start_benchmark(&bench);

	for (c = 0; c < input_image->channels; ++c) {
		uint8_t *out_data = output_image->data[c];
		
		for (y = 0; y < input_image->height; ++y) {
			for (x = 0; x < input_image->width; ++x) {
				
				int kx, ky;
				float *kernel_data = kernel.data;
				float sum = 0;

				for (ky = -radius_y; ky <= radius_y; ++ky) {
					for (kx = -radius_x; kx <= radius_x; ++kx) {
						int xx = clip(x + kx, 0, input_image->width - 1);
						int yy = clip(y + ky, 0, input_image->height - 1);

						sum += Image_getPixel(input_image, xx, yy, c) * *kernel_data++;
					}
				}

				sum /= kernel.sum;
				*out_data++ = clip(sum, 0, 255);
			}
		}
	}

	end_benchmark(&bench);
	printf("%lu ", bench.elapsed_ticks);
	printf("%lf\n", bench.elapsed_time);

	if ((error = TGA_writeImage(argv[2], output_image)) != 0) {
		printf("Error when writing image: %d\n", error);
	}

	Image_delete(input_image);
	Image_delete(output_image);
}
