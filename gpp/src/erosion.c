/**
 * @file erosion.c
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <image/image.h>
#include <image/tga.h>

#include "erosion.h"
#include "benchmark.h"

#define min(x, y) ((x < y) ? x : y)
#define max(x, y) ((x > y) ? x : y)

void erosion(int argc, char *argv[])
{
	if (argc != 3) {
		printf("Invalid number of arguments.\n");
		return;
	}

	Image *input_image;
	Image *output_image;
	int error;
	int radius;

	radius = (int) atoi(argv[1]);

	if (radius < 1) {
		printf("Invalid erosion radius value.\n");
		return;
	}

	if ((error = TGA_readImage(argv[0], &input_image)) != 0) {
		printf("Error when opening image: %d\n", error);
		return;
	}

	if ((error = Image_copy(input_image, &output_image)) != 0) {
		printf("Error when copying image: %d\n", error);
		Image_delete(input_image);
		return;
	}

	int x, y, c, i, size;
	int line_offset, row_offset;
	uint8_t *in_data, *out_data;
	uint8_t current_min;

	in_data = input_image->data;
	out_data = output_image->data;

	line_offset = input_image->width * input_image->channels;
	row_offset = input_image->channels;
	size = input_image->width * input_image->height * input_image->channels;

	Benchmark bench;
	start_benchmark(&bench);

	for (i = 0; i < size; ++i) {
		c = i % input_image->channels;
		x = ((i - c) / input_image->channels) % input_image->width;
		y = ((i - c) / input_image->channels) / input_image->height;
		current_min = *out_data;

		int x1 = max(0, x - radius);
		int x2 = min(input_image->width, x + radius);
		int y1 = max(0, y - radius);
		int y2 = min(input_image->height, y + radius);
		int xx, yy;

		uint8_t *region = in_data + Image_getOffset(input_image, x1, y1) + c;

		for (yy = y1; yy <= y2; ++yy) {
			for (xx = x1; xx <= x2; ++xx) {
				current_min = min(current_min, *region);
				region += row_offset;
			}

			region += line_offset - row_offset * (x2 - x1 + 1);
		}

		*out_data++ = current_min;
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
