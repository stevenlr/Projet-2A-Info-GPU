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
#include "memory.h"

#include <emmintrin.h>

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

	int c, i, size, x, y;
	int line_offset;
	uint8_t *in_data, *out_data;
	uint8_t *data_store;
	uint8_t current_min;
	__m128i src, tmpmin;

	data_store = (uint8_t *) aligned_malloc(sizeof(uint8_t) * 16, 16);

	if (data_store == NULL) {
		printf("Error when allocating store memory.\n");
		Image_delete(input_image);
		Image_delete(output_image);
		return;
	}

	line_offset = input_image->width;
	size = input_image->width * input_image->height;

	Benchmark bench;
	start_benchmark(&bench);

	for (c = 0; c < input_image->channels; ++c) {
		in_data = input_image->data[c];
		out_data = output_image->data[c];

		for (i = 0; i < size; ++i) {
			x = i % input_image->width;
			y = i / input_image->width;
			current_min = 0xff;

			int x1 = max(0, x - radius);
			int x2 = min(input_image->width - 1, x + radius);
			int y1 = max(0, y - radius);
			int y2 = min(input_image->height - 1, y + radius);
			int xx, yy;
			
			for (xx = x1; xx <= x2; xx += 16) {
				uint8_t *region = in_data + Image_getOffset(input_image, x1, y1);

				tmpmin = _mm_setr_epi32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);

				for (yy = y1; yy <= y2; ++yy) {
					src = _mm_loadu_si128((__m128i *) region);
					tmpmin = _mm_min_epu8(src, tmpmin);
					region += line_offset;
				}

				_mm_store_si128((__m128i *) data_store, tmpmin);
				uint8_t *data_storeptr;
				uint8_t *data_storeptr_end;

				data_storeptr = data_store;
				data_storeptr_end = data_storeptr +  min(xx + 16, x2 + 1) - xx;

				for (; data_storeptr < data_storeptr_end; ++data_storeptr)
					current_min = min(current_min, *data_storeptr);
			}

			*out_data++ = current_min;
		}
	}

	end_benchmark(&bench);
	printf("%lu ", bench.elapsed_ticks);
	printf("%lf\n", bench.elapsed_time);

	if ((error = TGA_writeImage(argv[2], output_image)) != 0) {
		printf("Error when writing image: %d\n", error);
	}

	aligned_free(data_store);
	Image_delete(input_image);
	Image_delete(output_image);
}
