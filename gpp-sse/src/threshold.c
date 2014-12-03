/**
 * @file threshold.c
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <image/image.h>
#include <image/tga.h>

#include "threshold.h"
#include "benchmark.h"

#include <emmintrin.h>

void threshold(int argc, char *argv[])
{
	if (argc != 3) {
		printf("Invalid number of arguments.\n");
		return;
	}

	Image *input_image;
	Image *output_image;
	int error;
	uint8_t value;

	value = (uint8_t) atoi(argv[1]);

	if ((error = TGA_readImage(argv[0], &input_image)) != 0) {
		printf("Error when opening image: %d\n", error);
		return;
	}

	if ((error = Image_copy(input_image, &output_image)) != 0) {
		printf("Error when copying image: %d\n", error);
		Image_delete(input_image);
		return;
	}

	int size, c;
	uint8_t *data, *data_end;
	__m128i threshold, src, dst, mask;

	size = input_image->width * input_image->height;

	Benchmark bench;
	start_benchmark(&bench);

	if (value == 0) {
		for (c = 0; c < input_image->channels; ++c) {
			memset(output_image->data[c], 0xff, size * sizeof(uint8_t));
		}
	} else {
		value -= 1;
		value ^= 0x80;
		
		mask = _mm_setr_epi32(0x80808080, 0x80808080, 0x80808080, 0x80808080);
		threshold = _mm_setr_epi8(
			value, value, value, value,
			value, value, value, value,
			value, value, value, value,
			value, value, value, value
		);

		for (c = 0; c < input_image->channels; ++c) {
			data = output_image->data[c];
			data_end = data + size;

			while (data < data_end) {
				src = _mm_load_si128((__m128i *) data);
				src = _mm_xor_si128(src, mask);
				dst = _mm_cmpgt_epi8(src, threshold);
				_mm_store_si128((__m128i *) data, dst);
				data += 16;
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