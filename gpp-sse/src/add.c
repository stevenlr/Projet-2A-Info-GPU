/**
 * @file add.c
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <image/image.h>
#include <image/tga.h>

#include "add.h"
#include "benchmark.h"

#include <emmintrin.h>

void add(int argc, char *argv[])
{
	if (argc != 3) {
		printf("Invalid number of arguments.\n");
		return;
	}

	Image *input_image1;
	Image *input_image2;
	Image *output_image;
	int error;

	if ((error = TGA_readImage(argv[0], &input_image1)) != 0) {
		printf("Error when opening image: %d\n", error);
		return;
	}

	if ((error = TGA_readImage(argv[1], &input_image2)) != 0) {
		printf("Error when opening image: %d\n", error);
		Image_delete(input_image2);
		return;
	}

	if (input_image1->height != input_image2->height ||
		input_image1->width != input_image2->width ||
		input_image1->channels != input_image2->channels) {
		printf("Error : Input images should be the same size.\n");
		Image_delete(input_image1);
		Image_delete(input_image2);
		return;
	}

	if ((error = Image_new(input_image1->width,
			       input_image1->height,
			       input_image1->channels,
			       &output_image)) != 0) {
		printf("Error when creating output image : %d\n", error);
		Image_delete(input_image1);
		Image_delete(input_image2);
		return;
	}

	int i, size, c;
	uint8_t *datao, *data1, *data2;
	__m128i v1, v2, res;

	size = input_image1->width * input_image1->height;

	Benchmark bench;
	start_benchmark(&bench);

	for (c = 0; c < input_image1->channels; ++c) {
		datao = output_image->data[c];
		data1 = input_image1->data[c];
		data2 = input_image2->data[c];

		for (i = 0; i < size; i += 16) {
			v1 = _mm_load_si128((__m128i *) data1);
			v2 = _mm_load_si128((__m128i *) data2);
			res = _mm_adds_epu8(v1, v2);
			_mm_store_si128((__m128i *) datao, res);

			datao += 16;
			data1 += 16;
			data2 += 16;
		}
	}

	end_benchmark(&bench);
	printf("%lu ", bench.elapsed_ticks);
	printf("%lf\n", bench.elapsed_time);

	if ((error = TGA_writeImage(argv[2], output_image)) != 0) {
		printf("Error when writing image: %d\n", error);
	}

	Image_delete(input_image1);
	Image_delete(input_image2);
	Image_delete(output_image);
}
