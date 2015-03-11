/**
 * @file erosion.c
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#include <cstdlib>
#include <iostream>
#include <cstdint>
#include <image/image.h>
#include <image/tga.h>
#include <tbb/tbb.h>

#include "erosion.h"
#include "benchmark.h"
#include "memory.h"

#include <emmintrin.h>

#define min(x, y) ((x < y) ? x : y)
#define max(x, y) ((x > y) ? x : y)

using namespace std;
using namespace tbb;

class ErosionParallel 
{
public:
	ErosionParallel(Image *input_image, Image *output_image, int radius, uint8_t *data_store, int channel) :
		input_image(input_image), output_image(output_image), radius(radius), data_store(data_store), c(channel)
 	{ }

	void operator()(int i) const
	{
		int j, x, y;
		uint8_t *in_data, *out_data;
		uint8_t current_min;
		int line_offset = input_image->width;
		__m128i src, tmpmin;

		in_data = input_image->data[c];
		out_data = output_image->data[c] + i * line_offset;
		y = i;

		for (j = 0; j < line_offset; ++j) {
			x = j;
			current_min = 0xff;

			int x1 = max(0, x - radius);
			int x2 = min(input_image->width - 1, x + radius);
			int y1 = max(0, y - radius);
			int y2 = min(input_image->height - 1, y + radius);
			int xx, yy;
			
			for (xx = x1; xx <= x2; xx += 16) {
				uint8_t *region = in_data + Image_getOffset(input_image, xx, y1);

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

private:
	Image *input_image;
	Image *output_image;
	int radius;
	uint8_t *data_store;
	int c;
};

void erosion(int argc, char *argv[])
{
	if (argc != 3) {
		cout << "Invalid number of arguments." << endl;
		return;
	}

	Image *input_image;
	Image *output_image;
	int error;
	int radius;

	radius = (int) atoi(argv[1]);

	if (radius < 1) {
		cout << "Invalid erosion radius value." << endl;
		return;
	}

	if ((error = TGA_readImage(argv[0], &input_image)) != 0) {
		cout << "Error when opening image: " << error << endl;
		return;
	}

	if ((error = Image_copy(input_image, &output_image)) != 0) {
		cout << "Error when copying image: " << error << endl;
		Image_delete(input_image);
		return;
	}

	uint8_t *data_store;

	data_store = (uint8_t *) aligned_malloc(sizeof(uint8_t) * 16, 16);

	if (data_store == NULL) {
		printf("Error when allocating store memory.\n");
		Image_delete(input_image);
		Image_delete(output_image);
		return;
	}

	Benchmark bench;
	start_benchmark(&bench);

	for (int i = 0; i < input_image->channels; ++i) {
		ErosionParallel erosionParallel(input_image, output_image, radius, data_store, i);
		parallel_for(0, input_image->height, erosionParallel);
	}

	end_benchmark(&bench);
	cout << bench.elapsed_ticks << " ";
	cout << bench.elapsed_time << endl;

	if ((error = TGA_writeImage(argv[2], output_image)) != 0) {
		cout << "Error when writing image " << error << endl;
	}

	aligned_free(data_store);
	Image_delete(input_image);
	Image_delete(output_image);
}