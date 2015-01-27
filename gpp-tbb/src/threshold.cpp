/**
 * @file threshold.c
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#include <cstdlib>
#include <iostream>
#include <cstdint>
#include <image/image.h>
#include <image/tga.h>
#include <tbb/tbb.h>

#include "threshold.h"
#include "benchmark.h"

#include <emmintrin.h>

using namespace std;
using namespace tbb;

class ThresholdParallel 
{
public:
	ThresholdParallel(Image *output_image, uint8_t value, int channel) :
		output_image(output_image), value(value), c(channel)
 	{ 
 		mask = _mm_setr_epi32(0x80808080, 0x80808080, 0x80808080, 0x80808080);
		threshold = _mm_setr_epi8(
			value, value, value, value,
			value, value, value, value,
			value, value, value, value,
			value, value, value, value
		);
 	}

	void operator()(int i) const
	{
		uint8_t *data, *data_end;
		__m128i src, dst;

		data = output_image->data[c] + i * output_image->width;
		data_end = data + output_image->width;

		while (data < data_end) {
			src = _mm_load_si128((__m128i *) data);
			src = _mm_xor_si128(src, mask);
			dst = _mm_cmpgt_epi8(src, threshold);
			_mm_store_si128((__m128i *) data, dst);
			data += 16;
		}
	}

private:
	Image *output_image;
	uint8_t value;
	__m128i mask, threshold;
	int c;
};

void threshold(int argc, char *argv[])
{
	if (argc != 3) {
		cout << "Invalid number of arguments." << endl;
		return;
	}

	Image *input_image;
	Image *output_image;
	int error;
	uint8_t value;

	value = (uint8_t) atoi(argv[1]);

	if ((error = TGA_readImage(argv[0], &input_image)) != 0) {
		cout << "Error when opening image: " << error << endl;
		return;
	}

	if ((error = Image_copy(input_image, &output_image)) != 0) {
		cout << "Error when copying image: " << error << endl;
		Image_delete(input_image);
		return;
	}

	Benchmark bench;
	start_benchmark(&bench);

	if (value == 0) {
		int size = input_image->width * input_image->height;
		
		for (int c = 0; c < input_image->channels; ++c) {
			memset(output_image->data[c], 0xff, size * sizeof(uint8_t));
		}
	} else {
		for (int i = 0; i < output_image->channels; ++i) {
			ThresholdParallel thresholdParallel(output_image, value, i);
			parallel_for(0, output_image->width, thresholdParallel);
		}
	}

	end_benchmark(&bench);
	cout << bench.elapsed_ticks << " ";
	cout << bench.elapsed_time << endl;

	if ((error = TGA_writeImage(argv[2], output_image)) != 0) {
		cout << "Error when writing image: " << error << endl;
	}

	Image_delete(input_image);
	Image_delete(output_image);
}