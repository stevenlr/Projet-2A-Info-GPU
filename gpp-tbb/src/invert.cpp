/**
 * @file invert.c
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#include <cstdlib>
#include <iostream>
#include <cstdint>
#include <image/image.h>
#include <image/tga.h>
#include <tbb/tbb.h>

#include "invert.h"
#include "benchmark.h"

#include <emmintrin.h>

using namespace std;
using namespace tbb;

class InvertParallel
{
public:
	InvertParallel(Image *output_image, int channel) :
		output_image(output_image), c(channel)
 	{ 
 		mask = _mm_setr_epi32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);
 	}

	void operator()(int i) const
	{
		uint8_t *datao, *data_end;
		__m128i src, dst;

		datao = output_image->data[c] + i * output_image->width;
		data_end = datao + output_image->width;

		while (datao < data_end) {
			src = _mm_load_si128((__m128i *) datao);
			dst = _mm_xor_si128(src, mask);
			_mm_store_si128((__m128i *) datao, dst);
			datao += 16;
		}
	}

private:
	Image *output_image;
	__m128i mask;
	int c;
};

void invert(int argc, char *argv[])
{
	if (argc != 2) {
		cout << "Invalid number of arguments." << endl;
		return;
	}

	Image *input_image;
	Image *output_image;
	int error;

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

	for (int i = 0; i < input_image->channels; ++i) {
		InvertParallel invertParallel(output_image, i);
		parallel_for(0, input_image->height, invertParallel);
	}

	end_benchmark(&bench);
	cout << bench.elapsed_ticks << " ";
	cout << bench.elapsed_time << endl;

	if ((error = TGA_writeImage(argv[1], output_image)) != 0) {
		cout << "Error when writing image: " << error << endl;
	}

	Image_delete(input_image);
	Image_delete(output_image);
}
