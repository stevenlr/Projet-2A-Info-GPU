/**
 * @file add.c
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#include <cstdlib>
#include <iostream>
#include <cstdint>
#include <image/image.h>
#include <image/tga.h>
#include <tbb/tbb.h>

#include "add.h"
#include "benchmark.h"

#include <emmintrin.h>

using namespace std;
using namespace tbb;

class AddParallel 
{
public:
	AddParallel(Image *i1, Image *i2, Image *o, int channel) :
		i1(i1), i2(i2), o(o), c(channel)
 	{ }

	void operator()(int i) const
	{
		uint8_t *data1, *data2, *datao;
		__m128i v1, v2, res;

		data1 = i1->data[c] + i * i1->width;
		data2 = i2->data[c] + i * i1->width;
		datao = o->data[c] + i * i1->width;

		for (int j = 0; j < i1->width; j += 16) {
			v1 = _mm_load_si128((__m128i *) data1);
			v2 = _mm_load_si128((__m128i *) data2);
			res = _mm_adds_epu8(v1, v2);
			_mm_store_si128((__m128i *) datao, res);

			datao += 16;
			data1 += 16;
			data2 += 16;
		}
	}

private:
	Image *i1;
	Image *i2;
	Image *o;
	int c;
};

void add(int argc, char *argv[])
{
	if (argc != 3) {
		cout << "Invalid number of arguments." << endl;
		return;
	}

	Image *input_image1;
	Image *input_image2;
	Image *output_image;
	int error;

	if ((error = TGA_readImage(argv[0], &input_image1)) != 0) {
		cout << "Error when opening image: " << error << endl;
		return;
	}

	if ((error = TGA_readImage(argv[1], &input_image2)) != 0) {
		cout << "Error when opening image: " << error << endl;
		Image_delete(input_image2);
		return;
	}

	if (input_image1->height != input_image2->height ||
		input_image1->width != input_image2->width ||
		input_image1->channels != input_image2->channels) {
		cout << "Error : Input images should be the same size." << endl;
		Image_delete(input_image1);
		Image_delete(input_image2);
		return;
	}

	if ((error = Image_new(input_image1->width,
			       input_image1->height,
			       input_image1->channels,
			       &output_image)) != 0) {
		cout << "Error when creating output image : " << error << endl;
		Image_delete(input_image1);
		Image_delete(input_image2);
		return;
	}

	Benchmark bench;
	start_benchmark(&bench);

	for (int i = 0; i < input_image1->channels; ++i) {
		AddParallel addParallel(input_image1, input_image2, output_image, i);
		parallel_for(0, input_image1->height, addParallel);
	}

	end_benchmark(&bench);
	cout << bench.elapsed_ticks << " ";
	cout << bench.elapsed_time << endl;

	if ((error = TGA_writeImage(argv[2], output_image)) != 0) {
		cout << "Error when writing image: " << error;
	}

	Image_delete(input_image1);
	Image_delete(input_image2);
	Image_delete(output_image);
}