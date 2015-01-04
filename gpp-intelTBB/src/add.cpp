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

using namespace std;
using namespace tbb;

/* PARALLEL_FOR ON ROW
class AddParallel 
{
public:
	AddParallel(Image *i1, Image *i2, Image *o) :
		i1(i1), i2(i2), o(o)
 	{ }

	void operator()(int i) const
	{
		uint16_t current_data;
		uint8_t *datai1, *datai2, *datao;

		for (int c = 0; c < i1->channels; ++c) {
			datai1 = i1->data[c] + i * i1->width;
			datai2 = i2->data[c] + i * i1->width;
			datao = o->data[c] + i * i1->width;

			for (int j = 0; j < i1->width; ++j) {
				current_data = (*datai1++) + (*datai2++);
				*datao++ = (current_data > 0xff) ? 0xff : current_data;
			}
		}
	}

private:
	Image *i1;
	Image *i2;
	Image *o;
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

	AddParallel addParallel(input_image1, input_image2, output_image);

	Benchmark bench;
	start_benchmark(&bench);

	parallel_for(0, input_image1->height, addParallel);

	end_benchmark(&bench);
	cout << bench.elapsed_ticks << endl;
	cout << bench.elapsed_time << endl;

	if ((error = TGA_writeImage(argv[2], output_image)) != 0) {
		cout << "Error when writing image: " << error;
	}

	Image_delete(input_image1);
	Image_delete(input_image2);
	Image_delete(output_image);
}*/

/* //PARALLEL_FOR ON CHANNELS
class AddParallel 
{
public:
	AddParallel(Image *input_image1, Image *input_image2, Image *output_image) :
		input_image1(input_image1), input_image2(input_image2), output_image(output_image)
 	{ }

	void operator()(int c) const
	{
		uint16_t current_data;
		uint8_t *data1, *data2, *datao;
		int size, i;

		datao = output_image->data[c];
		data1 = input_image1->data[c];
		data2 = input_image2->data[c];
		size = input_image1->width * input_image1->height;

		for (i = 0; i < size; ++i) {
			current_data = (*data1++) + (*data2++);
			*datao++ = (current_data > 0xff) ? 0xff : current_data;
		}
	}

private:
	Image *input_image1;
	Image *input_image2;
	Image *output_image;
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

	AddParallel addParallel(input_image1, input_image2, output_image);

	Benchmark bench;
	start_benchmark(&bench);

	parallel_for(0, input_image1->channels, addParallel);

	end_benchmark(&bench);
	cout << bench.elapsed_ticks << endl;
	cout << bench.elapsed_time << endl;

	if ((error = TGA_writeImage(argv[2], output_image)) != 0) {
		cout << "Error when writing image: " << error;
	}

	Image_delete(input_image1);
	Image_delete(input_image2);
	Image_delete(output_image);
} */


// Parallel_for ON 8 PARTS
class AddParallel 
{
public:
	AddParallel(Image *input_image1, Image *input_image2, Image *output_image, int nbParts) :
		input_image1(input_image1), input_image2(input_image2), output_image(output_image), nbParts(nbParts)
 	{ }

	void operator()(int p) const
	{
		uint16_t current_data;
		uint8_t *data1, *data2, *datao;
		int c, size, i, partSize, beginPart;
		size = input_image1->width * input_image1->height;
		partSize = size/nbParts;
		beginPart = partSize * p;

		for (c = 0; c < input_image1->channels; ++c) {
			datao = output_image->data[c] + beginPart;
			data1 = input_image1->data[c] + beginPart;
			data2 = input_image2->data[c] + beginPart;

			for (i = 0; i < partSize; ++i) {
				current_data = (*data1++) + (*data2++);
				*datao++ = (current_data > 0xff) ? 0xff : current_data;
			}
		}
	}

private:
	Image *input_image1;
	Image *input_image2;
	Image *output_image;
	int nbParts;
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

	int nbParts = 8; 

	AddParallel addParallel(input_image1, input_image2, output_image, nbParts);

	Benchmark bench;
	start_benchmark(&bench);

	parallel_for(0, nbParts, addParallel);

	end_benchmark(&bench);
	cout << bench.elapsed_ticks << endl;
	cout << bench.elapsed_time << endl;

	if ((error = TGA_writeImage(argv[2], output_image)) != 0) {
		cout << "Error when writing image: " << error;
	}

	Image_delete(input_image1);
	Image_delete(input_image2);
	Image_delete(output_image);
}