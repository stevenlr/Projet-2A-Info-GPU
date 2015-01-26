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

#define min(x, y) ((x < y) ? x : y)
#define max(x, y) ((x > y) ? x : y)

using namespace std;
using namespace tbb;

class ErosionParallel 
{
public:
	ErosionParallel(Image *input_image, Image *output_image, int radius) :
		input_image(input_image), output_image(output_image), radius(radius)
 	{ }

	void operator()(int i) const
	{
		int c, j, x, y;
		uint8_t *in_data, *out_data;
		uint8_t current_min;
		int line_offset = input_image->width;

		for (c = 0; c < input_image->channels; ++c) {
			in_data = input_image->data[c];
			out_data = output_image->data[c] + i * line_offset;
			y = i;

			for (j = 0; j < line_offset; ++j) {
				x = j;
				current_min = 0xff;

				int x1 = max(0, x - radius);
				int x2 = min(line_offset - 1, x + radius);
				int y1 = max(0, y - radius);
				int y2 = min(input_image->height - 1, y + radius);
				int xx, yy;

				uint8_t *region = in_data + Image_getOffset(input_image, x1, y1);

				for (yy = y1; yy <= y2; ++yy) {
					for (xx = x1; xx <= x2; ++xx) {
						current_min = min(current_min, *region);
						++region;
					}

					region += line_offset - (x2 - x1 + 1);
				}

				*out_data++ = current_min;
			}
		}
	}

private:
	Image *input_image;
	Image *output_image;
	int radius;
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

	ErosionParallel erosionParallel(input_image, output_image, radius);

	Benchmark bench;
	start_benchmark(&bench);

	parallel_for(0, input_image->height, erosionParallel);

	end_benchmark(&bench);
	cout << bench.elapsed_ticks << endl;
	cout << bench.elapsed_time << endl;

	if ((error = TGA_writeImage(argv[2], output_image)) != 0) {
		cout << "Error when writing image " << error << endl;
	}

	Image_delete(input_image);
	Image_delete(output_image);
}