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
		return;
	}

	if (input_image1->height != input_image2->height ||
		input_image1->width != input_image2->width ) {
		printf("Error : Images should be the same size.");
		return;
	}

	if ((error = Image_copy(input_image1, &output_image)) != 0) {
		printf("Error when copying image: %d\n", error);
		Image_delete(input_image);
		return;
	}

	int i, size;
	uint8_t *data, dataImg1, dataImg2;
	int currentData;

	data = output_image->data;
	dataImg1 = input_image1->data;
	dataImg2 = input_image2->data;
	size = input_image->width * input_image->height * input_image->channels;

	for (i = 0; i < size; ++i) {
		currentData = *dataImg1 + *dataImg2;
		*data = (currentData > 0xff) ? 0xff : currentData;
		++data;
		++dataImg1;
		++dataImg2;
	}

	if ((error = TGA_writeImage(argv[2], output_image)) != 0) {
		printf("Error when writing image: %d\n", error);
	}

	Image_delete(input_image);
	Image_delete(output_image);
}