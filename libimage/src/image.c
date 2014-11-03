/**
 * @file image.c
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "image.h"

int Image_new(int width, int height, int bpp, Image **imageptr)
{
	if (width < 0 || height < 0 || imageptr == NULL ||
	    (bpp != 8 && bpp != 24 && bpp != 32)) {
		return 2;
	}

	size_t size = width * height * (bpp >> 3);
	Image *image = (Image *) malloc(sizeof(Image));

	if (image == NULL) {
		return 1;
	}

	image->width = width;
	image->height = height;
	image->bpp = bpp;

	image->data = (uint8_t *) calloc(size, sizeof(uint8_t));

	if (image->data == NULL) {
		free(image);
		return 1;
	}

	*imageptr = image;

	return 0;
}

int Image_delete(Image *image)
{
	if (image == NULL || image->data == NULL) {
		return 1;
	}

	free(image->data);
	free(image);

	return 0;
}