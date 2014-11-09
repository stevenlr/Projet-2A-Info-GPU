/**
 * @file image.c
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "image/image.h"

int Image_new(int width, int height, int channels, Image **imageptr)
{
	if (width < 0 || height < 0 || imageptr == NULL ||
	   (channels != 1 && channels != 3)) {
		return 2;
	}

	size_t size = width * height * channels;
	Image *image = (Image *) malloc(sizeof(Image));

	if (image == NULL) {
		return 1;
	}

	image->width = width;
	image->height = height;
	image->channels = channels;

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

int Image_getOffset(Image *image, int x, int y)
{
	if (image == NULL
	    || x < 0 || x >= image->width
	    || y < 0 || y >= image->height) {
		return -1;
	}

	return (y * image->width + x) * image->channels;
}

uint8_t Image_getPixel(Image *image, int x, int y, int c)
{
	if (c < 0 || c >= image->channels)
		return 0;

	int offset = Image_getOffset(image, x, y);

	if (offset == -1)
		return 0;

	return image->data[offset + c];
}

void Image_setPixel(Image *image, int x, int y, int c, uint8_t value)
{
	if (c < 0 || c >= image->channels)
		return;

	int offset = Image_getOffset(image, x, y);

	if (offset == -1)
		return;

	image->data[offset + c] = value;
}