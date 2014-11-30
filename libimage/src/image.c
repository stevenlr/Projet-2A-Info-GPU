/**
 * @file image.c
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "image/image.h"

// Modified from http://stackoverflow.com/a/1920516
static void *aligned_malloc(size_t size, size_t align)
{
	void *mem = malloc(size + align * 2 + sizeof(void*));
	void **ptr = (void **)((uintptr_t)(mem + align + sizeof(void*)) & ~(align - 1));
	ptr[-1] = mem;
	return ptr;
}

static void aligned_free(void *ptr)
{
	free(((void**) ptr)[-1]);
}

int Image_new(int width, int height, int channels, Image **imageptr)
{
	if (width <= 0 || height <= 0 || imageptr == NULL ||
	   (channels != 1 && channels != 3)) {
		return 2;
	}

	size_t size = width * height;
	Image *image = (Image *) malloc(sizeof(Image));

	if (image == NULL) {
		return 1;
	}

	image->width = width;
	image->height = height;
	image->channels = channels;

	image->data = (uint8_t **) malloc(sizeof(uint8_t *) * channels);

	if (image->data == NULL) {
		free(image);
		return 1;
	}

	int i;

	for (i = 0; i < channels; ++i) {
		image->data[i] = (uint8_t *) aligned_malloc(size * sizeof(uint8_t), 16);

		if (image->data[i] == NULL) {
			int j;

			for (j = 0; j < i; ++j) {
				aligned_free(image->data[j]);
			}

			free(image->data);
			free(image);
			return 1;
		}

		memset(image->data[i], 0, size * sizeof(uint8_t));
	}

	*imageptr = image;

	return 0;
}

int Image_delete(Image *image)
{
	int i;

	if (image == NULL || image->data == NULL) {
		return 1;
	}

	for (i = 0; i < image->channels; ++i) {
		aligned_free(image->data[i]);
	}

	free(image->data);
	free(image);

	return 0;
}

int Image_copy(Image *src, Image **dst)
{
	if (dst == NULL || src == NULL || src->data == NULL) {
		return 2;
	}

	size_t size = src->width * src->height;
	Image *image = (Image *) malloc(sizeof(Image));

	if (image == NULL) {
		return 1;
	}

	image->width = src->width;
	image->height = src->height;
	image->channels = src->channels;

	image->data = (uint8_t **) malloc(sizeof(uint8_t *) * image->channels);

	if (image->data == NULL) {
		free(image);
		return 1;
	}

	int i = 0;

	for (i = 0; i < image->channels; ++i) {
		image->data[i] = (uint8_t *) aligned_malloc(size * sizeof(uint8_t), 16);

		if (image->data[i] == NULL) {
			int j;

			for (j = 0; j < i; ++j) {
				aligned_free(image->data[j]);
			}

			free(image->data);
			free(image);
			return 1;
		}

		memcpy(image->data[i], src->data[i], size * sizeof(uint8_t));
	}

	*dst = image;

	return 0;
}

int Image_getOffset(Image *image, int x, int y)
{
	if (image == NULL
	    || x < 0 || x >= image->width
	    || y < 0 || y >= image->height) {
		return -1;
	}

	return y * image->width + x;
}

uint8_t Image_getPixel(Image *image, int x, int y, int c)
{
	if (c < 0 || c >= image->channels)
		return 0;

	int offset = Image_getOffset(image, x, y);

	if (offset == -1)
		return 0;

	return image->data[c][offset];
}

void Image_setPixel(Image *image, int x, int y, int c, uint8_t value)
{
	if (c < 0 || c >= image->channels)
		return;

	int offset = Image_getOffset(image, x, y);

	if (offset == -1)
		return;

	image->data[c][offset] = value;
}