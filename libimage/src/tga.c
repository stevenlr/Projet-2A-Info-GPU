/**
 * @file tga.c
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "image/image.h"
#include "image/tga.h"

#pragma pack(push)
#pragma pack(1)

/**
 * Should be unused.
 */
struct TGAColorMapSpecification {
	uint16_t first_entry_index;
	uint16_t color_map_length;
	uint8_t color_map_entry_size;
};

struct TGAImageSpecification {
	uint16_t x_origin;
	uint16_t y_origin;

	uint16_t width;
	uint16_t height;

	/**
	 * Should be 8 if image_type is 3
	 * and 24 if image_type is 2.
	 */
	uint8_t pixel_depth;
	uint8_t image_descriptor;
};

struct TGAHeader {

	uint8_t id_length;

	/**
	 * Should always be 0. (No color map)
	 */
	uint8_t color_map_type;

	/**
	 * 2 : uncompressed true-color image.
	 * 3 : uncompressed grayscale image.
	 * other : not supported.
	 */
	uint8_t image_type;

	struct TGAColorMapSpecification color_map_specification;
	struct TGAImageSpecification image_specification;
};

#pragma pack(pop)

int TGA_readImage(const char *filename, Image **imageptr)
{
	if (imageptr == NULL)
		return 2;

	FILE* fp = fopen(filename, "rb");

	if (fp == NULL)
		return 2;

	struct TGAHeader header;
	int nb_channels;

	fread(&header, sizeof(struct TGAHeader), 1, fp);

	if (header.color_map_type != 0
	    || (header.image_type != 2 && header.image_type != 3)) {
	    	fclose(fp);
		return 3;
	}

	nb_channels = (header.image_type == 2) ? 3 : 1;

	if (header.image_specification.pixel_depth != nb_channels * 8) {
		fclose(fp);
		return 3;
	}

	// Skips image ID.
	fseek(fp, header.id_length, SEEK_CUR);

	int error = Image_new(header.image_specification.width,
			      header.image_specification.height,
			      nb_channels, imageptr);

	if (error) {
		fclose(fp);
		return error;
	}

	Image *image = *imageptr;
	size_t size = image->width * image->height * image->channels;
	fread(image->data, sizeof(uint8_t), size, fp);

	// Converts BGR to RGB.
	if (header.image_type == 2) {
		int nb_pixels = size / 3;
		int i;
		uint8_t *dataptr = image->data;
		uint8_t tmp;

		for (i = 0; i < nb_pixels; ++i, dataptr += 3) {
			tmp = dataptr[0];
			dataptr[0] = dataptr[2];
			dataptr[0] = tmp;
		}
	}

	fclose(fp);

	return 0;
}

int TGA_writeImage(const char *filename, Image **imageptr)
{

	return 0;
}