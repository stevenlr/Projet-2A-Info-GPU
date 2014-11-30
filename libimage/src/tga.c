/**
 * @file tga.c
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

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
	size_t size = image->width * image->height;
	int c;

	uint8_t *full_data = (uint8_t *) malloc(sizeof(uint8_t) * size * image->channels);

	if (full_data == NULL) {
		return 1;
	}

	fread(full_data, sizeof(uint8_t), size * image->channels, fp);
	fclose(fp);

	if (image->channels == 1) {
		uint8_t *data_ptr = image->data[0];
		uint8_t *scanline_ptr_dst = data_ptr + image->width * (image->height - 1);
		uint8_t *scanline_ptr_src = full_data;

		while (scanline_ptr_dst >= data_ptr) {
			memcpy(scanline_ptr_dst, scanline_ptr_src, image->width);
			scanline_ptr_dst -= image->width;
			scanline_ptr_src += image->width;
		}
	} else {
		for (c = 0; c < image->channels; ++c) {
			uint8_t *data_ptr = image->data[image->channels - c - 1];
			uint8_t *scanline_ptr_dst = data_ptr + image->width * (image->height - 1);
			uint8_t *scanline_ptr_src = full_data;

			while (scanline_ptr_dst >= data_ptr) {
				int x;

				for (x = 0; x < image->width; ++x)
					scanline_ptr_dst[x] = scanline_ptr_src[x * image->channels + c];

				scanline_ptr_dst -= image->width;
				scanline_ptr_src += image->width * image->channels;
			}
		}
	}

	free(full_data);

	return 0;
}

int TGA_writeImage(const char *filename, Image *image)
{
	if (image == NULL)
		return 1;

	FILE* fp = fopen(filename, "wb+");

	if (fp == NULL)
		return 1;

	struct TGAHeader header;

	header.id_length = 0;
	header.color_map_type = 0;
	header.image_type = (image->channels == 1) ? 3 : 2;

	header.color_map_specification.first_entry_index = 0;
	header.color_map_specification.color_map_length = 0;
	header.color_map_specification.color_map_entry_size = 0;

	header.image_specification.x_origin = 0;
	header.image_specification.y_origin = 0;
	header.image_specification.width = image->width;
	header.image_specification.height = image->height;
	header.image_specification.pixel_depth = image->channels * 8;
	header.image_specification.image_descriptor = 0;

	size_t size = image->width * image->height;
	uint8_t *data;

	fwrite(&header, sizeof(struct TGAHeader), 1, fp);

	if (image->channels == 1) {
		data = image->data[0];
	} else {
		data = (uint8_t *) malloc(size * image->channels * sizeof(uint8_t));

		if (data == NULL) {
			fclose(fp);
			return 1;
		}

		int c;
		unsigned int i;
		uint8_t *ptr_dst = data;

		for (i = 0; i < size; ++i) {
			for (c = 0; c < image->channels; ++c)
				*ptr_dst++ = image->data[image->channels - c - 1][i];
		}
	}

	uint8_t *scanline_ptr = data + image->width * (image->height - 1) * image->channels;

	// Inverts y axis
	while (scanline_ptr >= data) {
		fwrite(scanline_ptr, sizeof(uint8_t), image->width * image->channels, fp);
		scanline_ptr -= image->width * image->channels;
	}

	if (image->channels == 3) {
		free(data);
	}

	fclose(fp);

	return 0;
}