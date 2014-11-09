/**
 * @file tga.h
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 */

#ifndef _TGA_H
#define _TGA_H

#include <stdint.h>
#include "image/image.h"

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
	 * 2 : uncompressed color-mapped image.
	 * 3 : uncompressed grayscale image.
	 * other : not supported.
	 */
	uint8_t image_type;

	struct TGAColorMapSpecification;
	struct TGAImageSpecification;
};

#pragma pack(pop)

#endif