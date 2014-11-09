/**
 * @file tga.c
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 */

#include <stdio.h>
#include <stdlib.h>
 #include <stdint.h>

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

	struct TGAColorMapSpecification color_map_specification;
	struct TGAImageSpecification image_specification;
};

#pragma pack(pop)