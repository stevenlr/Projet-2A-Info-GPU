/**
 * @file image.h
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 */

#ifndef _IMAGE_H
#define _IMAGE_H

#include <stdint.h>

/**
 * Image data structure.
 */
struct Image {
	int width;	/**< Number of columns. */
	int height;	/**< Number of rows. */
	int bpp;	/**< Number of bits per pixel. (8 for grayscale, 24 for RGB, 32 for RGBA) */
	uint8_t *data;	/**< Image data. */
};

typedef struct Image Image;

/**
 * Creates an image and initializes it to 0.
 * @param width Number of columns.
 * @param height Number of rows.
 * @param bpp Number of bits per pixel. (8, 24 or 32)
 * @param imageptr Pointer to the image to create.
 * @return 	0 if image was created successfully,
 *		1 if there was an error when allocating memory,
 *		2 if arguments were invalid.
 */
int Image_new(int width, int height, int bpp, Image **imageptr);

/**
 * Deletes an image.
 * @param image Image to be deleted.
 * @return	0 if the image was deleted successfully,
 *		1 otherwise.
 */
int Image_delete(Image *image);

#endif