/**
 * @file image.h
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 */

#ifndef _IMAGE_H
#define _IMAGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>



/**
 * Image data structure.
 */
struct Image {
	int width;	/**< Number of columns. */
	int height;	/**< Number of rows. */
	int channels;	/**< Number of channels. (1 or 3) */
	uint8_t **data;	/**< Image data. */
};

typedef struct Image Image;

/**
 * Creates an image and initializes it to 0.
 * @param width Number of columns.
 * @param height Number of rows.
 * @param channels Number of channels. (1 or 3)
 * @param imageptr Pointer to the image to create.
 * @return 0 if image was created successfully,
 *	   1 if there was an error when allocating memory,
 *	   2 if arguments were invalid.
 */
int Image_new(int width, int height, int channels, Image **imageptr);

/**
 * Deletes an image.
 * @param image Image to be deleted.
 * @return 0 if the image was deleted successfully,
 *	   1 otherwise.
 */
int Image_delete(Image *image);

/**
 * Copies an image.
 * @param src Source image.
 * @param dst Pointer to the destination image.
 * @return 0 if the image was copied successfully,
 	   1 if there was an error when allocating memory,
 	   2 if arguments were invalid.
 */
int Image_copy(Image *src, Image **dst);

/**
 * Computes the offset of a pixel in the image data.
 * @param image The image.
 * @param x Column number of the pixel.
 * @param y Row number of the pixel.
 * @return The offset of the pixel in the data field. -1 if invalid.
 */
int Image_getOffset(Image *image, int x, int y);

/**
 * Gets the value of a pixel in a given channel.
 * @param image The image.
 * @param x Column number of the pixel.
 * @param y Row number of the pixel.
 * @param c The channel number.
 * @return The value of this pixel at the given channel.
 */
uint8_t Image_getPixel(Image *image, int x, int y, int c);

/**
 * Sets the value of a pixel in a given channel.
 * @param image The image.
 * @param x Column number of the pixel.
 * @param y Row number of the pixel.
 * @param c The channel number.
 * @param value Value of the pixel.
 */
void Image_setPixel(Image *image, int x, int y, int c, uint8_t value);

#ifdef __cplusplus
}
#endif

#endif