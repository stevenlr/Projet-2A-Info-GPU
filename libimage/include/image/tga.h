/**
 * @file tga.h
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 */

#ifndef _TGA_H
#define _TGA_H

#ifdef __cplusplus
extern "C" {
#endif

#include "image/image.h"

/**
 * Reads a Truevision Targa image.
 * @param filename Path to the image to open.
 * @param imageptr Pointer to the image to create.
 * @return 0 if the image was read successfully,
 *         1 if there was an error when allocating memory,
 *	   2 if arguments were invalid,
 *	   3 if the image format is not supported.
 */
int TGA_readImage(const char *filename, Image **imageptr);

/**
 * Writes an image to a Truevision Targa format file.
 * @param filename Path to the image to write.
 * @param image The image to create.
 * @return 0 if the image was saved successfully,
 *         1 if there was an error when writing the image.
 */
int TGA_writeImage(const char *filename, Image *image);

#ifdef __cplusplus
}
#endif

#endif