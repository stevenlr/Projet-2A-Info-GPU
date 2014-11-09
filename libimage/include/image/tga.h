/**
 * @file tga.h
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 */

#ifndef _TGA_H
#define _TGA_H

#include "image/image.h"

/**
 * Reads a Truevision Targa image.
 * @param filename Path to the image to open.
 * @param imageptr Pointer to the image to create.
 * @return 0 if the image was read successfully.
 *         1 if there was an error when allocating memory,
 *	   2 if arguments were invalid.
 */
int TGA_readImage(const char *filename, Image **imageptr);

#endif