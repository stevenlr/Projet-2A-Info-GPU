/**
 * @file tests.c
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <image/image.h>
#include <image/tga.h>
#include "tests.h"

void launchTestsOf(char *type, char *filters[], int argc)
{
	int i;
	char fileName[100];

	for (i = 0; i < argc; ++i) {
		if (strcmp(filters[i], "all") == 0) {
			test(type, "add");
			test(type, "invert");
			test(type, "threshold");
			test(type, "convolution-3x3sharpen-extend");
			test(type, "convolution-7x7gaussian-extend");
			test(type, "erosion-colors");
			test(type, "erosion-binary");
		} else if (strcmp(filters[i], "add") == 0) {
			test(type, filters[i]);
		} else if (strcmp(filters[i], "invert") == 0) {
			test(type, filters[i]);
		} else if (strcmp(filters[i], "threshold") == 0) {
			test(type, filters[i]);
		} else if (strcmp(filters[i], "convolution") == 0) {
			if (++i < argc) {
				sprintf(fileName, "%s-%s-extend", filters[i-1], filters[i]);
				test(type, fileName);
			} else {
				printf("Error number of arguments for convolution"\
					"(Matrice not found)");
			}
		} else if (strcmp(filters[i], "erosion") == 0) {
			if (++i < argc) {
				sprintf(fileName, "%s-%s", filters[i-1], filters[i]);
				test(type, fileName);
			} else {
				printf("Error number of arguments for erosion"\
					"(Mode not found)");
			}
		} else {
			printf("Filter not implemented.");
		}
	}
}

int test(char *type, char *filter)
{
	Image *ref_image;
	Image *test_image;
	int error = 0;
	char filePathReference[100];
	char filePathTest[100];

	sprintf(filePathReference, "../references/output-images/%s.tga", filter);
	sprintf(filePathTest, "../%s/output-images/%s.tga", type, filter);

	if ((error = TGA_readImage(filePathReference, &ref_image)) != 0) {
		printf("Error when opening image: %d (%s)\n", error, 
			filePathReference);
		printf("%s", filePathReference);
		return 0;
	}

	if ((error = TGA_readImage(filePathTest, &test_image)) != 0) {
		printf("Error when opening image: %d (%s)\n", error, filePathTest);
		printf("%s", filePathTest);
		return 0;
	}

	if (test_image->height != ref_image->height ||
		test_image->width != ref_image->width ) {
		printf("Error : Images should be the same size.");
		return 0;
	}

	int size, i;
	uint8_t *datar, *datat, datad, bestDifference = 0;

	datar = ref_image->data;
	datat = test_image->data;

	size = ref_image->width * ref_image->height * ref_image->channels;

	for (i = 0; i < size; ++i) {

		datad = abs((*datar++) - (*datat++));

		if (datad > 0) {
			if (datad > bestDifference)
				bestDifference = datad;
			++error;
		}
	}

	float differencePercent = ((float) error) / (ref_image->height * ref_image->width) * 100;

	if (error == 0) {
		printf("%s-%s: OK!\n", type, filter);
	} else {
		printf("%s-%s: Failed! (Difference : %.2f%%) (Highest difference for a pixel : %d)\n", type, filter, differencePercent, bestDifference);
	}

	return error == 0;
}
