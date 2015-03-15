/**
 * @file main.c
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <image/image.h>
#include <image/tga.h>

int main(int argc, char *argv[])
{
	if (argc < 3) {
		printf("Usage: %s <image1> <image2>\n", argv[0]);
		return 0;
	}

	Image *ref_image;
	Image *test_image;
	long error = 0;
	char *file_path_reference = argv[1];
	char *file_path_test = argv[2];

	if ((error = TGA_readImage(file_path_reference, &ref_image)) != 0) {
		printf("Error when opening image: %d (%s)\n", error, file_path_reference);
		return 0;
	}

	if ((error = TGA_readImage(file_path_test, &test_image)) != 0) {
		printf("Error when opening image: %d (%s)\n", error, file_path_test);
		return 0;
	}

	if (test_image->height != ref_image->height ||
		test_image->width != ref_image->width ||
		test_image->channels != ref_image->channels) {
		printf("Error : Images should be the same size.\n");
		return 0;
	}

	int size, i, c;
	uint8_t *datar, *datat, datad;
	float average_difference = 0;

	size = ref_image->width * ref_image->height * ref_image->channels;

	for (c = 0; c < ref_image->channels; ++c) {
		datar = ref_image->data[c];
		datat = test_image->data[c];

		for (i = 0; i < size; ++i) {
			datad = abs((*datar++) - (*datat++));

			if (datad > 0) {
				average_difference += datad;
				++error;
			}
		}
	}


	float difference_percent = average_difference / (((float) ref_image->height) * ref_image->width * ref_image->channels * 255) * 100;
	average_difference /= (float) error;

	if (error == 0) {
		printf("OK!\n");
	} else {
		printf("Failed! (Difference : %.3f%%) (Average difference : %.3f)\n", difference_percent, average_difference);
	}

	return 0;
}
