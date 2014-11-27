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
		printf("Usage: %s <type> <image1> <image2>", argv[0]);
		return 0;
	}

	Image *ref_image;
	Image *test_image;
	int error = 0;
	char filePathReference[100];
	char filePathTest[100];

	sprintf(filePathReference, argv[1]);
	sprintf(filePathTest, argv[2]);

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
	uint8_t *datar, *datat, datad;
	float averageDifference = 0;

	datar = ref_image->data;
	datat = test_image->data;

	size = ref_image->width * ref_image->height * ref_image->channels;

	for (i = 0; i < size; ++i) {

		datad = abs((*datar++) - (*datat++));

		if (datad > 0) {
			averageDifference += datad;
			++error;
		}
	}

	float differencePercent = ((float) error) / (ref_image->height * ref_image->width) * 100;
	averageDifference /= (float) (ref_image->height * ref_image->width) * 100;

	if (error == 0) {
		printf("OK!\n");
	} else {
		printf("Failed! (Difference : %.3f%%) (Average difference per pixel : %.3f)\n", differencePercent, averageDifference);
	}

	return 0;
}
