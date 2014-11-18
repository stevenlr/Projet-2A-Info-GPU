/**
 * @file tests.c
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int launchTestsOf(char *type, char *filters[], int argc)
{
	int i;
	char *matriceFileName;

	for (i = 0; i < argc; ++i) {
		if (strcmp(filters[i], "add" == 0) {
			add(type);
		} else if (strcmp(filters[i], "invert" == 0) {
			threshold(type);
		} else if (strcmp(filters[i], "threshold" == 0) {
			invert(type);
		} else if (strcmp(filters[i], "convolution" == 0) {
			++i;
			matriceFileName = filters[i];
			convolution(type, matriceFileName);
		} else if (strcmp(filters[i], "erosion" == 0) {
			erosion(type);
		} else {
			printf("Filter not implemented.");
		}
	}

	return 0;
}

int add(char *type)
{
	Image *ref_image;
	Image *test_image;
	int difference = 0;
	char filePathReferences[100] = "../references/output-images/add.tga";
	char filePathTest[100] = strcat(strcat("../", type), "/output-images/add.tga");

	if ((error = TGA_readImage(filePathReferences, &ref_image)) != 0) {
		printf("Error when opening image: %d\n", error);
		return;
	}

	if ((error = TGA_readImage(filePathTest, &test_image)) != 0) {
		printf("Error when opening image: %d\n", error);
		return;
	}

	if (test_image->height != ref_image->height ||
		test_image->width != ref_image->width ) {
		printf("Error : Images should be the same size.");
		return;
	}


}

int threshold(char *type);
int invert(char *type);
int convolution(char *type, char *matriceFileName);
int erosion(char *type);