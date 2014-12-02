/**
 * @file main.c
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "invert.h"
#include "threshold.h"
#include "add.h"
#include "erosion.h"
#include "convolution.h"
#include "benchmark.h"

int main(int argc, char *argv[])
{
	if (argc < 2) {
		printf("Usage: %s <filter> <input1> <input2> ... <inputn> <output>\n", argv[0]);
		printf("Filters:\n");
		printf("\tadd: 2 input images.\n");
		printf("\tinvert: 1 input image.\n");
		printf("\tthreshold: 1 input image, 1 input number [0; 255].\n");
		printf("\tconvolution: 1 input image, 1 input matrix.\n");
		printf("\terosion: 1 input image, 1 input positive number.\n\n");
		return 0;
	}

	if (strcmp(argv[1], "invert") == 0) {
		invert(argc - 2, argv + 2);
	} else if (strcmp(argv[1], "threshold") == 0) {
		threshold(argc - 2, argv + 2);
	} else if (strcmp(argv[1], "add") == 0) {
		add(argc - 2, argv + 2);
	} else if (strcmp(argv[1], "convolution") == 0) {
		//convolution(argc - 2, argv + 2);
	} else if (strcmp(argv[1], "erosion") == 0) {
		//erosion(argc - 2, argv + 2);
	} else {
		printf("Filter not implemented.\n");
	}

	return 0;
}
