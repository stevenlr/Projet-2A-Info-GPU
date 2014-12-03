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

int check_sse2()
{
	uint32_t highest_reg;
	uint32_t infos;
	uint32_t param = 0;

	__asm__ (
		"cpuid\n\t"
		: "=a" (highest_reg)
		: "a" (param)
	);

	if (highest_reg < 1)
		return 0;

	param = 1;
	__asm__ (
		"cpuid\n\t"
		: "=d" (infos)
		: "a" (param)
	);

	const uint32_t sse_flag = 1 << 25;
	const uint32_t sse2_flag = 1 << 26;

	return (sse_flag & infos) && (sse2_flag & infos);
}

int main(int argc, char *argv[])
{
	if (!check_sse2()) {
		printf("Error : SSE2 not supported.\n");
	}

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
		convolution(argc - 2, argv + 2);
	} else if (strcmp(argv[1], "erosion") == 0) {
		erosion(argc - 2, argv + 2);
	} else {
		printf("Filter not implemented.\n");
	}

	return 0;
}
