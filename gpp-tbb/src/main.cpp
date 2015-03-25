/**
 * @file main.cpp
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cstdint>

#include "add.h"
#include "invert.h"
#include "threshold.h"
#include "erosion.h"
#include "convolution.h"

using namespace std;

int check_sse()
{
	uint32_t highest_reg;
	uint32_t infos_d, infos_c;
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
		: "=d" (infos_d), "=c" (infos_c)
		: "a" (param)
	);

	const uint32_t sse_flag = 1 << 25;
	const uint32_t sse2_flag = 1 << 26;

	return (sse_flag & infos_d) && (sse2_flag & infos_d);
}

int main(int argc, char *argv[])
{
	if (!check_sse()) {
		cout << "Error : SSE or SSE2 not supported." << endl;
		return 1;
	}

	if (argc < 3) {
		cout << "Usage: " << argv[0] << "<filter> <input1> <input2> ... <inputn> <output>\n" << endl;
		cout << "Filters:" << endl;
		cout << "\tadd: 2 input images." << endl;
		cout << "\tinvert: 1 input image." << endl;
		cout << "\tthreshold: 1 input image, 1 input number [0; 255]." << endl;
		cout << "\tconvolution: 1 input image, 1 input matrix." << endl;
		cout << "\terosion: 1 input image, 1 input positive number.\n" << endl;
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
		cout << "Filter not implemented." << endl;
	}

	return 0;
}
