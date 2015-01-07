/**
 * @file main.cpp
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#include <cstdlib>
#include <iostream>
#include <cstring>

#include "add.h"
#include "invert.h"
#include "threshold.h"
#include "erosion.h"
#include "convolution.h"

using namespace std;

int main(int argc, char *argv[])
{
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
