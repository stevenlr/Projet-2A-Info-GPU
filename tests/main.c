/**
 * @file main.c
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int launchTestOf(char *type, char *filters[]) {

} // par GPP / CUDA / OpenCL

int main(int argc, char *argv[])
{
	if (argc < 2) {
		printf("Usage: %s <type> <filter1> <filter2> <filtern>", argv[0]);
		return 0;
	}

	if (strcmp(argv[1], "all") == 0) {
	} else if (strcmp(argv[1], "gpp") == 0) {
		launchTestOf("gpp", argv + 2, argc - 2);
	} else if (strcmp(argv[1], "cuda") == 0) {
		printf("Not implemented.\n");
	} else if (strcmp(argv[1], "opencl") == 0) {
		printf("Not implemented.\n");
	} else {
		printf("Type not found.\n");
	}
}