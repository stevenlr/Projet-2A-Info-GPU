/**
 * @file memory.c
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#include <stdlib.h>
#include <stdint.h>
#include "memory.h"

 // Modified from http://stackoverflow.com/a/1920516
void *aligned_malloc(size_t size, size_t align)
{
	void *mem = malloc(size + align + sizeof(void*));
	void **ptr = (void **)((uintptr_t) (mem + align + sizeof(void*)) & ~(align - 1));
	ptr[-1] = mem;
	return ptr;
}

void aligned_free(void *ptr)
{
	free(((void**) ptr)[-1]);
}