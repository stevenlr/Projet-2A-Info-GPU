/**
 * @file memory.h
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#ifndef _MEMORY_H
#define _MEMORY_H

/**
 * Allocates aligned memory.
 * @param size Memory to allocate.
 * @param align Desired lignment.
 */
void *aligned_malloc(size_t size, size_t align);

/**
 * Deallocates aligned memory allocated width aligned_malloc.
 * @param ptr Pointer to deallocate.
 */
void aligned_free(void *ptr);

#endif