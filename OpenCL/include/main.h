/**
 * @file main.h
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#ifndef _MAIN_H
#define _MAIN_H

/**
 * Displays the error code.
 * @param c error code.
 * @return char error code as a char
 */
char errorMessage(int c);

/**
 * Rounds up the global work size in function of the local work size and the numbers of items
 * @param localWorkSize Size of local work in OpenCL
 * @param numItems Number of items
 * @return global work size
 */
size_t shrRoundUp(size_t localWorkSize, size_t numItems);

/**
 * Gets the source of a file
 * @param filePath Path of the source file
 * @return Code source in a char*
 */
const char* getSource(const char* filePath);

#endif