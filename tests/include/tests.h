/**
 * @file tests.h
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#ifndef _TESTS_H
#define _TESTS_H

/**
 * Runs the tests according to the type and desired filters.
 * @param type Architecture type desired.
 * @param filters table of filters desired with their arguments.
 * @param number of filters and their arguments.
 */
void launchTestsOf(char *type, char *filters[], int argc);

/**
 * Executes the test according to the type and the filter.
 * (Compares the reference image with the output image from 
  * the type of Architure choosed.)
 * @param type Architecture type desired.
 * @param filter filter desired with their arguments.
 * @return the result of the test (boolean).
 */
int test(char *type, char* filter);

#endif