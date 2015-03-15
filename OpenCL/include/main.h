/**
 * @file main.h
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#ifndef _MAIN_H
#define _MAIN_H

char errorMessage(int c);


size_t shrRoundUp(size_t localWorkSize, size_t numItems);


const char* getSource(const char* filePath);

#endif