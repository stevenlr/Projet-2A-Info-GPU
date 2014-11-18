/**
 * @file tests.h
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

int launchTestsOf(char *type, char *filters[]);

int add(char *type);
int threshold(char *type);
int invert(char *type);
int convolution(char *type, char *matriceFileName);
int erosion(char *type);