/**
 * @file unittest.h
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 */

#ifndef _UNITTEST_H
#define _UNITTEST_H

#include <stdio.h>

#define ut_assert(message, test) do { if (!(test)) printf("[FAIL] %s:%d: %s\n", __FILE__, __LINE__, message); } while (0)

#endif