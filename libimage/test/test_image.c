/**
 * @file test_imae.h
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <image/image.h>
#include <unittest.h>

void test_image_new_delete()
{
	Image *img;

	ut_assert("0-size image", Image_new(0, 2, 3, &img) == 2);
	ut_assert("negative-size image", Image_new(4, -2, 3, &img) == 2);
	ut_assert("unsupported channel number -4", Image_new(4, 2, -4, &img) == 2);
	ut_assert("unsupported channel number 2", Image_new(4, 2, 2, &img) == 2);
	ut_assert("unsupported channel number 4", Image_new(4, 2, 4, &img) == 2);
	ut_assert("NULL image pointer", Image_new(4, 2, 3, NULL) == 2);
	ut_assert("correct image creation", Image_new(4, 2, 3, &img) == 0);

	ut_assert("new image width", img->width == 4);
	ut_assert("new image height", img->height == 2);
	ut_assert("new image channels", img->channels == 3);
	ut_assert("new image data", img->data != NULL);

	ut_assert("image deletion", Image_delete(img) == 0);
	ut_assert("NULL image deletion", Image_delete(NULL) == 1);

	Image img2;
	img2.data = NULL;

	ut_assert("NULL data image deletion", Image_delete(&img2) == 1);
}

void test_image_all()
{
	test_image_new_delete();
}