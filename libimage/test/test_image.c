/**
 * @file test_imae.h
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
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

void test_image_copy()
{
	Image *img1, *img2;

	Image_new(8, 8, 1, &img1);
	srand(time(NULL));

	for (int i = 0; i < img1->width * img1->height; ++i) {
		img1->data[i] = rand() % 256;
	}

	ut_assert("copy null src image", Image_copy(NULL, &img2) == 2);
	ut_assert("copy null dst image", Image_copy(img1, NULL) == 2);
	ut_assert("copy image", Image_copy(img1, &img2) == 0);

	ut_assert("copy image width", img1->width == img2->width);
	ut_assert("copy image height", img1->height == img2->height);
	ut_assert("copy image channels", img1->channels == img2->channels);

	ut_assert("copy image data", memcmp(img1->data, img2->data, img1->width * img1->height * sizeof(uint8_t)) == 0);

	Image_delete(img1);
	Image_delete(img2);
}

void test_image_all()
{
	test_image_new_delete();
	test_image_copy();
	test_ulage_data();
}