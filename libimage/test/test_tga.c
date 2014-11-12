/**
 * @file test_tga.h
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <image/tga.h>
#include <unittest.h>

void test_tga_open()
{
	Image *img_color;
	Image *img_gray;
	const char *file_rgb = "test/rgb4x2.tga";
	const char *file_gray = "test/grayscale.tga";
	const char *file_compressed = "test/compressed.tga";
	const char *file_colormapped = "test/color-mapped.tga";

	ut_assert("null image pointer", TGA_readImage(file_rgb, NULL) == 2);
	ut_assert("file not readable", TGA_readImage("doesnotexists", &img_color) == 2);

	ut_assert("compressed not supported", TGA_readImage(file_compressed, &img_color) == 3);
	ut_assert("color-mapped not supported", TGA_readImage(file_colormapped, &img_color) == 3);

	ut_assert("open color image", TGA_readImage(file_rgb, &img_color) == 0);
	ut_assert("open gray image", TGA_readImage(file_gray, &img_gray) == 0);

	uint8_t colorsequence[] = {0x00, 0x00, 0x00,
				   0xff, 0x00, 0x00,
				   0x00, 0xff, 0x00,
				   0xff, 0xff, 0x00,
				   0x00, 0x00, 0xff,
				   0xff, 0x00, 0xff,
				   0x00, 0xff, 0xff,
				   0xff, 0xff, 0xff};

	ut_assert("color image width", img_color->width == 4);
	ut_assert("color image height", img_color->height == 2);
	ut_assert("color image channels", img_color->channels == 3);
	ut_assert("color image data", memcmp(img_color->data, colorsequence, 4 * 2 * 3 * sizeof(uint8_t)) == 0);

	ut_assert("gray image width", img_gray->width == 256);
	ut_assert("gray image height", img_gray->height == 256);
	ut_assert("gray image channels", img_gray->channels == 1);

	srand(time(NULL));
	int i, error = 0;

	for (i = 0; i < 1000; i++) {
		int x = rand() % img_gray->width;
		int y = rand() % img_gray->height;
		uint8_t result = (x + y >= 256) ? 255 : (x + y);
		uint8_t color = Image_getPixel(img_gray, x, y, 0);

		if (color != result)
			error = 1;
	}

	ut_assert("gray image data", error == 0);

	Image_delete(img_color);
	Image_delete(img_gray);
}

void test_tga_all()
{
	test_tga_open();
}

