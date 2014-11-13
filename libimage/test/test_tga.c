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

void test_tga_read()
{
	Image *img_color;
	Image *img_gray;
	const char *file_rgb = "test/rgb4x2.tga";
	const char *file_gray = "test/grayscale.tga";
	const char *file_compressed = "test/compressed.tga";
	const char *file_colormapped = "test/color-mapped.tga";

	ut_assert("null image pointer", TGA_readImage(file_rgb, NULL) == 2);
	ut_assert("file not readable", TGA_readImage("doesnotexist", &img_color) == 2);

	ut_assert("compressed not supported", TGA_readImage(file_compressed, &img_color) == 3);
	ut_assert("color-mapped not supported", TGA_readImage(file_colormapped, &img_color) == 3);

	ut_assert("open color image", TGA_readImage(file_rgb, &img_color) == 0);
	ut_assert("open gray image", TGA_readImage(file_gray, &img_gray) == 0);

	uint8_t color_sequence[] = {0x00, 0x00, 0x00,
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
	ut_assert("color image data", memcmp(img_color->data, color_sequence, 4 * 2 * 3 * sizeof(uint8_t)) == 0);

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

void test_tga_write()
{
	Image *image_color;
	const char *file_rgb = "testimage_color.tga";

	uint8_t color_sequence[] = {0x00, 0x00, 0x00,
				   0xff, 0x00, 0x00,
				   0x00, 0xff, 0x00,
				   0xff, 0xff, 0x00,
				   0x00, 0x00, 0xff,
				   0xff, 0x00, 0xff,
				   0x00, 0xff, 0xff,
				   0xff, 0xff, 0xff};
	size_t color_data_size = 42;
	uint8_t image_color_data[] = {
		0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x02, 0x00,
		0x18, 0x00, 0xff, 0x00, 0x00, 0xff, 0x00, 0xff,
		0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0x00, 0x00,
		0x00, 0x00, 0x00, 0xff, 0x00, 0xff, 0x00, 0x00,
		0xff, 0xff};

	if (Image_new(4, 2, 3, &image_color)) {
		printf("Error while creating color test image\n");
		return;
	}

	ut_assert("write invalid image", TGA_writeImage(file_rgb, NULL) == 1);

	memcpy(image_color->data, color_sequence, 4 * 2 * 3 * sizeof(uint8_t));
	ut_assert("write color image", TGA_writeImage(file_rgb, image_color) == 0);

	uint8_t image_color_data_read[color_data_size];
	FILE* fp = fopen(file_rgb, "rb");

	if (fp == NULL) {
		printf("error when opening color test file\n");
		return;
	}

	fread(image_color_data_read, sizeof(uint8_t), color_data_size, fp);
	fclose(fp);

	ut_assert("color image data", memcmp(image_color_data, image_color_data_read, color_data_size) == 0);

	Image_delete(image_color);

	Image *image_gray;
	const char *file_gray = "testimage_gray.tga";
	size_t gray_data_size = 26;
	uint8_t gray_sequence[] = {
		0x00, 0x20, 0x40, 0x60,
		0x80, 0xa0, 0xc0, 0xe0};
	uint8_t image_gray_data[] = {
		0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x02, 0x00,
		0x08, 0x00, 0x80, 0xa0, 0xc0, 0xe0, 0x00, 0x20,
		0x40, 0x60};

	if (Image_new(4, 2, 1, &image_gray)) {
		printf("Error while creating gray test image\n");
		return;
	}

	memcpy(image_gray->data, gray_sequence, 4 * 2 * 1 * sizeof(uint8_t));
	ut_assert("write gray image", TGA_writeImage(file_gray, image_gray) == 0);

	uint8_t image_gray_data_read[gray_data_size];
	fp = fopen(file_gray, "rb");

	if (fp == NULL) {
		printf("error when opening gray test file\n");
		return;
	}

	fread(image_gray_data_read, sizeof(uint8_t), gray_data_size, fp);
	fclose(fp);

	ut_assert("gray image data", memcmp(image_gray_data, image_gray_data_read, gray_data_size) == 0);

	Image_delete(image_gray);

	remove(file_gray);
	remove(file_rgb);
}

void test_tga_all()
{
	test_tga_read();
	test_tga_write();
}
