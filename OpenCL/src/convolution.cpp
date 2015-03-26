/**
 * @file convolution.cpp
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <string>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <CL/opencl.h>
#include <image/image.h>
#include <image/tga.h>

#include "main.h"
#include "opencl_launcher.h"

using namespace std;

#define PROCESSED_SIZE 32
#define PROCESSED_SIZE_INTEL 16

typedef struct {
	int width;
	int height;
	float sum;
	float *data;
} Kernel;

/**
 * Opens a convolution kernel from a file.
 * @param file_name Path to the kernel file.
 * @param kernel Kernel to write data in.
 * @return 0 if the kernel was retrieved successfully,
 *	   1 if there was an error when allocating memory,
 *	   2 if arguments were invalid,
 *	   3 if the matrix dimensions are invalid. (Has to be odd)
 */
int get_kernel(const char *file_name, Kernel *kernel)
{
	FILE *fp;

	if (kernel == NULL) {
		return 2;
	}

	kernel->data = NULL;
	kernel->sum = 0;

	if ((fp = fopen(file_name, "r")) == NULL) {
		return 2;
	}

	fscanf(fp, "%d*%d", &(kernel->width), &(kernel->height));

	if (kernel->width % 2 != 1 || kernel->height % 2 != 1
		|| kernel->width < 0 || kernel->height < 0) {
		fclose(fp);
		return 3;
	}

	size_t size = kernel->width * kernel->height;
	unsigned int i;

	kernel->data = (float *) malloc(sizeof(float) * size);

	if (kernel->data == NULL) {
		fclose(fp);
		return 1;
	}

	for (i = 0; i < size; ++i) {
		fscanf(fp, "%f", kernel->data + i);
		kernel->sum += kernel->data[i];
	}

	if (kernel->sum >= -FLT_EPSILON && kernel->sum <= FLT_EPSILON) {
		kernel->sum = 1;
	}

	fclose(fp);

	return 0;
}

void kernel_delete(Kernel kernel)
{
	if (kernel.data != NULL) {
		free(kernel.data);
	}
}

int convolution(int argc, char* argv[]) {
	if (argc != 4) {
		cout << "Invalid number of arguments." << endl;
		return 0;
	}

	Image *input_image;
	Image *output_image;
	int errortga;

	if ((errortga = TGA_readImage(argv[1], &input_image)) != 0) {
		cout << "Error when opening image: " << errortga << endl;
		return 0;
	}

	if ((errortga = Image_copy(input_image, &output_image)) != 0) {
		cout << "Error when copying image: " << errortga << endl;
		Image_delete(input_image);
		return 0;
	}

	Kernel kernel;

	if ((errortga = get_kernel(argv[2], &kernel)) != 0) {
		printf("Error when opening kernel : %d\n", errortga);
		Image_delete(input_image);
		Image_delete(output_image);
		return 0;
	} 

	Opencl_launcher ocl(argv[0]);
	cl_int error;
	cl_kernel convolution_kernel = ocl.load_kernel("src/kernel/convolution_kernel.cl", "convolution");
	cl_context context = ocl.get_context();
	cl_command_queue queue = ocl.get_queue();

	int height = input_image->height, width = input_image->width;
	int size = input_image->height * input_image->width / 16;
	const int mem_size = sizeof(cl_uchar16) * size;
	cl_mem data, dataInput, data_kernel;
	
	size_t local_ws[2];
	if (strncmp(argv[0], "Intel", 5) == 0) {
		local_ws[0] = PROCESSED_SIZE_INTEL; local_ws[1] = PROCESSED_SIZE_INTEL;
	}
	else {
		local_ws[0] = PROCESSED_SIZE; local_ws[1] = PROCESSED_SIZE;
	}
	const size_t global_ws[2] = {shrRoundUp(local_ws[0], width), shrRoundUp(local_ws[1], height)};
	cl_event event;

	for (int c = 0; c < input_image->channels; ++c) {
		dataInput = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, output_image->data[c], &error);
		assert(error == CL_SUCCESS);
		data = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size, output_image->data[c], &error);
		assert(error == CL_SUCCESS);
		data_kernel = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * kernel.width * kernel.height , kernel.data, &error);
		assert(error == CL_SUCCESS);

		error = clSetKernelArg(convolution_kernel, 0, sizeof(cl_mem), &dataInput);
		error |= clSetKernelArg(convolution_kernel, 1, sizeof(cl_mem), &data);
		error |= clSetKernelArg(convolution_kernel, 2, sizeof(size_t), &width);
		error |= clSetKernelArg(convolution_kernel, 3, sizeof(size_t), &height);
		error |= clSetKernelArg(convolution_kernel, 4, sizeof(cl_mem), &data_kernel);
		error |= clSetKernelArg(convolution_kernel, 5, sizeof(size_t), &(kernel.width));
		error |= clSetKernelArg(convolution_kernel, 6, sizeof(size_t), &(kernel.height));
		error |= clSetKernelArg(convolution_kernel, 7, sizeof(size_t), &(kernel.sum));
		assert(error == CL_SUCCESS);

		clFinish(queue);
		error = clEnqueueNDRangeKernel(queue, convolution_kernel, 2, NULL, global_ws, local_ws, 0, NULL, &event);
		assert(error == CL_SUCCESS);
		clWaitForEvents(1 , &event);

		ocl.benchmark(event, "Execution time");

		error = clEnqueueReadBuffer(queue, data, CL_TRUE, 0, mem_size, output_image->data[c], 0, NULL, &event);

		ocl.benchmark(event, "Transfer time");
		assert(error == CL_SUCCESS);

		ocl.total_time();
	}

	if ((errortga = TGA_writeImage(argv[3], output_image)) != 0) {
		cout << "Error when writing image: " << errortga << endl;
	}

	Image_delete(input_image);
	Image_delete(output_image);
	kernel_delete(kernel);

	return 1;
}