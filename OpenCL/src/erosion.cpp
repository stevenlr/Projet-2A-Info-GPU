/**
 * @file erosion.cpp
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
#include <CL/opencl.h>
#include <image/image.h>
#include <image/tga.h>

#include "main.h"
#include "opencl_launcher.h"

using namespace std;

#define TILE_SIZE 128
#define PROCESSED_SIZE 32
#define MAX_RADIUS ((TILE_SIZE - PROCESSED_SIZE - 1) / 2)

//#define ALGO2N
#define SHARED

int erosion(int argc, char* argv[]) {
	if (argc != 5) {
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

	int rx = atoi(argv[2]);
	int ry = atoi(argv[3]);

	if (rx < 0 || rx >= MAX_RADIUS || ry < 0 || ry >= MAX_RADIUS) {
		printf("Invalid radius value.\n");
		Image_delete(input_image);
		return 1;
	}

	int height = input_image->height, width = input_image->width;
	int size = height * width;
	const int mem_size = sizeof(cl_uchar) * size;
	cl_mem data, dataInput;
	
	const size_t local_ws[2] = {PROCESSED_SIZE, PROCESSED_SIZE};
	const size_t global_ws[2] = {shrRoundUp(local_ws[0], width), shrRoundUp(local_ws[1], height)};

	#ifdef SHARED
		const string kernel_name = "src/kernel/erosion_kernel2.cl";
	#else
		const string kernel_name = "src/kernel/erosion_kernel.cl";
	#endif

	Opencl_launcher ocl(argv[0]);
	cl_int error;
	cl_kernel erosion_kernel = ocl.load_kernel(kernel_name.c_str(), "erosion");
	cl_context context = ocl.get_context();
	cl_command_queue queue = ocl.get_queue();
	cl_event event;

	for (int c = 0; c < input_image->channels; ++c) {
		data = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size, output_image->data[c], &error);

		#ifdef ALGO2N
		ry = 0;
		dataInput = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size, input_image->data[c], &error);
		error = clSetKernelArg(erosion_kernel, 0, sizeof(cl_mem), &data);
		error |= clSetKernelArg(erosion_kernel, 1, sizeof(cl_mem), &dataInput);
		#else
		dataInput = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, input_image->data[c], &error);
		error = clSetKernelArg(erosion_kernel, 0, sizeof(cl_mem), &dataInput);
		error |= clSetKernelArg(erosion_kernel, 1, sizeof(cl_mem), &data);
		#endif

		error |= clSetKernelArg(erosion_kernel, 2, sizeof(size_t), &width);
		error |= clSetKernelArg(erosion_kernel, 3, sizeof(size_t), &height);
		error |= clSetKernelArg(erosion_kernel, 4, sizeof(int), &rx);
		error |= clSetKernelArg(erosion_kernel, 5, sizeof(int), &ry);
		assert(error == CL_SUCCESS);

		clFinish(queue);
		error = clEnqueueNDRangeKernel(queue, erosion_kernel, 2, NULL, global_ws, local_ws, 0, NULL, &event);

		assert(error == CL_SUCCESS);
		clWaitForEvents(1 , &event);

		#ifdef ALGO2N 
		ocl.benchmark(event, "ExecD");
		ry = atoi(argv[3]);
		rx = 0;

		error = clSetKernelArg(erosion_kernel, 0, sizeof(cl_mem), &dataInput);
		error |= clSetKernelArg(erosion_kernel, 1, sizeof(cl_mem), &data);
		error |= clSetKernelArg(erosion_kernel, 2, sizeof(size_t), &width);
		error |= clSetKernelArg(erosion_kernel, 3, sizeof(size_t), &height);
		error |= clSetKernelArg(erosion_kernel, 4, sizeof(int), &rx);
		error |= clSetKernelArg(erosion_kernel, 5, sizeof(int), &ry);
		assert(error == CL_SUCCESS);

		clFinish(queue);
		error = clEnqueueNDRangeKernel(queue, erosion_kernel, 2, NULL, global_ws, local_ws, 0, NULL, &event);
		assert(error == CL_SUCCESS);
		clWaitForEvents(1 , &event);
		#endif

		ocl.benchmark(event, "Execution time");
		

		error = clEnqueueReadBuffer(queue, data, CL_TRUE, 0, mem_size, output_image->data[c], 0, NULL, &event);

		ocl.benchmark(event, "Transfer time");
		assert(error == CL_SUCCESS);

		ocl.total_time();
	}
	if ((errortga = TGA_writeImage(argv[4], output_image)) != 0) {
		cout << "Error when writing image: " << errortga << endl;
	}

	Image_delete(input_image);
	Image_delete(output_image);

	return 1;
}