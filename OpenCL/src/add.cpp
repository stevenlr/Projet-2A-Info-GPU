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

int add(int argc, char* argv[]) {
	if (argc != 3) {
		cout << "Invalid number of arguments." << endl;
		return 0;
	}

	Image *input_image1;
	Image *input_image2;
	Image *output_image;
	int errortga;

	if ((errortga = TGA_readImage(argv[0], &input_image1)) != 0) {
		cout << "Error when opening image: " << errortga << endl;
		return 0;
	}

	if ((errortga = TGA_readImage(argv[1], &input_image2)) != 0) {
		cout << "Error when opening image: " << errortga << endl;
		Image_delete(input_image1);
		return 0;
	}

	if ((errortga = Image_copy(input_image1, &output_image)) != 0) {
		cout << "Error when copying image: " << errortga << endl;
		Image_delete(input_image1);
		Image_delete(input_image2);
		return 0;
	}

	Opencl_launcher ocl;
	cl_int error;
	cl_kernel add_kernel = ocl.load_kernel("src/add_kernel.cl", "add");
	cl_context context = ocl.getContext();
	cl_command_queue queue = ocl.getQueue();

	int size = input_image1->height * input_image1->width;
	const int mem_size = sizeof(uint8_t) * size;
	cl_mem data1, data2;

	const size_t local_ws = 128;
	const size_t global_ws = shrRoundUp(local_ws, size);

	cl_event event;

	for (int c = 0; c < input_image1->channels; ++c) {
		data1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size, output_image->data[c], &error);
		data2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, input_image2->data[c], &error);
		assert(error == CL_SUCCESS);

		error = clSetKernelArg(add_kernel, 0, sizeof(cl_mem), &data1);
		error |= clSetKernelArg(add_kernel, 1, sizeof(cl_mem), &data2);
		error |= clSetKernelArg(add_kernel, 2, sizeof(size_t), &size);
		assert(error == CL_SUCCESS);

		clFinish(queue);
		error = clEnqueueNDRangeKernel(queue, add_kernel, 1, NULL, &global_ws, &local_ws, 0, NULL, &event);
		assert(error == CL_SUCCESS);
		clWaitForEvents(1 , &event);

		ocl.benchmark(event, "Execution time");

		error = clEnqueueReadBuffer(queue, data1, CL_TRUE, 0, mem_size, output_image->data[c], 1, &event, &event);

		ocl.benchmark(event, "Transfer time");
		assert(error == CL_SUCCESS);
	}

	if ((errortga = TGA_writeImage(argv[2], output_image)) != 0) {
		cout << "Error when writing image: " << errortga << endl;
	}

	Image_delete(input_image1);
	Image_delete(input_image2);
	Image_delete(output_image);

	return 1;
}