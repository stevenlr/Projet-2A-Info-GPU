#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
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

int invert(int argc, char* argv[]) {
	if (argc != 3) {
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

	Opencl_launcher ocl(argv[0]);
	cl_int error;
	cl_kernel invert_kernel = ocl.load_kernel("src/invert_kernel.cl", "invert");
	cl_context context = ocl.get_context();
	cl_command_queue queue = ocl.get_queue();

	int size = input_image->height * input_image->width / 16;
	const int mem_size = sizeof(cl_uchar16) * size;
	cl_mem data;
	const size_t local_ws = 192;
	const size_t global_ws = shrRoundUp(local_ws, size);
	cl_uchar16* dataInput; 
	cl_event event;

	for (int c = 0; c < input_image->channels; ++c) {
		dataInput =  (cl_uchar16*) output_image->data[c];
		data = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size, dataInput, &error);
		assert(error == CL_SUCCESS);

		error = clSetKernelArg(invert_kernel, 0, sizeof(cl_mem), &data);
		error |= clSetKernelArg(invert_kernel, 1, sizeof(size_t), &size);
		assert(error == CL_SUCCESS);

		clFinish(queue);
		error = clEnqueueNDRangeKernel(queue, invert_kernel, 1, NULL, &global_ws, &local_ws, 0, NULL, &event);
		assert(error == CL_SUCCESS);
		clWaitForEvents(1 , &event);

		ocl.benchmark(event, "Execution time");

		error = clEnqueueReadBuffer(queue, data, CL_TRUE, 0, mem_size, dataInput, 0, NULL, &event);

		ocl.benchmark(event, "Transfer time");
		assert(error == CL_SUCCESS);

		ocl.total_time();

		output_image->data[c] = (uint8_t*) dataInput;
	}

	if ((errortga = TGA_writeImage(argv[2], output_image)) != 0) {
		cout << "Error when writing image: " << errortga << endl;
	}

	Image_delete(input_image);
	Image_delete(output_image);

	return 1;
}