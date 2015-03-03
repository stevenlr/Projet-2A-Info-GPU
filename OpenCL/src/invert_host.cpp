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

using namespace std;

char errorMessage(int c) {
	return (char) c;
}


size_t shrRoundUp(size_t localWorkSize, size_t numItems) {
	size_t x = numItems % localWorkSize;
	if (!x) {
		return numItems;
	}
	return numItems + (localWorkSize - x) ;
}


const char* getSource(const char* filePath) {
	ifstream ifs;
	ifs.open(filePath);
	string s;

	char c;
	while(ifs.get(c)) {
		s += c;
	}

	return s.c_str();
}

int main(int argc, char* argv[]) {
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

	cl_int error = 0;	// Used to handle error codes
	cl_platform_id platform;
	cl_uint nbPlatforms;
	cl_context context;
	cl_command_queue queue;
	cl_device_id device;

	// Platform
	cl_platform_id* platforms;
	clGetPlatformIDs(0, NULL, &nbPlatforms);
	platforms = new cl_platform_id[nbPlatforms];
	clGetPlatformIDs(nbPlatforms, platforms, NULL);

	char vendor[100];
	for (int i = 0; i < nbPlatforms; ++i) {
		clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
		
		if (strncmp(vendor, "NVIDIA", 5) == 0)
			platform = platforms[i];
	}

	delete[] platforms;
	if (!platform){
		cout << "Error getting platform" << endl;
		exit(0);
	}
	// Device
	error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if (error != CL_SUCCESS) {
		cout << "Error getting device ids: " << errorMessage(error) << endl;
		exit(error);
	}
	// Context
	context = clCreateContext(0, 1, &device, NULL, NULL, &error);
	if (error != CL_SUCCESS) {
		cout << "Error creating context: " << errorMessage(error) << endl;
		exit(error);
	}
	// Command-queue
	queue = clCreateCommandQueue(context, device, 0, &error);
	if (error != CL_SUCCESS) {
		cout << "Error creating command queue: " << errorMessage(error) << endl;
		exit(error);
	}

	int size = input_image->height * input_image->width;
	const int mem_size = sizeof(uint8_t) * size;
	const char* path = "src/invert_kernel.cl";
	const char* source = getSource(path);
	size_t src_size[] = {strlen(source)};
	cl_program program = clCreateProgramWithSource(context, 1, &source, src_size, &error);
	assert(error == CL_SUCCESS);

	error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	switch (error) {
		case CL_BUILD_PROGRAM_FAILURE:
			cl_build_status b;
			error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &b, NULL);
			switch (error) {
				case CL_BUILD_IN_PROGRESS:
				 	cout << "progress" << endl;
				 	break;
				case CL_BUILD_NONE:
					cout << "none" << endl;
					break;
				case CL_BUILD_ERROR:
					cout << "error" << endl;
					break;
				case CL_BUILD_SUCCESS:
					cout << "yeah" << endl;
					break;
				default:
					cout << "ERROR in BUILD_PROG_FAILURE: Other error (please write extra cases)" << endl;
			}

			char s[10000];
			error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(s), s, NULL);
			cout << s << endl;
			cout << "fail" << endl;
			break;
		case CL_INVALID_VALUE:
			cout << "invalid" << endl;
			break;
		case CL_INVALID_DEVICE:
			cout << "inv dev" << endl;
			break;
		case CL_COMPILER_NOT_AVAILABLE:
			cout << "comp" << endl;
			break;
		case CL_INVALID_OPERATION:
			cout << "op" << endl;
			break;
		default:
			cout << "SUCCESS PROGRAM BUILD" << endl;
	}

	cl_kernel invert_kernel = clCreateKernel(program, "invert", &error);
	if (error == CL_INVALID_PROGRAM)
		cout << "InvalidProg" << endl;
	else if (error == CL_INVALID_PROGRAM_EXECUTABLE)
		cout << "exec" << endl;
	else if (error == CL_INVALID_KERNEL_NAME)
		cout << "name" << endl;
	else if (error == CL_INVALID_KERNEL_DEFINITION)
		cout << "def" << endl;
	assert(error == CL_SUCCESS);
	cl_mem data, size_buffer, data2;
	const size_t local_ws = 256;
	const size_t global_ws = shrRoundUp(local_ws, size);

	uint8_t *dataImg;
	uint8_t testInvertWorks = output_image->data[0][size-1];
	cl_event eventNDRange;
	cout << int(testInvertWorks) << endl;

	for (int c = 0; c < input_image->channels; ++c) {
		dataImg = output_image->data[c];
		data = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, output_image->data[c], &error);
		size_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(size_t), &size, &error);
		data2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, mem_size, NULL, &error);
		assert(error == CL_SUCCESS);
		//size = clCreateBuffer(context, CL_MEM_READ)

		error = clSetKernelArg(invert_kernel, 0, sizeof(cl_mem), &data);
		assert(error == CL_SUCCESS);
		error |= clSetKernelArg(invert_kernel, 1, sizeof(size_t), &size);
		assert(error == CL_SUCCESS);
		error |= clSetKernelArg(invert_kernel, 2, sizeof(cl_mem), &data2);
		assert(error == CL_SUCCESS);
		error = clEnqueueNDRangeKernel(queue, invert_kernel, 1, NULL, &global_ws, &local_ws, 0, NULL, &eventNDRange);
		assert(error == CL_SUCCESS);
		error = clEnqueueReadBuffer(queue, data2, CL_TRUE, 0, mem_size, output_image->data[c], 1, &eventNDRange, NULL);
		assert(error == CL_SUCCESS);

	}
	testInvertWorks = output_image->data[0][size-1];
	cout << int(testInvertWorks) << endl;

	if ((errortga = TGA_writeImage(argv[2], output_image)) != 0) {
		cout << "Error when writing image: " << errortga << endl;
	}

	Image_delete(input_image);
	Image_delete(output_image);
	clReleaseKernel(invert_kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}