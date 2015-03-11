#include <cstdlib>
#include <cassert>
#include <iostream>
#include <string>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <cmath>

#include "opencl_launcher.h"
#include "main.h"

using namespace std;

Opencl_launcher::Opencl_launcher(){
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
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
	if (error != CL_SUCCESS) {
		cout << "Error creating command queue: " << errorMessage(error) << endl;
		exit(error);
	}
}

Opencl_launcher::~Opencl_launcher(){
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}

cl_context Opencl_launcher::getContext(){
	return context;
}

cl_command_queue Opencl_launcher::getQueue(){
	return queue;
}

void Opencl_launcher::benchmark(cl_event event){
	cl_ulong time_start, time_end;
	double total_time;

	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	
	total_time = time_end - time_start;
	cout << "Execution time in seconds = " << (total_time / 1000000000.0) << "s" << endl;
}

cl_kernel Opencl_launcher::load_kernel(string file, string name){
	const char* path = file.c_str();
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

	kernel = clCreateKernel(program, name.c_str(), &error);
	if (error == CL_INVALID_PROGRAM)
		cout << "InvalidProg" << endl;
	else if (error == CL_INVALID_PROGRAM_EXECUTABLE)
		cout << "exec" << endl;
	else if (error == CL_INVALID_KERNEL_NAME)
		cout << "name" << endl;
	else if (error == CL_INVALID_KERNEL_DEFINITION)
		cout << "def" << endl;
	assert(error == CL_SUCCESS);

	return kernel;
}