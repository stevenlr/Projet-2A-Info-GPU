/**
 * @file opencl_launcher.cpp
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */
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

Opencl_launcher::Opencl_launcher(string name){
	// Platform
	cl_platform_id* platforms;
	cl_device_type device_type;
	clGetPlatformIDs(0, NULL, &nb_platforms);
	platforms = new cl_platform_id[nb_platforms];
	clGetPlatformIDs(nb_platforms, platforms, NULL);

	char vendor[100];
	for (int i = 0; i < nb_platforms; ++i) {
		clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);

		if (strncmp(vendor, name.c_str(), 5) == 0) {
			clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(vendor), vendor, NULL);
			
			if (strncmp(name.c_str(), "Intel", 5) == 0){
				if (strncmp(vendor, "OpenCL 1", 8) == 0){
					platform = platforms[i];
					break;
				}
			} else
				platform = platforms[i];
		}
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

cl_context Opencl_launcher::get_context(){
	return context;
}

cl_command_queue Opencl_launcher::get_queue(){
	return queue;
}

void Opencl_launcher::benchmark(cl_event event, string name){
	cl_ulong time_start, time_end;
	double total_time;

	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	
	total_time = time_end - time_start;

	if (strncmp(name.c_str(), "Exec", 4) == 0)
		time_transfer_exec += total_time;
	else
		time_transfer_exec += total_time * 2;
	
	if (strncmp(name.c_str(), "ExecD", 5) != 0 && !multiple_exec)
		cout << name << " in seconds = " << (total_time / 1000000.0) << "ms" << endl;
	else if (multiple_exec){
		cout << name << " in seconds = " << (time_transfer_exec / 1000000.0) << "ms" << endl;
		multiple_exec = false;
	}
	else
		multiple_exec = true;
}

void Opencl_launcher::total_time(){
	cout << "Total time in seconds = " << (time_transfer_exec / 1000000.0) << "ms" << endl;
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
			cout << "";//cout << "SUCCESS PROGRAM BUILD" << endl;
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
	else if (error != CL_SUCCESS)
		cout << error << endl;

	assert(error == CL_SUCCESS);

	return kernel;
}