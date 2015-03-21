/**
 * @file main.h
 * @author Steven Le Rouzic <lerouzic@ecole.ensicaen.fr>
 * @author Gautier Boëda <boeda@ecole.ensicaen.fr>
 */

#ifndef _OPENCL_LAUNCHER_H
#define _OPENCL_LAUNCHER_H

#include <CL/opencl.h>
#include <iostream>
#include <string>

class Opencl_launcher{
public:
	/**
	 * Create an OpenCL launcher
	 * @param name Type of launcher (Intel or NVidia)
	 */
	Opencl_launcher(std::string name);
	~Opencl_launcher();

	/**
	 * Load the kernel
	 * @param file Path of the kernel file
	 * @param name Kernel name in the kernel file
	 * @return Kernel
	 */
	cl_kernel load_kernel(std::string file, std::string name);

	/**
	 * Gets the context
	 */
	cl_context get_context();

	/**
	 * Gets the command queue
	 */
	cl_command_queue get_queue();

	/**
	 * Benchmark of the selected event
	 * @param event Event which we want a benchmark of.
	 * @param name Type of the benchmark (Transfert/Execution)
	 */
	void benchmark(cl_event event, std::string name);

	/**
	 * Displays the total_time (Execution time + Transfert time)
	 */
	void total_time();
private:
	cl_int error = 0;
	cl_platform_id platform;
	cl_uint nb_platforms;
	cl_context context;
	cl_command_queue queue;
	cl_device_id device;
	cl_kernel kernel;
	double time_transfer_exec = 0;
};

#endif