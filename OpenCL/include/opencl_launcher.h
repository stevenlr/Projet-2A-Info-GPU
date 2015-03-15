#include <CL/opencl.h>
#include <iostream>
#include <string>

class Opencl_launcher{
public:
	Opencl_launcher(std::string name);
	~Opencl_launcher();
	cl_kernel load_kernel(std::string file, std::string name);
	cl_context get_context();
	cl_command_queue get_queue();
	void benchmark(cl_event event, std::string name);
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