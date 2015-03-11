#include <CL/opencl.h>
#include <string>

class Opencl_launcher(){
public:
	Opencl_launcher();
	Boolean Opencl_load_kernel(String file);
	cl_context getContext();
	cl_queue getQueue();
private:
	cl_int error = 0;
	cl_platform_id platform;
	cl_uint nbPlatforms;
	cl_context context;
	cl_command_queue queue;
	cl_device_id device;
	cl_kernel kernel;
}