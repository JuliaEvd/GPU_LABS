#include <CL/cl.h>
#include <iostream>
#define SIZE 512
int main() {

	cl_int error = 0;

	const char* source = "__kernel void mykernel() {\n"
		"int global_id= get_global_id(0);\n"
		"int local_id=get_local_id(0);\n"
		"int group_id=get_group_id(0);\n"
		"printf(\"HELLO FROM block=%d , thred=%d, global_thread=%d \\n \", group_id, local_id, global_id);\n"
		"}\n";

	cl_uint numPlatforms = 0;
	clGetPlatformIDs(0, NULL, &numPlatforms);
	cl_platform_id platform = NULL;

	if (0 < numPlatforms) {
		cl_platform_id * platforms = new cl_platform_id[numPlatforms];
		error=clGetPlatformIDs(numPlatforms, platforms, NULL);
        if (error != CL_SUCCESS) {
            std::cout << "Error clCreateContextFromType" << std::endl;
        }
        if (error == CL_SUCCESS) {
            std::cout << "Success clCreateContextFromType" << std::endl;
        }


		platform = platforms[0];
		delete[] platforms;
	}
	cl_context_properties properties[3] = {
		CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0
	};

	cl_context context = clCreateContextFromType((NULL == platform) ? NULL : properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &error);

	if (error != CL_SUCCESS) {
		std::cout << "Error clCreateContextFromType" << std::endl;
	}
    if (error == CL_SUCCESS) {
        std::cout << "Success clCreateContextFromType" << std::endl;
    }

	size_t size = 0;

	error = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);

	if (error != CL_SUCCESS) {
		std::cout << "Error clGetContextInfo" << std::endl;
	}
    if (error == CL_SUCCESS) {
        std::cout << "Success clGetContextInfo" << std::endl;
    }

	cl_device_id device = 0;

	if (size > 0) {
		cl_device_id * devices = (cl_device_id *)alloca(size);
		clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices, NULL);
		device = devices[0];
	}

	cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &error);

	if (error != CL_SUCCESS) {
		std::cout << "Error clCreateCommandQueueWithProperties" << std::endl;
	}
    if (error == CL_SUCCESS) {
        std::cout << "Success clCreateCommandQueueWithProperties" << std::endl;
    }

	size_t srclen[] = { strlen(source) };

	cl_program program = clCreateProgramWithSource(context, 1, &source, srclen, &error);
    if (error != CL_SUCCESS) {
        std::cout << "Error clCreateProgramWithSource" << std::endl;
    }
    if (error == CL_SUCCESS) {
        std::cout << "Success clCreateProgramWithSource" << std::endl;
    }
	error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (error != CL_SUCCESS) {
        std::cout << "Error clBuildProgram" << std::endl;
    }
    if (error == CL_SUCCESS) {
        std::cout << "Success clBuildProgram" << std::endl;
    }

	cl_kernel kernel = clCreateKernel(program, "mykernel", &error);
    if (error != CL_SUCCESS) {
        std::cout << "Error clCreateKernel" << std::endl;
    }
    if (error == CL_SUCCESS) {
        std::cout << "Success clCreateKernel" << std::endl;
    }

	float data[SIZE];
	float results[SIZE];

	for (int i = 0; i < SIZE; i++)
		data[i] = (float)rand();

	cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * SIZE, NULL, NULL);

	cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * SIZE, NULL, NULL);
	clEnqueueWriteBuffer(queue, input, CL_TRUE, 0, sizeof(float) * SIZE, data, 0, NULL, NULL);

	size_t count = SIZE;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
	clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);

	size_t group;

	clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, NULL);
	clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &count, &group, 0, NULL, NULL);

	clFinish(queue);

	clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL);

	clReleaseMemObject(input);
	clReleaseMemObject(output);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	//cl_uint platformCount = 0;
	//clGetPlatformIDs(0, nullptr, &platformCount);

	//cl_platform_id* platform = new cl_platform_id[platformCount];
	//clGetPlatformIDs(platformCount, platform, nullptr);

	//for (cl_uint i = 0; i < platformCount; ++i) {
	//	char platformName[128];
	//	clGetPlatformInfo(platform[i], CL_PLATFORM_NAME,
	//		128, platformName, nullptr);
	//	std::cout << platformName << std::endl;
	//}

	return 0;
}
