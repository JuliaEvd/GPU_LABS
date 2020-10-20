#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define SIZE 256000
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <CL/cl.h>

const double threshold = 5e-10;

bool checkResults(float* first, float* second, int n) {
	bool check = false;
	for (int i = 0; i < n; i++) {
		if (fabs(first[i] - second[i]) <= threshold) {
			check = true;
		}
		else {
			check = false;
		}
	}
	return check;
}

const char* kernelJacobi = "__kernel								\n"
"void jacobi(														\n"
"	__global float * A,												\n"
"	__global float * b,												\n"
"	__global float * x_old,											\n"
"	__global float * x_new											\n"
") {																\n"
"	const size_t i = get_global_id(0);								\n"
"	const size_t size = get_global_size(0);							\n"
"	float acc = 0.0f;												\n"
"	for (size_t j = 0; j < size; j++) {								\n"
"		acc += A[j * size + i] * x_old[j] * (float)(i != j);		\n"
"	}																\n"
"	x_new[i] = (b[i] - acc) / A[i * size + i];						\n"
"}																	\n";

std::string load_file(const std::string & path) {
	return std::string(
		std::istreambuf_iterator<char>(
			*std::make_unique<std::ifstream>(path)
			),
		std::istreambuf_iterator<char>()
	);
}

float * gen_symmetric_positive_mat(size_t size) {
	static std::random_device rd;
	static std::default_random_engine re(rd());
	static std::uniform_real_distribution<float> dist{ -10, 10 };

	auto * mem = new float[size * size];
	for (size_t i = 0; i < size; ++i) {
		float sum = 0.0f;
		for (size_t j = 0; j < size; ++j) {
			float tmp = dist(rd);
			mem[i * size + j] = tmp;
			sum += abs(tmp);
		}
		mem[i * size + i] = sum + abs(dist(rd)) + 1.0f;
	}

	return mem;
}

float * gen_vec(size_t size) {
	static std::random_device rd;
	static std::default_random_engine re(rd());
	static std::uniform_real_distribution<float> dist{ -10, 10 };

	auto * mem = new float[size];
	for (size_t i = 0; i < size; ++i) {
		mem[i] = dist(rd);
	}

	return mem;
}

void print_mat(float * mat, size_t rows, size_t cols) {
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			std::cout << mat[i * cols + j] << " ";
		}
		std::cout << std::endl;
	}
}

int main() {
	cl_int error = 0;

	// Îïðåäåëíèå ðàçìåðà
	size_t size;
	std::cout << "Enter matrix size: ";
	std::cin >> size;

	// Çàïîëíåíèå ïàìÿòè
	float * A = gen_symmetric_positive_mat(size);
	float * b = gen_vec(size);

#ifndef NDEBUG
	printf("Matrix A[%zu, %zu] generated\n", size, size);
	//print_mat(A, size, size);

	printf("Vector b[%zu] generated\n", size);
	//print_mat(b, size, 1);
#endif

	std::cout << "Enter count of Max Iters: ";
	size_t MAX_ITERS;
	std::cin >> MAX_ITERS;

	std::cout << "Enter EPS: ";
	float EPS;
	std::cin >> EPS;

	float * tmp = new float[size];
	for (size_t i = 0; i < size; ++i) {
		tmp[i] = rand();
	}

	for (size_t k = 0; k < 2; ++k) {
		float * x_old = new float[size];
		for (size_t i = 0; i < size; ++i) {
			x_old[i] = tmp[i];
		}
		float * x_new = new float[size]();

		// Состояние для проверки ошибки
		cl_int error;

		// Определение платформы
		cl_uint num_platforms = 0;
		clGetPlatformIDs(0, NULL, &num_platforms);
		cl_platform_id current_platform = NULL;
		if (num_platforms > 0) {
			std::vector<cl_platform_id> platforms(num_platforms);
			clGetPlatformIDs(num_platforms, platforms.data(), NULL);
			current_platform = platforms[1];

			// Вывод информации о платформе
			char platform_name[128];
			clGetPlatformInfo(current_platform, CL_PLATFORM_NAME, 128, platform_name, nullptr);
			std::cout << "Current platform is: " << platform_name << std::endl;
		}

		// Îïðåäåëåíèå óñòðîéñòâà
		cl_uint num_devices = 0;
		clGetDeviceIDs(current_platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
		std::vector<cl_device_id> devices(num_devices);
		clGetDeviceIDs(current_platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);

		// Определение устройства
		for (size_t i = 0; i < num_devices; i++) {
			char device_name[128];
			clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, device_name, nullptr);
			std::cout << i + 1 << ") " << device_name << std::endl;
		}

		int id_device;
		std::cout << "Choose device: ";
		std::cin >> id_device;
		cl_device_id current_device = devices[id_device - 1];

		cl_context context = clCreateContext(nullptr, 1, &current_device, nullptr, nullptr, nullptr);

		size_t srclen[] = { strlen(kernelJacobi) };

		cl_program program = clCreateProgramWithSource(context, 1, &kernelJacobi, srclen, &error);
		if (error != CL_SUCCESS) { std::cout << "Create program failed: " << error << std::endl; }

		error = clBuildProgram(program, 1, &current_device, nullptr, nullptr, nullptr);
		if (error == CL_BUILD_PROGRAM_FAILURE) {
			size_t log_size;
			clGetProgramBuildInfo(program, current_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
			char *log = new char[log_size];
			clGetProgramBuildInfo(program, current_device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
			std::cout << log << std::endl;
		}

		cl_kernel kernel_jacobi = clCreateKernel(program, "jacobi", &error);
		if (error != CL_SUCCESS) { std::cout << "Create kernel failed: " << error << std::endl; }

		cl_mem A_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
			size * size * sizeof(float), A, &error);
		if (error != CL_SUCCESS) { std::cout << "Create A_buf failed: " << error << std::endl; }
		cl_mem b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
			size * sizeof(float), b, &error);
		if (error != CL_SUCCESS) { std::cout << "Create b_buf failed: " << error << std::endl; }
		cl_mem x_old_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
			size * sizeof(float), x_old, &error);
		if (error != CL_SUCCESS) { std::cout << "Create x_old_buf failed: " << error << std::endl; }
		cl_mem x_new_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
			size * sizeof(float), x_new, &error);
		if (error != CL_SUCCESS) { std::cout << "Create x_new_buf failed: " << error << std::endl; }

		error = clSetKernelArg(kernel_jacobi, 0, sizeof(cl_mem), &A_buf);
		if (error != CL_SUCCESS) { std::cout << "Set kernel args for A_buf failed: " << error << std::endl; }
		error |= clSetKernelArg(kernel_jacobi, 1, sizeof(cl_mem), &b_buf);
		if (error != CL_SUCCESS) { std::cout << "Set kernel args for b_buf failed: " << error << std::endl; }
		error |= clSetKernelArg(kernel_jacobi, 2, sizeof(cl_mem), &x_old_buf);
		if (error != CL_SUCCESS) { std::cout << "Set kernel args for x_old_buf failed: " << error << std::endl; }
		error |= clSetKernelArg(kernel_jacobi, 3, sizeof(cl_mem), &x_new_buf);
		if (error != CL_SUCCESS) { std::cout << "Set kernel args for x_new_buf failed: " << error << std::endl; }

		const size_t global_work_offset[] = { 0 };
		const size_t global_work_size[] = { size };
		const size_t local_work_size[] = { 16 };

		cl_command_queue queue = clCreateCommandQueue(context, current_device, CL_QUEUE_PROFILING_ENABLE, &error);
		if (error != CL_SUCCESS) { std::cout << "Create command queue failed: " << error << std::endl; }

		cl_event evt;
		cl_ulong time = 0;

		float conv = CL_INFINITY;
		size_t iters = 0;

		auto start_chrn_time = std::chrono::steady_clock::now();
		do {
			error = clEnqueueNDRangeKernel(queue, kernel_jacobi, 1,
				global_work_offset, global_work_size, local_work_size, 0, nullptr, &evt);
			error = clWaitForEvents(1, &evt);
			error = clEnqueueReadBuffer(queue, x_new_buf, CL_TRUE, 0, size * sizeof(float), x_new, 0, nullptr,
				nullptr);

			cl_ulong start_time = 0, end_time = 0;
			error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, nullptr);
			error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, nullptr);
			time += end_time - start_time;

			conv = 0.0f;
			for (size_t i = 0; i < size; ++i) {
				conv += (x_new[i] - x_old[i]) * (x_new[i] - x_old[i]);
			}
			conv = std::sqrt(conv);
			iters++;

			//printf("%g, %g\n", x_old[0], x_new[0]);
			std::swap(x_old, x_new);
			//printf("%g, %g\n", x_old[0], x_new[0]);
			std::swap(x_old_buf, x_new_buf);

			error = clSetKernelArg(kernel_jacobi, 2, sizeof(cl_mem), &x_old_buf);
			error |= clSetKernelArg(kernel_jacobi, 3, sizeof(cl_mem), &x_new_buf);

#ifndef NDEBUG
			printf("Current iter is - %zu, convergence = %g\n", iters, conv);
#endif
		} while (conv > EPS && iters < MAX_ITERS);
		auto end_chrn_time = std::chrono::steady_clock::now();

		error = clFinish(queue);
		error = clEnqueueReadBuffer(queue, x_old_buf, CL_TRUE, 0, size * sizeof(float), x_new, 0, nullptr, nullptr);

		if (conv <= EPS) {
			printf("OK! convergence achieved\n");
		}
		else if (iters >= MAX_ITERS) {
			printf("WARNING: convergence not achieved, number of iterations exceeded\n");
		}
		printf("Convergence = %g with %zu iterations and %f seconds\n",
			conv, iters, time * 1e-09);

		std::cout << "Chrono time is: " << std::chrono::duration_cast<std::chrono::nanoseconds>
			(end_chrn_time - start_chrn_time).count() * 1e-09 << " seconds" << std::endl;

		float err = 0.0f;
		for (size_t i = 0; i < size; ++i) {
			float acc = 0.0f;
			for (size_t j = 0; j < size; ++j) {
				acc += A[j * size + i] * x_new[j];
			}
			err += (acc - b[i]) * (acc - b[i]);
		}
		err = std::sqrt(err);
		printf("Jacobi solver: err = %g\n", err);
		if (err <= EPS) {
			printf("OK: final solution error <= %g\n", EPS);
		}
		else {
			printf("INFO: final solution error > %g\n", EPS);
		}

		clReleaseKernel(kernel_jacobi);
		clReleaseProgram(program);
		clReleaseContext(context);
	}
	system("pause");
	return 0;
}
