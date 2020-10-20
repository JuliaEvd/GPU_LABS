
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <iostream>
#include <random>
#include <chrono>
#include <string>

using namespace std;

inline int idx(size_t i, size_t j, size_t size) {
  return i * size + j;
}

void matrix_print(float *&M, const size_t size) {
  for (size_t i = 0; i < 10; i++) {
    for (size_t j = 0; j < 10; j++)
      cout << M[idx(i, j, size)] << " ";
    cout << endl;
  }
  cout << endl;
}

void matrix_mult(float *&A, float *&B, float *&C, size_t size) {
  for (size_t i = 0; i < size; i++)
    for (size_t j = 0; j < size; j++) {
      C[idx(i, j, size)] = 0;
      for (size_t k = 0; k < size; k++)
        C[idx(i, j, size)] += A[idx(i, k, size)] * B[idx(k, j, size)];
    }
}

bool is_equal(float x, float y) {
  return std::fabs(x - y) < std::numeric_limits<double>::epsilon();
}

void matrix_generate(float *&M, const size_t size) {
  std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<float> distribution(-10, 10);
  for (size_t i = 0; i < size; i++)
    for (size_t j = 0; j < size; j++)
      M[idx(i, j, size)] = distribution(generator);
}

void matrix_mult_openmp(float *&A, float *&B, float *&C, size_t n) {
  {
    int i, j, k;
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        float sum = 0;
#pragma omp parallel for
        for (k = 0; k < n; k++) {
          sum += A[i*n + k] * B[k*n + j];
        }
        C[i*n + j] = sum;
      }
    }

  }
}


const char *source =
"kernel void matmulImage(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c) {\n"
"  int row = get_local_id(0);\n"
"  int col = get_local_id(1);\n"
"  const int globalRow = BS*get_group_id(0) + row;\n"
"  const int globalCol = BS*get_group_id(1) + col;\n"
"  int n = get_global_size(0);\n"
"  local float Asub[BS][BS];\n"
"  local float Bsub[BS][BS];\n"
""
"  float acc = 0.0f;\n"
"  const int numTiles = n/BS;\n"
"  for (int t = 0; t < numTiles; t++) {\n"
"    const int tiledRow = BS*t+row;\n"
"    const int tiledCol = BS*t+col;\n"
"    const int2 idA = {tiledCol, globalRow};\n"
"    const int2 idB = {globalCol, tiledRow};\n"
"    Asub[col][row] = read_imagef(a, idA).x;\n"
"    Bsub[col][row] = read_imagef(b, idB).x;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int k=0; k < BS; k++) {\n"
"      acc += Asub[k][row]*Bsub[col][k];\n"
"     }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"  }\n"
"  const int2 idC = {globalCol, globalRow};\n"
"  write_imagef(c, idC, acc);\n"
"}\n";


void matmultOMP(float *a, float *b, float *c, int SIZE) {

  int BS = 16;
  float sum;
#pragma omp parallel for
  for (int kk = 0; kk < SIZE; kk += BS) {
    //#pragma omp parallel for
    for (int jj = 0; jj < SIZE; jj += BS) {
#pragma omp parallel for
      for (int i = 0; i < SIZE; i++) {
        for (int j = jj; j < jj + BS; j++) {
          sum = c[i*SIZE + j];
          for (int k = kk; k < kk + BS; k++) {
            sum += a[i*SIZE + k] * b[SIZE*k + j];
          }
          c[i*SIZE + j] = sum;
        }
      }
    }
  }
}



int main() {
  int err;
  size_t matrix_size;

  cout << "Enter the matrix size: ";
  cin >> matrix_size;
  
  /*  const char* matrix_mult =
      "__kernel void matrix_mult(__global float* A, __global float* B, __global float* C, int n) {\n" \
      "	float sum = 0;\n" \
      "	int row = get_global_id(1);\n" \
      "	int col = get_global_id(0);\n" \
      "	for (int k = 0; k < n; k++) {\n" \
      "		sum += A[row * n + k] * B[k * n + col];\n" \
      "	}\n" \
      "	C[row * n + col] = sum;\n" \
      "}";*/

  const char* matrix_mult =
    "__kernel void matrix_mult(__global float* a, __global float* b, __global float* c, int n) {\n" \
    "  int BS = 16;\n" \
    "  int row = get_local_id(0);\n" \
    "  int col = get_local_id(1);\n" \
    "  const int globalRow = BS*get_group_id(0) + row;\n" \
    "  const int globalCol = BS*get_group_id(1) + col;\n" \
    "  local float Asub[16][16];\n" \
    "  local float Bsub[16][16];\n"\
    ""\
    "  float acc = 0.0f;\n"\
    "  const int numTiles = n/16;\n"\
    "  for (int t = 0; t < numTiles; t++) {\n"\
    "    const int tiledRow = 16*t+row;\n"\
    "    const int tiledCol = 16*t+col;\n"\
    "    Asub[col][row] = a[tiledCol*n + globalRow];\n"\
    "    Bsub[col][row] = b[globalCol*n + tiledRow];\n"\
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"\
    "    for (int k=0; k < 16; k++) {\n"\
    "      acc = mad(Asub[k][row], Bsub[col][k], acc);\n"\
    "     }\n"\
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"\
    "  }\n"\
    "  c[globalCol*n+globalRow] = acc;\n"\
    "}";
   

  cl_uint num_platforms = 0;
  clGetPlatformIDs(0, NULL, &num_platforms);
  cl_platform_id platform = NULL;
  if (num_platforms > 0) {
    cl_platform_id* platforms = new cl_platform_id[num_platforms];
    clGetPlatformIDs(num_platforms, platforms, NULL);
    platform = platforms[0];
    delete[] platforms;
  }

  struct
  {
    cl_device_type type;
    const char* name;
    cl_uint count;
    cl_device_id device_id;
    double time;
    float* C;
  }
  devices[] =
  {
    { CL_DEVICE_TYPE_CPU, "GPU", 0, NULL, 0, NULL },
    { CL_DEVICE_TYPE_GPU, "CPU", 0, NULL, 0, NULL },
  };

  const int NUM_OF_DEVICE_TYPES = sizeof(devices) / sizeof(devices[0]);

  for (int i = 0; i < NUM_OF_DEVICE_TYPES; ++i)
  {
    devices[i].C = (float*)malloc(matrix_size * matrix_size * sizeof(float));

    for (int k = 0; k < matrix_size * matrix_size; k++) {
      devices[i].C[k] = 0;
    }

    err = clGetDeviceIDs(
      platform,
      devices[i].type,
      1,
      &devices[i].device_id,
      &devices[i].count
    );

    if (CL_DEVICE_NOT_FOUND == err) {
      devices[i].count = 0;
      err = CL_SUCCESS;
    }
  }

  float* data_A = (float*)malloc(matrix_size * matrix_size * sizeof(float));
  float* data_B = (float*)malloc(matrix_size * matrix_size * sizeof(float));
  float* data_C = (float*)malloc(matrix_size * matrix_size * sizeof(float));

  for (int i = 0; i < matrix_size*matrix_size; i++) {
    data_C[i] = 0;
    data_A[i] = 2;
    data_B[i] = 3;
  }

  //matrix_generate(data_A, matrix_size);
  //matrix_generate(data_B, matrix_size);
  /*
  cout << "matrix A:" << endl;
  matrix_print(data_A, matrix_size);
  cout << "matrix B:" << endl;
  matrix_print(data_B, matrix_size);*/
  float* omp_C = (float*)malloc(matrix_size * matrix_size * sizeof(float));

  clock_t start = clock();
  cl_ulong start1 = 0, end1 = 0;


  matmultOMP(data_A, data_B, omp_C, 1024);

  clock_t finish = clock();

  double openmp_time = (double)(finish - start) / CLOCKS_PER_SEC;


  for (int i = 0; i < NUM_OF_DEVICE_TYPES; ++i) {

    cl_device_id device_id = devices[i].device_id;

    cl_context context = clCreateContext(0, devices[i].count, &device_id, NULL, NULL, &err);

    cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);



    cl_program program = clCreateProgramWithSource(context, 1, &matrix_mult, NULL, &err);

    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
      size_t len;
      char buffer[2048];
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
      printf("%s\n", buffer);
      system("pause");
      //exit(1);
    }

    cl_kernel kernel = clCreateKernel(program, "matrix_mult", &err);
    if (err != CL_SUCCESS) {
      printf("Error create kernel\n");
      system("pause");
      //exit(1);
    }
    cl_mem A = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * matrix_size * matrix_size, NULL, &err);
    if (err != CL_SUCCESS) {
      printf("Error create buffer\n");
      system("pause");
    }
    cl_mem B = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * matrix_size * matrix_size, NULL, &err);
    if (err != CL_SUCCESS) {
      printf("Error create buffer\n");
      system("pause");
    }
    cl_mem C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * matrix_size * matrix_size, NULL, &err);
    if (err != CL_SUCCESS) {
      printf("Error create buffer\n");
      system("pause");
    }


    clEnqueueWriteBuffer(queue, A, CL_TRUE, 0, sizeof(float) * matrix_size * matrix_size, data_A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, B, CL_TRUE, 0, sizeof(float) * matrix_size * matrix_size, data_B, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, C, CL_TRUE, 0, sizeof(float) * matrix_size * matrix_size, devices[i].C, 0, NULL, NULL);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &A);
    if (err != CL_SUCCESS) {
      printf("Error set kernel arg 1\n");
      system("pause");
    }
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &B);
    if (err != CL_SUCCESS) {
      printf("Error set kernel arg 2\n");
      system("pause");
    }
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &C);
    if (err != CL_SUCCESS) {
      printf("Error set kernel arg 3\n");
      system("pause");
    }
    err = clSetKernelArg(kernel, 3, sizeof(int), &matrix_size);
    if (err != CL_SUCCESS) {
      printf("Error set kernel arg 4\n");
      system("pause");
    }

    size_t group_size[2] = { 16, 16 };
    size_t global_work_size[2] = { matrix_size, matrix_size };

    //clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group_size, NULL);


    clock_t start = clock();

    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, group_size, 0, NULL, NULL);

    clFinish(queue);

    clock_t finish = clock();

    devices[i].time = (double)(finish - start) / CLOCKS_PER_SEC;

    clEnqueueReadBuffer(queue, C, CL_TRUE, 0, sizeof(float) * matrix_size * matrix_size, devices[i].C, 0, NULL, NULL);

    //	std::cout << devices[i].name << ":" << std::endl;

    //	matrix_print(devices[i].C, matrix_size);
    //	std::cout << std::endl;

    clReleaseMemObject(A);
    clReleaseMemObject(B);
    clReleaseMemObject(C);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
  }



  int tmp = 0;
  for (int i = 0; i < matrix_size * matrix_size; i++) {
    if (devices[0].C[i] != omp_C[i] || devices[1].C[i] != omp_C[i]) tmp = 1;

    if (is_equal(devices[0].C[i], omp_C[i]) || is_equal(devices[1].C[i], omp_C[i])) tmp = 1;
  }



  std::cout << "Time on CPU_OpenCL: " << devices[0].time << std::endl;
  std::cout << "Time on GPU_OpenCl: " << devices[1].time << std::endl;
  std::cout << "Time on OpenMP: " << openmp_time << std::endl;



  cl_uint num_platforms1 = 0;
  clGetPlatformIDs(0, NULL, &num_platforms1);

  cl_platform_id platform1 = NULL;
  if (0 < num_platforms1) {
    cl_platform_id *platforms1 = new cl_platform_id[num_platforms1];
    clGetPlatformIDs(num_platforms1, platforms1, NULL);
    platform1 = platforms1[0];

    char platform_name1[128];
    clGetPlatformInfo(platform1, CL_PLATFORM_NAME, 128, platform_name1, nullptr);
    //	std::cout << platform_name1 << std::endl;

    delete[] platforms1;
  }

  cl_context_properties properties1[3] = { CL_CONTEXT_PLATFORM,
                       (cl_context_properties)platform1, 0 };
  cl_int error1 = 0;

  cl_context context1 =
    clCreateContextFromType((NULL == platform1) ? NULL : properties1,
      CL_DEVICE_TYPE_GPU, NULL, NULL, &error1);
  //	std::cout << "Create device for images" << std::endl;
  if (error1 != CL_SUCCESS) {
    std::cout << "Create context from type failed" << std::endl;
  }

  size_t size1 = 0;

  clGetContextInfo(context1, CL_CONTEXT_DEVICES, 0, NULL, &size1);

  cl_device_id device1 = 0;
  if (size1 > 0) {
    cl_device_id *devices1 = (cl_device_id *)alloca(size1);
    clGetContextInfo(context1, CL_CONTEXT_DEVICES, size1, devices1, NULL);
    device1 = devices1[0];

    char device_name1[128];
    clGetDeviceInfo(device1, CL_DEVICE_NAME, 128, device_name1, nullptr);
    //std::cout << device_name1 << std::endl;

    char device_size1[128];
    clGetDeviceInfo(device1, CL_DEVICE_IMAGE2D_MAX_HEIGHT, 128, device_size1, nullptr);
    //std::cout << device_size1 << std::endl;
  }
  cl_command_queue queue1 =
    clCreateCommandQueue(context1, device1, CL_QUEUE_PROFILING_ENABLE, &error1);
  if (error1 != CL_SUCCESS) {
    std::cout << "Create command queue with properties failed" << std::endl;
  }

  size_t srclen1[] = { strlen(source) };

  cl_program program1 =
    clCreateProgramWithSource(context1, 1, &source, srclen1, &error1);
  if (error1 != CL_SUCCESS) {
    std::cout << "Create program with source failed" << std::endl;
  }
  std::string buildOpts1 = "-DBS=" + std::to_string(16);
  error1 = clBuildProgram(program1, 1, &device1, buildOpts1.c_str(), nullptr, nullptr);

  if (error1 != CL_SUCCESS) {
    std::cout << "Build prog failed" << std::endl;
    size_t logSize = 0;
    clGetProgramBuildInfo(program1, device1, CL_PROGRAM_BUILD_LOG, 0,
      nullptr, &logSize);
    char *log = new char[logSize];
    clGetProgramBuildInfo(program1, device1, CL_PROGRAM_BUILD_LOG, logSize,
      log, nullptr);
    std::cout << log;
  }
  cl_kernel kernel = clCreateKernel(program1, "matmulImage", &error1);
  cl_image_format format;
  format.image_channel_data_type = CL_FLOAT;
  format.image_channel_order = CL_R;
  if (error1 != CL_SUCCESS) {
    std::cout << "Create kernel failed" << std::endl;
    std::cout << error1 << std::endl;
  }


  _cl_image_desc desc = {};
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_width = matrix_size;
  desc.image_height = matrix_size;
  desc.image_depth = 1;
  cl_mem aBuf = clCreateImage(context1, CL_MEM_READ_WRITE, &format, &desc, NULL, &error1);
  if (error1 != CL_SUCCESS) {
    std::cout << "Err creating image: " << error1 << std::endl;
  }

  cl_mem bBuf = clCreateImage(context1, CL_MEM_READ_WRITE, &format, &desc, NULL, &error1);
  if (error1 != CL_SUCCESS) {
    std::cout << "Err creating image: " << error1 << std::endl;
  }

  cl_mem cBuf = clCreateImage(context1, CL_MEM_WRITE_ONLY, &format, &desc, NULL, &error1);
  if (error1 != CL_SUCCESS) {
    std::cout << "Err creating image: " << error1 << std::endl;
  }
  size_t Origin[] = { 0,0,0 };
  size_t Region[] = { matrix_size, matrix_size, 1 };

  error1 = clEnqueueWriteImage(queue1, aBuf, CL_TRUE, Origin, Region, 0, 0, data_A, 0, NULL, NULL);
  if (error1 != CL_SUCCESS) {
    std::cout << "Err writing image A: " << error1 << std::endl;
  }

  error1 = clEnqueueWriteImage(queue1, bBuf, CL_TRUE, Origin, Region, 0, 0, data_B, 0, NULL, NULL);
  if (error1 != CL_SUCCESS) {
    std::cout << "Err writing image B: " << error1 << std::endl;
  }

  error1 = clEnqueueWriteImage(queue1, cBuf, CL_TRUE, Origin, Region, 0, 0, data_C, 0, NULL, NULL);
  if (error1 != CL_SUCCESS) {
    std::cout << "Err writing image C: " << error1 << std::endl;
  }



  error1 = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuf);
  if (error1 != CL_SUCCESS) {
    std::cout << "Failed to set kernel args 0: " << error1 << std::endl;
  }
  error1 = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bBuf);
  if (error1 != CL_SUCCESS) {
    std::cout << "Failed to set kernel args 1: " << error1 << std::endl;
  }
  error1 = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cBuf);
  if (error1 != CL_SUCCESS) {
    std::cout << "Failed to set kernel args 2: " << error1 << std::endl;
  }


  const size_t offsets[] = { 0, 0 };
  const size_t sizes[] = { matrix_size, matrix_size };
  const size_t local_size[] = { 16, 16 };
  cl_event evt;
  auto startSys = std::chrono::steady_clock::now();


  error1 = clEnqueueNDRangeKernel(queue1, kernel, 2, offsets, sizes, local_size, 0,
    0, &evt);


  if (error1 != CL_SUCCESS) {
    std::cout << "Enqueue failed: " << error1 << std::endl;
  }

  clWaitForEvents(1, &evt);
  auto endSys = std::chrono::steady_clock::now();


  error1 = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start1, nullptr);
  if (error1 != CL_SUCCESS) {
    std::cout << "Error getting start time: " << error1 << std::endl;
  }
  error1 = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end1, nullptr);
  if (error1 != CL_SUCCESS) {
    std::cout << "Error getting end time: " << error1 << std::endl;
  }
  clEnqueueReadImage(queue1, cBuf, CL_TRUE, Origin, Region, 0, 0, data_C, 0, NULL, NULL);
  std::cout << "Time on IMAGES: " << std::chrono::duration_cast<std::chrono::nanoseconds>(endSys - startSys).count()*(1e-09) << std::endl;
  //matrix_print(data_C, matrix_size);

  if (tmp != 1) {
    std::cout << "error occurred! check mistakes." << std::endl;
  }
  else {
    std::cout << "cpu/gpu/openMP/Images answers are correct." << std::endl;
  }

  system("pause");
  return 0;
}