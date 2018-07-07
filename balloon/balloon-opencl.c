#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
//#include <unistd.h>
#include "balloon.h"
//#include "../sha256-sse/sha256.h"
//#include <sys/time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
/* see check_opencl.h for docs on the CHECK_* macros */
#include "check_opencl.h"

#define val_size 10000
char val[val_size];
#define MAX_SOURCE_SIZE (0x10000000)
char *sourcepath= "balloon/balloon-opencl.cl";

#define __global__
#define __device__
#define __constant__

__global__ void conv_onethread(int n,int fn, const float * signal, const float * filter, float * retSignal);
__device__ void cuda_hash_state_mix (struct hash_state *s, int32_t mixrounds, uint64_t *prebuf_le);
__device__ void cuda_hash_state_extract (const struct hash_state *s, uint8_t out[BLOCK_SIZE]);
__device__ void cuda_compress (uint64_t *counter, uint8_t *out, const uint8_t *blocks[], size_t blocks_to_comp);
__device__ void cuda_expand (uint64_t *counter, uint8_t *buf, size_t blocks_in_buf);
__device__ void cuda_hash_state_fill (struct hash_state *s, const uint8_t *in, size_t inlen, int32_t t_cost, int64_t s_cost);
void update_device_data(int gpuid);

#define DEBUG
//#define CUDA_DEBUG
//#define CUDA_OUTPUT
//#define LOWMEM

#define PREBUF_LEN 409600
uint64_t host_prebuf_le[20][PREBUF_LEN / 8];
uint8_t host_prebuf_filled[20] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
#define BLOCK_SIZE (32)

cl_mem device_prebuf_le[20];
cl_mem device_winning_nonce[20];
cl_mem device_sbuf[20];
struct hash_state *device_s[20];
cl_mem device_target[20];
cl_mem device_is_winning[20];
cl_mem device_out[20];
cl_mem device_input[20];
cl_mem device_hs_sbufs[20];
cl_mem device_sbufs[20];

uint8_t balloon_inited[20] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
uint8_t syncmode_set[20] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };

int opencl_query() {
	DECLARE_CHECK;
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;

    CHECK_clGetPlatformIDs(1, &platform_id, &ret_num_platforms,
			   err_clGetPlatformIDs);

    fprintf(stderr, "[opencl_query] ret_num_platforms=%i\n", ret_num_platforms);

    CHECK_clGetDeviceIDs( platform_id,
			  CL_DEVICE_TYPE_GPU,
			  1,
			  &device_id,
			  &ret_num_devices,
			  err_clGetDeviceIDs);

	return ret_num_devices;
 err_clGetDeviceIDs:
    // XXX deallocate platform_id ?
 err_clGetPlatformIDs:
    return CHECK_errors;
}

uint8_t opencl_inited = 0;
cl_command_queue command_queue;
cl_kernel kernel;
cl_context context;
int *A, *B, *OUT; // TODO: REMOVE
FILE *fp;
cl_mem a_mem_obj, b_mem_obj, c_mem_obj; // TODO: REMOVE
const int LIST_SIZE = 1024; // TODO: REMOVE
int balloon_opencl_init(int gpuid, uint32_t num_threads, uint32_t num_blocks) {
    DECLARE_CHECK;
	printf("gpuid: %d, num_threads: %d, num_blocks: %d\n", gpuid, num_threads, num_blocks);

	if (!opencl_inited) {
		// Create the two input vectors
		A = malloc(sizeof(int)*LIST_SIZE);
		B = malloc(sizeof(int)*LIST_SIZE);

		// Read the memory buffer C on the device to the local variable C
		OUT = malloc(sizeof(int)*LIST_SIZE);

		for(int i = 0; i < LIST_SIZE; i++) {
		A[i] = i;
		B[i] = LIST_SIZE - i;
		}
		printf("gpuid: %d, num_threads: %d, num_blocks: %d\n", gpuid, num_threads, num_blocks);

		// Load the kernel source code into the array source_str
		fp = fopen(sourcepath, "r");
		if (!fp) {
		fprintf(stderr, "Failed to open kernel file '%s': %s\n", sourcepath,
			strerror(errno));
		inc_CHECK_errors();
		goto err_fopen;
		}

		char *source_str = CHECK_malloc(MAX_SOURCE_SIZE, err_malloc_source_str);
		size_t source_size= fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    	fclose(fp);

		/*Step1: Getting platforms and choose an available one.*/
		cl_uint numPlatforms;	//the NO. of platforms
		cl_platform_id platform = NULL;	//the chosen platform
		cl_int	status = clGetPlatformIDs(0, NULL, &numPlatforms);
		if (status != CL_SUCCESS)
		{
			printf("Error: Getting platforms!\n");
			return -1;
		}

		/*For clarity, choose the first available platform. */
		if(numPlatforms > 0)
		{
			cl_platform_id* platforms = (cl_platform_id* )malloc(numPlatforms* sizeof(cl_platform_id));
			status = clGetPlatformIDs(numPlatforms, platforms, NULL);
			platform = platforms[0];
			free(platforms);
		}

		/*Step 2:Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device.*/
		cl_uint				numDevices = 0;
		cl_device_id        *devices;
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);	
		if (numDevices == 0)	//no GPU available.
		{
			printf("No GPU device available.\n");
			printf("Choose CPU as default device.\n");
			status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);	
			devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
			status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
		}
		else
		{
			devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
			status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
		}


		/*Step 3: Create context.*/
		context = clCreateContext(NULL,1, devices,NULL,NULL,NULL);

		/*Step 4: Creating command queue associate with the context.*/
		command_queue = clCreateCommandQueue(context, devices[0], 0, NULL);


	printf("creating program\n");
		// Create a program from the kernel source
		cl_program program =
		CHECK_clCreateProgramWithSource(context,
						1,
						(const char**)&source_str,
						&source_size,
						err_clCreateProgramWithSource);

		free(source_str); //XXX can we do that while program is still alive ?

		// Build the program
	printf("building program\n");
		//cl_int ret = clBuildProgram(program, 1, devices, "-cl-opt-disable -g", NULL, NULL);
		cl_int ret = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
		//cl_int ret = clBuildProgram(program, 1, devices, "-g", NULL, NULL);
		if (ret) {
			printf("clBuildprogram ret: %d\n", ret);
		//cl_int ret0=ret; XX print it?
		size_t sizeused;
		clGetProgramBuildInfo (program,
						 devices[0],
						 CL_PROGRAM_BUILD_LOG,
						 val_size-1, //?
						 &val,
						 &sizeused);

		printf("clBuildProgram error: (sizeused %lu) '%s'\n",
			   (uintptr_t) sizeused, val);
		err_clGetProgramBuildInfo:
		goto err_clBuildProgram;
		}

		// Create the OpenCL kernel
	printf("Creating opencl kernel\n");
		kernel = CHECK_clCreateKernel(program, "cudaized_multi",
							err_clCreateKernel);
		opencl_inited = 1;
	}
	if (!balloon_inited[gpuid]) {
		printf("Initiated GPU %d\n", gpuid);
		device_prebuf_le[gpuid] = clCreateBuffer(context, CL_MEM_READ_ONLY, PREBUF_LEN/8 * sizeof(uint64_t), NULL, NULL);
		device_sbuf[gpuid] = clCreateBuffer(context, CL_MEM_READ_ONLY, /*s.n_blocks*/4096 * BLOCK_SIZE, NULL, NULL);
		device_is_winning[gpuid] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t), NULL, NULL);
		device_winning_nonce[gpuid] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t), NULL, NULL);
		device_s[gpuid] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(struct hash_state), NULL, NULL);
		device_target[gpuid] = clCreateBuffer(context, CL_MEM_READ_ONLY, 8*sizeof(uint32_t), NULL, NULL);
		device_out[gpuid] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, BLOCK_SIZE * sizeof(uint8_t), NULL, NULL);
		device_input[gpuid] = clCreateBuffer(context, CL_MEM_READ_ONLY, /*len*/80, NULL, NULL);
		printf("Allocating %u bytes to device_hs_sbufs (num_threads %d, num_blocks %d)\n", num_threads*num_blocks*4096*BLOCK_SIZE, num_threads, num_blocks);
		device_hs_sbufs[gpuid] = clCreateBuffer(context, CL_MEM_READ_WRITE, num_threads * num_blocks * 4096 * BLOCK_SIZE, NULL, NULL);

		a_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 10, NULL, NULL);
		balloon_inited[gpuid] = 1;
	}
    // Set the arguments of the kernel
	// hash_state
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_hs_sbufs[gpuid]);
	// prebuf_le
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_prebuf_le[gpuid]);
	// input
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &device_input[gpuid]);
	// mixrounds
	cl_int mixrounds = 4;
    CHECK_clSetKernelArg(kernel, 1, sizeof(cl_int), &mixrounds, err_clSetKernelArg);
	// len
	cl_uint len = 80;
    CHECK_clSetKernelArg(kernel, 4, sizeof(cl_uint), &len, err_clSetKernelArg);
	// output
    CHECK_clSetKernelArg(kernel, 5, sizeof(cl_mem), &device_out[gpuid], err_clSetKernelArg);
	// s_cost
	cl_long s_cost = 128;
    CHECK_clSetKernelArg(kernel, 6, sizeof(cl_long), &s_cost, err_clSetKernelArg);
	// max_nonce
	cl_int max_nonce = 1000;
    CHECK_clSetKernelArg(kernel, 7, sizeof(cl_int), &max_nonce, err_clSetKernelArg);
	// gpuid
	cl_int cl_gpu = gpuid;
    CHECK_clSetKernelArg(kernel, 8, sizeof(cl_int), &cl_gpu, err_clSetKernelArg);
	// winning_nonce
    CHECK_clSetKernelArg(kernel, 9, sizeof(cl_mem), &device_winning_nonce[gpuid], err_clSetKernelArg);
	// num_threads
	cl_int cl_threads = num_threads;
    CHECK_clSetKernelArg(kernel, 10, sizeof(cl_int), &cl_threads, err_clSetKernelArg);
	// device_target
    CHECK_clSetKernelArg(kernel, 11, sizeof(cl_mem), &device_target[gpuid], err_clSetKernelArg);
	// is_winning
    CHECK_clSetKernelArg(kernel, 12, sizeof(cl_mem), &device_is_winning[gpuid], err_clSetKernelArg);
	// num_blocks
	cl_int cl_blocks = num_blocks;
    CHECK_clSetKernelArg(kernel, 13, sizeof(cl_int), &cl_blocks, err_clSetKernelArg);
	// sbufs
    CHECK_clSetKernelArg(kernel, 14, sizeof(cl_mem), &device_sbuf[gpuid], err_clSetKernelArg);



 err_clEnqueueReadBuffer:
 err_malloc_C:
    CHECK_clFlush(command_queue, err_clFlush);
 err_clFlush:
    CHECK_clFinish(command_queue, err_clEnqueueNDRangeKernel);
 err_clEnqueueNDRangeKernel:
 err_clSetKernelArg:
    //CHECK_clReleaseKernel(kernel, err_clCreateKernel);
 err_clCreateKernel:
 err_clBuildProgram:
    //CHECK_clReleaseProgram(program, err_clCreateProgramWithSource);
 err_clCreateProgramWithSource:
 err_clEnqueueWriteBuffer_B:
 err_clEnqueueWriteBuffer_A:
    //CHECK_clReleaseMemObject(c_mem_obj, err_c_mem_obj);
 err_c_mem_obj:
    //CHECK_clReleaseMemObject(b_mem_obj, err_b_mem_obj);
 err_b_mem_obj:
    //CHECK_clReleaseMemObject(a_mem_obj, err_a_mem_obj);
 err_a_mem_obj:
    //CHECK_clReleaseCommandQueue(command_queue, err_clCreateCommandQueue);
 err_clCreateCommandQueue:
    //CHECK_clReleaseContext(context, err_clCreateContext);
 err_clCreateContext:
    // XXX deallocate device_id ?
 err_clGetDeviceIDs:
    // XXX deallocate platform_id ?
 err_clGetPlatformIDs:
 err_malloc_source_str:
 err_fopen:
    //free(B);
 err_malloc_B:
    //free(A);
 err_malloc_A:
	printf("ending opencl init\n");
    return CHECK_errors;
}

void fill_prebuf(struct hash_state *s, int gpuid) {
#ifdef DEBUG
	printf("DEBUG GPU %d: entering fill_prebuf\n", gpuid);
#endif
	uint8_t host_prebuf[PREBUF_LEN];
	if (!host_prebuf_filled[gpuid]) {
		bitstream_fill_buffer (&s->bstream, host_prebuf, PREBUF_LEN);
		host_prebuf_filled[gpuid] = 1;
		uint8_t *buf = host_prebuf;
		uint64_t *lebuf = host_prebuf_le[gpuid];
		for (int i = 0; i < PREBUF_LEN; i+=8) {
			bytes_to_littleend8_uint64(buf, lebuf);
			*lebuf %= 4096;
			*lebuf <<= 5; // multiply by 32
			lebuf++;
			buf += 8;
		}
		update_device_data(gpuid);
		//printf("Filled prebuf for GPU %d\n", gpuid);
	}
#ifdef DEBUG
	printf("DEBUG GPU %d: leaving fill_prebuf\n", gpuid);
#endif
}

void reset_host_prebuf() {
	for (int i = 0; i < 20; i++) {
		host_prebuf_filled[i] = 0;
	}
}


void update_device_data(int gpuid) {
#ifdef DEBUG
	printf("DEBUG GPU %d: entering update_device_data\n", gpuid);
#endif
    clEnqueueWriteBuffer(command_queue, device_prebuf_le[gpuid], CL_TRUE, 0, PREBUF_LEN, host_prebuf_le[gpuid], 0, NULL, NULL);
#ifdef DEBUG
	printf("DEBUG GPU %d: leaving update_device_data\n", gpuid);
#endif
}

void balloon_cuda_free(int gpuid) {
	//cudaFree(device_prebuf_le[gpuid]);
	//cudaFree(device_sbuf[gpuid]);
	//cudaFree(device_s[gpuid]);
	//cudaFree(device_winning_nonce[gpuid]);
	//cudaFree(device_is_winning[gpuid]);
	//cudaFree(device_out[gpuid]);
	//cudaFree(device_input[gpuid]);
#ifdef LOWMEM
	//cudaFree(device_sbufs[gpuid]);
#endif
	//balloon_inited = 0;
}

uint32_t balloon_128_cuda (int gpuid, unsigned char *input, unsigned char *output, uint32_t *target, uint32_t max_nonce, uint32_t num_threads, uint32_t *is_winning, uint32_t num_blocks) {
	return cuda_balloon (gpuid, input, output, 80, 128, 4, target, max_nonce, num_threads, is_winning, num_blocks);
}

//#define NUM_THREADS 256
//#define NUM_THREADS 384
//#define NUM_THREADS 384
//#define NUM_BLOCKS 480
//#define NUM_BLOCKS 48


uint32_t cuda_balloon(int gpuid, unsigned char *input, unsigned char *output, int32_t len, int64_t s_cost, int32_t t_cost, uint32_t *target, uint32_t max_nonce, uint32_t num_threads, uint32_t *ret_is_winning, uint32_t num_blocks) {
#ifdef DEBUG
	printf("DEBUG GPU %d: entering opencl_balloon\n", gpuid);
#endif
	DECLARE_CHECK;

	struct balloon_options opts;
	struct hash_state s;
	balloon_init(&opts, s_cost, t_cost);
	hash_state_init(&s, &opts, input);
	fill_prebuf(&s, gpuid);
	uint8_t *pc_sbuf = s.buffer;

#ifdef DEBUG
	if (s.n_blocks > 4096) printf("s.n_blocks = %llu\n", s.n_blocks);
#endif

	uint32_t first_nonce = ((input[76] << 24) | (input[77] << 16) | (input[78] << 8) | input[79]);

	printf("opencl_ballon, gpu %d, start_nonce: %d, max_nonce: %d\n", gpuid, first_nonce, max_nonce);

	uint32_t host_winning_nonce = 0;
	uint32_t host_is_winning = 0;

	printf("Copying data to opencl\n");

	// s_cost
	cl_long cl_s_cost = s_cost;
    clSetKernelArg(kernel, 6, sizeof(cl_long), &cl_s_cost);
	// max_nonce
	cl_int cl_max_nonce = max_nonce;
    clSetKernelArg(kernel, 7, sizeof(cl_int), &cl_max_nonce);

    clEnqueueWriteBuffer(command_queue, device_input[gpuid], CL_TRUE, 0, len, input, 0, NULL, NULL);
    //clEnqueueWriteBuffer(command_queue, device_prebuf_le[gpuid], CL_TRUE, 0, PREBUF_LEN, host_prebuf_le[gpuid], 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, device_sbuf[gpuid], CL_TRUE, 0, s.n_blocks*BLOCK_SIZE, s.buffer, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, device_winning_nonce[gpuid], CL_TRUE, 0, sizeof(uint32_t), &host_winning_nonce, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, device_is_winning[gpuid], CL_TRUE, 0, sizeof(uint32_t), &host_is_winning, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, device_target[gpuid], CL_TRUE, 0, 8 * sizeof(uint32_t), target, 0, NULL, NULL);

    // Execute the OpenCL kernel on the list
printf("About to execute opencl kernel\n");
fflush(stdout);
    //size_t global_item_size = LIST_SIZE; // Process the entire lists
    size_t global_item_size = num_threads * num_blocks; // Process the entire lists
    //size_t local_item_size = 64; // Process in groups of 64
    size_t local_item_size = num_threads; // Process in groups of 64
    CHECK_clEnqueueNDRangeKernel
	(command_queue, kernel, 1, NULL,
	 &global_item_size, &local_item_size, 0, NULL, NULL,
	 err_clEnqueueNDRangeKernel);

	printf("Waiting for kernel to complete\n");
	fflush(stdout);

    CHECK_clFinish(command_queue, err_clEnqueueNDRangeKernel);
printf("Opencl kernel done\n");
	fflush(stdout);

#ifdef CUDA_OUTPUT
    CHECK_clEnqueueReadBuffer(command_queue, device_out[gpuid], CL_TRUE, 0, 
			      BLOCK_SIZE * sizeof(uint8_t), output, 0, NULL, NULL,
			      err_clEnqueueReadBuffer);
	printf("output: ");
	for (int i = 0; i < 32; i++) {
		printf("%02x ", output[i]);
	}
	printf("\n");
#endif
    CHECK_clEnqueueReadBuffer(command_queue, device_winning_nonce[gpuid], CL_TRUE, 0, 
			      sizeof(uint32_t), &host_winning_nonce, 0, NULL, NULL,
			      err_clEnqueueReadBuffer);
    CHECK_clEnqueueReadBuffer(command_queue, device_is_winning[gpuid], CL_TRUE, 0, 
			      sizeof(uint32_t), &host_is_winning, 0, NULL, NULL,
			      err_clEnqueueReadBuffer);

	printf("after read buffer\n");

#ifdef DEBUG
	if (host_is_winning) {
		printf("[Host (GPU %d)] Winning (%d) nonce: %u\n", gpuid, host_is_winning, host_winning_nonce);
	}
#endif

	s.buffer = pc_sbuf;
	hash_state_free(&s);

	*ret_is_winning = host_is_winning;
	if (host_is_winning == 0) {
		host_winning_nonce = first_nonce + num_threads*num_blocks - 1;
	}

err_clEnqueueReadBuffer:
err_clEnqueueNDRangeKernel:

	return host_winning_nonce;
}

