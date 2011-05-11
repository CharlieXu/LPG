///////////////////////////////////////////////////////////////////////////////
// LPG - libLPG																 //
// Implements RSM for the solution of LPs									 //
// ------------------------------------------------------------------------- //
//																			 //
// GPUsupport.cpp															 //
// All support functions for the GPU code.                           		 //
//																			 //
// (c) Iain Dunning 2011													 //
///////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
// Prototypes, definitions, etc.
#include "libLPG.hpp"
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Standard Lib Includes
#include <cstdio>
#include <cstdlib>
#include <cmath>
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// OpenCL error checker - useful for debugging
// USE: Just write CL_ERR_CHECK (no semicolon) after a OpenCL call
//		You can find the various error codes in CL/cl.h
#define CL_ERR_CHECK if (ret != 0) { printf("Err code=%d\n", ret); assert(ret==0); }
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// LoadKernel
// Loads an OpenCL kernel from a file
// Inputs: fileName, kernelName
// Outputs: program, kernel
// NOTE: Makes use of globals - thus only call after InitGPU
void LPG::LoadKernel(char* fileName, char* kernelName, cl_program& program, cl_kernel& kernel)
{
	// Prevent this running before InitGPU
	if (!isOpenCLinit) {
		printf("LoadKernel: Cannot call before InitGPU is called!");
		exit(EXIT_FAILURE);
	}

	// OpenCL error code
	int errCode = 0;
	
	// Load the kernel from file
	FILE *fp = NULL;
    char *source_str;
    size_t source_size;
    fp = fopen(fileName, "r");
	if (fp == NULL) {
		printf("LoadKernel: Failed to load kernel %s (fileName: %s).\n", kernelName, fileName);
		exit(EXIT_FAILURE);
    }
    source_str  = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

	// Create a program from the kernel source
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &errCode);
	if (errCode != 0) {
		printf("LoadKernel: Error code after clCreateProgramWithSource for kernal '%s': %d\n", kernelName, errCode);
		exit(EXIT_FAILURE);
	}

    // Build the program
    errCode = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);
	if (errCode != 0) {
		printf("LoadKernel: Error code after clBuildProgram for kernal '%s': %d\n", kernelName, errCode);
		exit(EXIT_FAILURE);
	}

    // Create the OpenCL kernel
    kernel = clCreateKernel(program, kernelName, &errCode);
	if (errCode != 0) {
		printf("LoadKernel: Error code after clCreateKernel for kernal '%s': %d\n", kernelName, errCode);
		exit(EXIT_FAILURE);
	}

	free(source_str);
	return;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// InitGPU
// Sets up the device, context and commandQueue
void LPG::InitGPU()
{
	if (isOpenCLinit) return;
	cl_int ret;
	
	ret = clGetPlatformIDs(0, NULL, &numPlatforms); CL_ERR_CHECK 
	printf("clGetPlatformIDs: numPlatforms = %d\n", numPlatforms);
	
	cl_platform_id* platforms = new cl_platform_id[numPlatforms]; CL_ERR_CHECK
	ret = clGetPlatformIDs(numPlatforms, platforms, NULL); CL_ERR_CHECK 

	for (int i = 0; i < numPlatforms; i++)
	{
		char pbuf[100];
		ret = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR,
								sizeof(pbuf), pbuf, NULL); CL_ERR_CHECK
		printf("clGetPlatformInfo: #%d = %s\n", i, pbuf);
		platformID = platforms[i];
		// Try and take the preferred platform
		if (!strcmp(pbuf, PREFPLATFORM)) break;
	}

	delete[] platforms;
	
    ret = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &numDevices);
	CL_ERR_CHECK
	
	// Create an OpenCL context
    context = clCreateContext( NULL, 1, &deviceID, NULL, NULL, &ret);
	CL_ERR_CHECK
 
    // Create a command queue
    commandQueue = clCreateCommandQueue(context, deviceID, 0, &ret);
	CL_ERR_CHECK

	// Set flag to let rest of program know its ok to proceed
	isOpenCLinit = true;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// InitKernels
// Loads all the kernels that will be needed
void LPG::InitKernels()
{	
	LoadKernel("C:/Users/Iain/Dev/LPG/libLPG/kernel/dual_kernel.cl",	"dual",		dual_program,		dual_kernel		);
	LoadKernel("C:/Users/Iain/Dev/LPG/libLPG/kernel/rc1_kernel.cl",		"rc1",		rc1_program,		rc1_kernel		);
	LoadKernel("C:/Users/Iain/Dev/LPG/libLPG/kernel/rc2_kernel.cl",		"rc2",		rc1_program,		rc2_kernel		);
	LoadKernel("C:/Users/Iain/Dev/LPG/libLPG/kernel/binvas_kernel.cl",	"binvas",	binvas_program,		binvas_kernel	);
	LoadKernel("C:/Users/Iain/Dev/LPG/libLPG/kernel/tableau1_kernel.cl","tableau1",	tableau1_program,	tableau1_kernel	);
	LoadKernel("C:/Users/Iain/Dev/LPG/libLPG/kernel/tableau2_kernel.cl","tableau2",	tableau2_program,	tableau2_kernel	);
}


//-----------------------------------------------------------------------------
// FreeGPUandKernels
// Unload all the kernels and close down OpenCL
void LPG::FreeGPUandKernels()
{
	cl_int ret;
	// Kernels
	ret = clReleaseKernel(dual_kernel);	    ret = clReleaseProgram(dual_program);
	ret = clReleaseKernel(rc1_kernel);	    ret = clReleaseProgram(rc1_program);
	ret = clReleaseKernel(rc2_kernel);	    ret = clReleaseProgram(rc2_program);
	ret = clReleaseKernel(binvas_kernel);	ret = clReleaseProgram(binvas_program);
	ret = clReleaseKernel(tableau1_kernel);	ret = clReleaseProgram(tableau1_program);
	ret = clReleaseKernel(tableau2_kernel);	ret = clReleaseProgram(tableau2_program);
	
	// Context
    ret = clReleaseCommandQueue(commandQueue);
    ret = clReleaseContext(context);

	isOpenCLinit = false;
}