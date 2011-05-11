///////////////////////////////////////////////////////////////////////////////
// LPG - libLPG																 //
// Implements RSM for the solution of LPs									 //
// ------------------------------------------------------------------------- //
//																			 //
// libLPG.hpp																 //
// Main include for LPG. Defines the LPG class, support functions and LPG    //
// constants.																 //
//																			 //
// (c) Iain Dunning 2011													 //
///////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
// Include guard
#ifndef libLPG_HPP
#define libLPG_HPP
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Includes from COIN
#include "CoinPackedMatrix.hpp"
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Specify precision to use
#include <cfloat>
// Must also be set in common.clh for correct operation of GPU solver
//#define SCALAR float
//#define LPG_TOL 1e-7f
//#define LPG_BIG FLT_MAX
#define SCALAR double
#define LPG_TOL 1e-7
#define LPG_BIG 1e100

//-----------------------------------------------------------------------------
// OpenCL
#include <CL/cl.h>
#define MAX_SOURCE_SIZE (0x100000)

// Preferred OpenCL platform - will use this platform if present
#define PREFPLATFORM "Advanced Micro Devices, Inc."
//#define PREFPLATFORM "NVIDIA Corporation"

// Preferred OpenCL device type
#define PREFDEVICE CL_DEVICE_TYPE_CPU
//#define PREFDEVICE CL_DEVICE_TYPE_GPU
//#define PREFDEVICE CL_DEVICE_TYPE_DEFAULT

// OpenCL error checker - useful for debugging
// USE: Just write CL_ERR_CHECK (no semicolon) after a OpenCL call
//		You can find the various error codes in CL/cl.h
#define CL_ERR_CHECK if (ret != 0) { printf("Err code=%d\n", ret); assert(ret==0); }
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Constants
// Flags for optimisation status
#define LPG_OPTIMAL 0
#define LPG_INFEASIBLE 1
#define LPG_UNBOUNDED 2
#define LPG_UNKNOWN 3
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Sparse matrix class
// Reinventing the wheel? Yes!
// Educational? You bet
// Of note: this matrix is read-only, after initial creation
class LPGSparseMatrix {

public:
	int m, n; // Matrix size
	
	SCALAR** values;
	int** indices;
	int* nzeros;

	LPGSparseMatrix() { m = -1; n = -1; values = NULL; indices = NULL; }

	LPGSparseMatrix(SCALAR* fullMat, int numRows, int numCols);

	void BuildSparse(SCALAR* fullMat, int numRows, int numCols);

	void PrintMatrix();
};

//-----------------------------------------------------------------------------
// LPG Class
class LPG {
public:

	// Constructor
	LPG  (bool prepareGPU = false, bool verboseMode = false)
		{ 
			isLoaded = false; 
			verbose = verboseMode;
			isOpenCLinit = false;
			if (prepareGPU) { InitGPU(); InitKernels(); }
		}
	~LPG ()
		{ FreeModelIfNeeded(); }

	// Solvers
	void SolveCPU();
	void SolveGPU();
	
	// Optional GPU setup functions
	void InitKernels();
	void InitGPU();
	void FreeGPUandKernels();

	// GPU status
	static bool GPUloaded;

	// Model manipulation
	void LoadMPS(const char* filename);
	void LoadLP (const char* filename);

	// Other
	bool verbose;

//private:
	bool isLoaded;
	void FreeModelIfNeeded();

	int m, n;
	SCALAR *A;
	CoinPackedMatrix* coinSparseA;
	LPGSparseMatrix* sparseA;
	SCALAR *b, *c;
	SCALAR *xLB, *xUB;

	SCALAR z;
	SCALAR *x;
	int status;

	// Convert to internal storage format from the CoinUtils
	// loader format
	void InternalForm(
		const CoinPackedMatrix& matrix,	
		const double* collb, const double* colub, 
		const double* obj, 
		const double* rowlb, const double* rowub
		);

private:

	// OpenCL helper functions
	void LoadKernel(char* fileName, char* kernelName, cl_program& program, cl_kernel& kernel);

	// OpenCL globals
	bool isOpenCLinit;
	cl_platform_id platformID;
	cl_device_id deviceID;   
	cl_uint numDevices, numPlatforms;
	cl_context context;
	cl_command_queue commandQueue;

	// OpenCL kernels
	cl_program dual_program;		cl_kernel dual_kernel;
	cl_program rc1_program;			cl_kernel rc1_kernel;
	cl_program rc2_program;			cl_kernel rc2_kernel;
	cl_program binvas_program;		cl_kernel binvas_kernel;
	cl_program tableau1_program;	cl_kernel tableau1_kernel;
	cl_program tableau2_program;	cl_kernel tableau2_kernel;
};
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Misc functions
void DebugPrint(char* what, SCALAR* data,	int size);
void DebugPrint(char* what, int* data,		int size);

//-----------------------------------------------------------------------------
// Include guard
//-----------------------------------------------------------------------------
#endif