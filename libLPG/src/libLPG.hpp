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
// OpenCL
// Additional include directories:
//C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA\v3.2\lib\Win32
//C:\Program Files (x86)\ATI Stream\lib\x86
// Additional library directories:
//C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA\v3.2\include;
//C:\Program Files (x86)\ATI Stream\include
#pragma comment(lib, "OpenCL.lib")
#include <CL/cl.h>
#define MAX_SOURCE_SIZE (0x100000)
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Constants
// Flags for optimisation status
#define LPG_OPTIMAL 0
#define LPG_INFEASIBLE 1
#define LPG_UNBOUNDED 2
#define LPG_UNKNOWN 3
// Implement precision, floating point stuff
#define LPG_TOL 1e-7
#define LPG_BIG 1e100
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Sparse matrix class
// Reinventing the wheel? Yes!
// Educational? You bet
// Of note: this matrix is read-only, after initial creation
class LPGSparseMatrix {

public:
	int m, n; // Matrix size
	
	double** values;
	int** indices;
	int* nzeros;

	LPGSparseMatrix() { m = -1; n = -1; values = NULL; indices = NULL; }

	LPGSparseMatrix(double* fullMat, int numRows, int numCols);

	void BuildSparse(double* fullMat, int numRows, int numCols);

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
	double *A;
	CoinPackedMatrix* coinSparseA;
	LPGSparseMatrix* sparseA;
	double *b, *c;
	double *xLB, *xUB;

	double z;
	double *x;
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
void DebugPrint(char* what, double* data,	int size);
void DebugPrint(char* what, int* data,		int size);

//-----------------------------------------------------------------------------
// Include guard
//-----------------------------------------------------------------------------
#endif


		

//-----------------------------------------------------------------------------

//	void SolveLP_C (
//					int m, int n,							// Problem size
//					double *A, double *b, double *c_orig,	// } Problem
//					double *xLB, double *xUB,				// }
//					double &z, double *x_ans, int &status);	// Output
//	void SolveLP_G (
//					int m, int n,							// Problem size
//					double *A, double *b, double *c_orig,	// } Problem
//					double *xLB, double *xUB,				// }
//					double &z, double *x_ans, int &status);	// Output
//-----------------------------------------------------------------------------