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
// GPU-specific initialisation functions
void InitKernels();
void InitGPU();
void FreeGPUandKernels();
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// LPG Class
class LPG {
public:

	// Constructor
	LPG(bool verboseMode = false)
		{ isLoaded = false; verbose = verboseMode; }
	~LPG()
		{ FreeModelIfNeeded(); }

	// Solvers
	void SolveCPU();
	void SolveGPU();

	// GPU status
	static bool GPUloaded;

	// Model manipulation
	void LoadMPS(const char* filename);
	void LoadLP (const char* filename);

	// Other
	bool verbose;

private:

	bool isLoaded;
	void FreeModelIfNeeded();

	int m, n;
	double *A;
	CoinPackedMatrix* coinSparseA;
	double *b, *c;
	double *xLB, *xUB;

	double z;
	double *x_ans;
	int status;

	// Convert to internal storage format from the CoinUtils
	// loader format
	void InternalForm(
		const CoinPackedMatrix& matrix,	
		const double* collb, const double* colub, 
		const double* obj, 
		const double* rowlb, const double* rowub
		);

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