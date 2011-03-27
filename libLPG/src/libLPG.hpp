///////////////////////////////////////////////////////////////////////////////
// LPG - libLPG																 //
// Version 1.0																 //
// Implements RSM for the solution of LPs									 //
// ------------------------------------------------------------------------- //
//																			 //
// libLPG.hpp																 //
// Main include for LPG. Prototypes various solvers, allows config using	 //
// defines, and provides some constants.									 //
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
// Flags for optimisation status
#define LPG_OPTIMAL 0
#define LPG_INFEASIBLE 1
#define LPG_UNBOUNDED 2
#define LPG_UNKNOWN 3
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
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
// Solvers
void SolveLP_C (
				int m, int n,							// Problem size
				double *A, double *b, double *c_orig,	// } Problem
				double *xLB, double *xUB,				// }
				double &z, double *x_ans, int &status);	// Output
void SolveLP_G (
				int m, int n,							// Problem size
				double *A, double *b, double *c_orig,	// } Problem
				double *xLB, double *xUB,				// }
				double &z, double *x_ans, int &status);	// Output
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Utility functions
void ConvertFromGeneralFormToInternal(
	// Input
	const CoinPackedMatrix& matrix,	
	const double* collb, const double* colub, 
	const double* obj, 
	const double* rowlb, const double* rowub,
	// Output
	int& m, int& n,	
	CoinPackedMatrix *&sparseA,	double *&A, 
	double *&b, double *&c,	
	double *&xLB, double *&xUB,
	// Options
	bool silentMode = false
	);

//-----------------------------------------------------------------------------
// Misc functions
void DebugPrint(char* what, double* data, int size);
void DebugPrint(char* what, int* data, int size);

//-----------------------------------------------------------------------------
// Include guard
//-----------------------------------------------------------------------------
#endif