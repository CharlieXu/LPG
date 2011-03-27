///////////////////////////////////////////////////////////////////////////////
// LPG - libLPG																 //
// Version 1.0																 //
// Implements RSM for the solution of LPs									 //
// ------------------------------------------------------------------------- //
//																			 //
// model.cpp																 //
// Functions to manipulate models											 //
//																			 //
// (c) Iain Dunning 2011													 //
///////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
// Function prototypes and necessary definitions
#include "libLPG.hpp"
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Standard Lib Includes
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// ConvertFromGeneralFormToInternal
// Take a problem in the form:
//		min   c^T x
//		st b1 <= Ax <= b2
//		   l  <=  x <= u
// and convert to
//		min   c^T x
//		st       Ax == b (>=0)
//		    l <=  x <= u
void ConvertFromGeneralFormToInternal(
	// Input
	const CoinPackedMatrix& matrix,
	const double* collb, const double* colub,   
	const double* obj,
	const double* rowlb, const double* rowub,
	// Output
	int& m, int& n,
	CoinPackedMatrix *&sparseA,
	double *&A, double *&b, double *&c,
	double *&xLB, double *&xUB,
	// Options
	bool silentMode
	)
{

	//-------------------------------------------------------------------------
	// 1. INITIAL PROBLEM SIZE
	n = matrix.getNumCols();	// Number of variables
	m = matrix.getNumRows();	// Number of constraints
	int originalN = n, originalM = m;

	//-------------------------------------------------------------------------
	// 2. CREATE LOCAL COPY OF A MATRIX
	sparseA = new CoinPackedMatrix(matrix);
	if (!sparseA->isColOrdered()) sparseA->reverseOrdering(); // Force to column ordering

	//-------------------------------------------------------------------------
	// 3. CONVERT CONSTRAINTS
	std::vector<double> correctedRHS;
	// 3.1 Pass #1: Add slacks to LEQ, GEQs, turn ranges into two rows
	for (int row = 0; row < m; row++) {
		//---------------------------------------------------------------------
		// EQUALITY CONSTRAINT
		//---------------------------------------------------------------------
		if (fabs(rowlb[row] - rowub[row]) < LPG_TOL) {
			// Lower bound and upper bound are equal
			if (!silentMode) printf("ROW%03d: Equality constraint. No action required.\n", row);
			// Record the RHS
			correctedRHS.push_back(rowlb[row]); // Either UB or LB is OK

		//---------------------------------------------------------------------
		// LEQ CONSTRAINT
		//---------------------------------------------------------------------
		} else if (rowlb[row] < -LPG_BIG) {
			// Lower bound is really negative -> <= constraint
			if (!silentMode) printf("ROW%03d: LEQ constraint. Action: add +ve slack variable.\n", row);
			// Add a column to A matrix
			double slackCoeff = +1.0;
			sparseA->appendCol(1, &row, &slackCoeff);
			n++; // Variable added
			// Record the RHS
			correctedRHS.push_back(rowub[row]); // UB is correct one

		//---------------------------------------------------------------------
		// GEQ CONSTRAINT
		//---------------------------------------------------------------------
		} else if (rowub[row] > LPG_BIG) {
			// Upper bound is really positive -> >= constraint
			if (!silentMode) printf("ROW%03d: GEQ constraint. Action: add -ve slack variable.\n", row);
			// Add a column to A matrix
			double slackCoeff = -1.0;
			sparseA->appendCol(1, &row, &slackCoeff);
			n++; // Variable added
			// Record the RHS
			correctedRHS.push_back(rowlb[row]); // LB is correct one

		//---------------------------------------------------------------------
		// RANGE CONSTRAINT (e.g. l <= aTx <= b) - NOT SUPPORTED!!!!
		//---------------------------------------------------------------------
		} else {
			printf("ROW%03d: RANGE constraint. Action: PANIC! [not supported]\n", row);
			assert(false); // Force crash
		}
	}
	// 3.2 Pass #2: Force RHS to be positive
	for (int row = 0; row < m; row++) {
		if (correctedRHS[row] < 0) {
			// This constraint has a negative RHS, so make it positive...
			correctedRHS[row] = -correctedRHS[row];
			// ... and flip signs in the corresponding row of A
			for (int col = 0; col < n; col++) {
				sparseA->modifyCoefficient(row, col, sparseA->getCoefficient(row,col) * -1.0);
			}
		}
	}

	//-------------------------------------------------------------------------
	// 4. ALLOCATE MEMORY
	A   = (double*)malloc(sizeof(double) * m * n);
	b   = (double*)malloc(sizeof(double) * m    );
	c   = (double*)malloc(sizeof(double) *     n); 
	xUB = (double*)malloc(sizeof(double) *     n);
	xLB = (double*)malloc(sizeof(double) *     n);
	
	//-------------------------------------------------------------------------
	// 5. COPY DATA INTO NEW MEMORY
	// 5.1		b vector
	for (int row = 0; row < m; row++) b[row] = correctedRHS[row];

	// 5.2		c vector
	for (int col = 0; col < originalN; col++) c[col] = obj[col];
	for (int col = originalN; col < n; col++) c[col] = 0.0;

	// 5.3		xUB, xLB
	// 5.3.1	Check for any free variables
	for (int i = 0; i < originalN; i++) {
		double absLB = fabs(collb[i]), absUB = fabs(colub[i]);
		if ( (absLB > LPG_BIG) && (absUB > LPG_BIG) ) 
			printf("COL%03d: Free variables are unsupported! Can't handle yet. Reformulate!\n", i);
	}
	// 5.3.2	Copy over to new structure
	for (int col = 0; col < originalN; col++) xLB[col] = collb[col];
	for (int col = 0; col < originalN; col++) xUB[col] = colub[col];
	for (int col = originalN; col < n; col++) xLB[col] = 0.0;
	for (int col = originalN; col < n; col++) xUB[col] = LPG_BIG;

	// 5.4		A matrix (sparseA is col-major)
	for (int element = 0; element < m*n; element++)	A[element] = 0.0;
	for (int col = 0; col < n; col++) {
		int start	  = sparseA->getVectorStarts()[col];
		int startNext = (col == n-1) ? sparseA->getNumElements() : sparseA->getVectorStarts()[col+1];
		for (int i = start; i < startNext; i++) {
			A[sparseA->getIndices()[i]*n + col] = sparseA->getMutableElements()[i];
		}
	}

	//-------------------------------------------------------------------------
	// 6. DONE!
}


//-----------------------------------------------------------------------------
void DebugPrint(char* what, double* data, int size) {
	
	int curPos = 0;
	printf("%s:\n",what);
	while (true) {
		if (size - curPos > 8) {
			for (int i = 0; i < 8; i++) printf("[%8d]", curPos+i);
			for (int i = 0; i < 8; i++) printf("[%8.3f]", data[curPos+i]);
			curPos += 8;
		} else {
			for (int i = curPos; i < size; i++) printf("[%8d]", i);
			printf("\n");
			for (int i = curPos; i < size; i++) printf("[%8.3f]", data[i]);
			printf("\n");
			break;
		}
	}
}
void DebugPrint(char* what, int* data, int size) {
	
	int curPos = 0;
	printf("%s:\n",what);
	while (true) {
		if (size - curPos > 8) {
			for (int i = 0; i < 8; i++) printf("[%8d]", curPos+i);
			for (int i = 0; i < 8; i++) printf("[%8d]", data[curPos+i]);
			curPos += 8;
		} else {
			for (int i = curPos; i < size; i++) printf("[%8d]", i);
			printf("\n");
			for (int i = curPos; i < size; i++) printf("[%8d]", data[i]);
			printf("\n");
			break;
		}
	}
}

