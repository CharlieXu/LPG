///////////////////////////////////////////////////////////////////////////////
// LPG - libLPG																 //
// Implements RSM for the solution of LPs									 //
// ------------------------------------------------------------------------- //
//																			 //
// LPGSparseMatrix.cpp														 //
// Defines the internal sparse matrix class									 //
//																			 //
// (c) Iain Dunning 2011													 //
///////////////////////////////////////////////////////////////////////////////

#include "libLPG.hpp"
#include <cmath>
#include <cstdio>

LPGSparseMatrix::LPGSparseMatrix(double* fullMat, int numRows, int numCols) {
	BuildSparse(fullMat, numRows, numCols);
}

void LPGSparseMatrix::BuildSparse(double* fullMat, int numRows, int numCols) {

	m = numRows;
	n = numCols;

	indices = (int**)	malloc(sizeof(int*)		* numCols);
	values  = (double**)malloc(sizeof(double*)	* numCols);
	nzeros  = (int*)	malloc(sizeof(int)		* numCols);
	
	for (int c = 0; c < numCols; c++) {
		int nonzeroCount = 0;
		
		for (int r = 0; r < numRows; r++) {
			if (fabs(fullMat[r + c*numRows]) > LPG_TOL) nonzeroCount++;
		}

		indices[c]	= (int*)	malloc(sizeof(int)		* nonzeroCount);
		values[c]   = (double*)	malloc(sizeof(double)	* nonzeroCount);
		nzeros[c]	= nonzeroCount;

		nonzeroCount = 0;
		for (int r = 0; r < numRows; r++) {
			if (fabs(fullMat[r + c*numRows]) > LPG_TOL) {
				indices[c][nonzeroCount] = r;
				values [c][nonzeroCount] = fullMat[r + c*numRows];
				nonzeroCount++;
			}
		}
	}				

}

void LPGSparseMatrix::PrintMatrix() {

	for (int r = 0; r < m; r++) {
		for (int c = 0; c < n; c++) {
			bool foundVal = false;
			for (int z = 0; z < nzeros[c]; z++) {
				if (indices[c][z] == r) {
					foundVal = true;
					printf("%5.1f ", values[c][z]);
				}
			}
			if (!foundVal) printf("%5.1f ", 0);
		}
		printf("\n");
	}
}