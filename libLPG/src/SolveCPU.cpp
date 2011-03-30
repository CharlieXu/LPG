///////////////////////////////////////////////////////////////////////////////
// LPG - libLPG																 //
// Implements RSM for the solution of LPs									 //
// ------------------------------------------------------------------------- //
//																			 //
// SolveCPU.cpp															     //
// Solve LPs on the CPU                              						 //
//																			 //
// (c) Iain Dunning 2011													 //
///////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
// Interesting things to search for
// ###ERR		Error checking code
// ###DEBUG		Uncomment this for some (maybe) useful debug code
// ###TODO		...
//-----------------------------------------------------------------------------

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
// SolveLP_C
// Given a problem in SCF, solve using RSM BOUNDED
void LPG::SolveCPU()
{
	
	//-------------------------------------------------------------------------
	// 1.	GENERAL SETUP
	size_t m_floats  = m   * sizeof(double);
	size_t n_floats  = n   * sizeof(double);
	size_t mm_floats = m*m * sizeof(double);
	size_t nm_floats = n*m * sizeof(double);
	
	const int BASIC			=  0;
	const int NONBASIC_L	= +1;
	const int NONBASIC_U	= -1;

	const int PRINT_ITER_EVERY = 100;

	//-------------------------------------------------------------------------
	// 2.	ALLOCATE MEMORY
	// 2.1	Basis
	int*	varStatus	= (int*)	malloc(sizeof(int)*(n+m)	);
	int*	basicVars	= (int*)	malloc(sizeof(int)*m		);
	double*	Binv		= (double*)	malloc(mm_floats			);
	double* cBT			= (double*) malloc(m_floats				);
	// 2.2	General
	//double*	x			= (double*)	malloc(sizeof(double)*(n+m)	);
	double* pi			= (double*) malloc(m_floats				);
	double* rc			= (double*) malloc(n_floats				);
	double* BinvAs		= (double*) malloc(m_floats				);

	//-------------------------------------------------------------------------
	// 3.	INITIALISE MEMORY
	// 3.1	Initial values of variables
	// 3.1.1	Real variables
	for (int i = 0; i < n; i++) {
		double absLB = fabs(xLB[i]), absUB = fabs(xUB[i]);
		x[i]		 = (absLB < absUB) ? xLB[i]		: xUB[i];
		varStatus[i] = (absLB < absUB) ? NONBASIC_L : NONBASIC_U;
	}
	// 3.1.2	Artificial variables
	for (int i = n; i < n+m; i++) {
		x[i] = b[i-n];
		//for (int i2 = 0; i2 < n; i2++) x[i] -= A[i2 + (i-n)*n]*x[i2];
		// Change A from row-major to col major
		// i is row, i2 is col
		for (int i2 = 0; i2 < n; i2++) x[i] -= A[(i-n) + i2*m]*x[i2];
		assert(x[i] > -LPG_TOL); //###ERR: artificials start positive, drive towards zero
	}

	// 3.2	Basis
	for (int i = 0; i < m; i++)		basicVars[i] = i+n;
	for (int i = n; i < n+m; i++)	varStatus[i] = BASIC;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			Binv[i+j*m] = (i==j) ? 1.0 : 0.0;
		}
	}
	for (int i = 0; i < m; i++) {
		cBT[i] = +1.0;
	}
	
	//-------------------------------------------------------------------------
	// 4.	BEGIN ITERATIONS
	bool phaseOne = true;
	int iteration = 0;
	//###DEBUG: DebugPrint("x[] at start",x,n+m);
	while (true) {
		//---------------------------------------------------------------------
		// Iteration counter
		iteration++;
		if (iteration % PRINT_ITER_EVERY == 0){
			printf("Iteration %d\n", iteration);
			if (phaseOne) {
				double z_one = 0.0;
				for (int i = n; i < n+m; i++) z_one += x[i];
				printf("\t[phase one] z = %.5f\n", z_one);
			} else {
				double z_two = 0.0;
				for (int i = 0; i < n; i++) z_two += x[i]*c[i];
				printf("\t[phase two] z = %.5f\n", z_two);
			}
		}
		//---------------------------------------------------------------------


		//###DEBUG Check bounds on reals - which should never be violated
		//for (int i = 0; i < n;   i++) {
		//	if (x[i] < xLB[i]-LPG_TOL) { printf("A real variable is violating bounds! LB\n"); assert(false); }
		//	if (x[i] > xUB[i]+LPG_TOL) { printf("A real variable is violating bounds! UB\n"); assert(false); }
		//}
		//for (int i = n; i < n+m;   i++) {
		//	if (x[i] < -LPG_TOL ) { printf("An art. variable is violating bounds! LB\n"); assert(false); }
		//	if (x[i] >  LPG_BIG ) { printf("An art. variable is violating bounds! UB\n"); assert(false); }
		//}

		//---------------------------------------------------------------------
		// STEP ONE: DUALS AND REDUCED COSTS
		// piT = cbT Binv
		for (int i = 0; i < m; i++) {
			pi[i] = 0.0;
			for (int j = 0; j < m; j++)	pi[i] += cBT[j] * Binv[i + j*m];
		}
		//###DEBUG: DebugPrint("pi[]",pi,m);

		// P1: rc = 0 - A^T pi
		// P2: rc = c - A^T pi
		for (int i = 0; i < n; i++) {
			rc[i] = phaseOne ? 0.0 : c[i];
			for (int nz = 0; nz < sparseA->nzeros[i]; nz++) 
				rc[i] -= sparseA->values[i][nz] * pi[sparseA->indices[i][nz]];
		}
		//###DEBUG: DebugPrint("rc[]",rc,n);
		//---------------------------------------------------------------------

		//---------------------------------------------------------------------
		// STEP TWO: CHECK OPTIMALITY, PICK EV
		double minRC = -LPG_TOL;
		int s = -1;

		for (int i = 0; i < n; i++) {
			// If NONBASIC_L (= +1), rc[i] must be negative (< 0) -> +rc[i] < -LPG_TOL
			// If NONBASIC_U (= -1), rc[i] must be positive (> 0) -> -rc[i] < -LPG_TOL
			//													  -> +rc[i] > +LPG_TOL
			// If BASIC	(= 0), can't use this rc -> 0 * rc[i] < -LPG_TOL -> alway FALSE
			// Then, by setting initial value of minRC to -LPG_TOL, can collapse this
			// check and the check for a better RC into 1 IF statement!
			if (varStatus[i] * rc[i] < minRC) { minRC = varStatus[i] * rc[i]; s = i; }
		}
		//###DEBUG: printf("minRC = %.5f, s = %d\n", minRC, s);

		if (s == -1) {
			if (phaseOne) {
				printf("\tOptimality in Phase 1!\n");
				z = 0.0;	for (int i = 0; i < m; i++) z += cBT[i] * x[basicVars[i]];
				if (z > LPG_TOL) {
					printf("\tPhase 1 objective: z = %.3f > 0 -> infeasible!\n", z);
					status = LPG_INFEASIBLE;
					break;
				} else {
					printf("\tTransitioning to phase 2\n");
					phaseOne = false;
					for (int i = 0; i < m; i++) {
						cBT[i] = (basicVars[i] < n) ? (c[basicVars[i]]) : (0.0);
					}
					continue;
				}
			} else {
				printf("\tOptimality in Phase 2!\n");
				status = LPG_OPTIMAL;
				z = 0.0;
				for (int i = 0; i < n; i++) {
					z += c[i] * x[i];
				}
				break;
			}
		}
		//---------------------------------------------------------------------

		//---------------------------------------------------------------------
		// STEP THREE: CALCULATE BINVAS
		for (int i = 0; i < m; i++) {
			BinvAs[i] = 0.0;
			// row = j, col = s
			//for (int j = 0; j < m; j++) BinvAs[i] += Binv[j + i*m] * A[s*m + j];
			for (int nz = 0; nz < sparseA->nzeros[s]; nz++) 
				BinvAs[i] += sparseA->values[s][nz] * Binv[sparseA->indices[s][nz] + i*m];
		}
		//###DEBUG: DebugPrint("BinvAs[]", BinvAs, m);
		//---------------------------------------------------------------------

		//---------------------------------------------------------------------
		// STEP FOUR: MIN RATIO TEST
		double minRatio = LPG_BIG, ratio = 0.0;
		int r = -1;
		bool rIsEV = false;
		bool forceOutArtificial = false;
		
		//###TODO: Collapse if statements
		if (varStatus[s] == NONBASIC_L) {
			//###DEBUG: printf("EV is NBL...\n");
			// NBL, -> rc[s] < 0 -> want to INCREASE x[s]
			assert(rc[s] < -LPG_TOL); //###ERR
			// Option 1: Degenerate iteration
			ratio = xUB[s] - xLB[s];
			if (ratio <= minRatio) { minRatio = ratio; r = -1; rIsEV = true; }
			// Option 2: Basic variables
			for (int i = 0; i < m; i++) {
				int j = basicVars[i];
				if (j >= n) {
					if (BinvAs[i] < -LPG_TOL) { // BinvAs[i] < 0
						ratio = (LPG_BIG - x[j]) / (-BinvAs[i]);
						if (ratio <= minRatio) { minRatio = ratio; r = i; rIsEV = false; }
					}
					if (BinvAs[i] > +LPG_TOL) { // BinvAs[i] > 0
						ratio = (x[j] - 0.0000) / (+BinvAs[i]);
						if (ratio <= minRatio) { minRatio = ratio; r = i; rIsEV = false; }
					}
				} else {
					if (BinvAs[i] < -LPG_TOL) { // BinvAs[i] < 0
						ratio = (xUB[j] - x[j]) / (-BinvAs[i]);
						if (ratio <= minRatio) 
							{ minRatio = ratio; r = i; rIsEV = false; }
					}
					if (BinvAs[i] > +LPG_TOL) { // BinvAs[i] > 0
						ratio = (x[j] - xLB[j]) / (+BinvAs[i]);
						if (ratio <= minRatio) 
							{ minRatio = ratio; r = i; rIsEV = false; }
					}
				}
				assert(minRatio > -LPG_TOL);
			}
			//###DEBUG: printf("minRatio = %.5f, r = %d, rIsEV = %d\n", minRatio, r, rIsEV);
			
		}
		if (varStatus[s] == NONBASIC_U) { 
			//###DEBUG: printf("EV is NBU...\n");
			// NBU, -> rc[s] > 0 -> want to DECREASE x[s]
			assert(rc[s] > +LPG_TOL); //###ERR
			// Option 1: Degenerate iteration
			ratio = xUB[s] - xLB[s];
			if (ratio <= minRatio) { minRatio = ratio; r = -1; rIsEV = true; }
			// Option 2: Basic variables
			for (int i = 0; i < m; i++) {
				int j = basicVars[i];
				if (j >= n) {
					if (BinvAs[i] > +LPG_TOL) { // BinvAs[i] > 0
						ratio = (LPG_BIG - x[j]) / (+BinvAs[i]);
						if (ratio <= minRatio) { minRatio = ratio; r = i; rIsEV = false; }
					}
					if (BinvAs[i] < -LPG_TOL) { // BinvAs[i] < 0
						ratio = (x[j] - 0.000) / (-BinvAs[i]);
						if (ratio <= minRatio) { minRatio = ratio; r = i; rIsEV = false; }
					}
				} else {
					if (BinvAs[i] > +LPG_TOL) { // BinvAs[i] > 0
						ratio = (xUB[j] - x[j]) / (+BinvAs[i]);
						if (ratio <= minRatio) { minRatio = ratio; r = i; rIsEV = false; }
					}
					if (BinvAs[i] < -LPG_TOL) { // BinvAs[i] < 0
						ratio = (x[j] - xLB[j]) / (-BinvAs[i]);
						if (ratio <= minRatio) { minRatio = ratio; r = i; rIsEV = false; }
					}
				}
				assert(minRatio > -LPG_TOL);
			}
			//###DEBUG: printf("minRatio = %.5f, r = %d, rIsEV = %d\n", minRatio, r, rIsEV);
			
		}
		// Check ratio
		if (minRatio >= LPG_BIG) {
			if (phaseOne) {
				// Not sure what this means - nothing good!
				assert(false);
			} else {
				// PHASE 2: Unbounded!
				status = LPG_UNBOUNDED;
				printf("\tUnbounded in Phase 2!\n");
				break;
			}
		}
		//---------------------------------------------------------------------

		//---------------------------------------------------------------------
		// STEP FIVE: UPDATE SOLUTION AND BASIS
		x[s] += varStatus[s] * minRatio;
		for (int i = 0; i < m; i++) x[basicVars[i]] -= varStatus[s] * minRatio * BinvAs[i];

		/*if (varStatus[s] == NONBASIC_L) {
			x[s] += minRatio;
			for (int i = 0; i < m; i++) x[basicVars[i]] -= minRatio * BinvAs[i];
		}
		if (varStatus[s] == NONBASIC_U) {
			x[s] -= minRatio;
			for (int i = 0; i < m; i++) x[basicVars[i]] += minRatio * BinvAs[i];
		}*/
		//###DEBUG: DebugPrint("x[] updated",x,n+m);
		if (!rIsEV) {
			// Basis change! Update Binv, flags
			assert(r>=0); //###ERR
			assert(r<m); //###ERR
			// RSM tableau: [Binv B | Binv | Binv As]
			// -> GJ pivot on the BinvAs column, rth row
			double erBinvAs = BinvAs[r];
			// All non-r rows
			for (int i = 0; i < m; i++) {
				if (i != r) {
					double eiBinvAsOvererBinvAs = BinvAs[i] / erBinvAs;
					for (int j = 0; j < m; j++) {
						Binv[j+i*m] -= eiBinvAsOvererBinvAs * Binv[j+r*m];
					}
				}
			}
			// rth row
			for (int j = 0; j < m; j++) Binv[j+r*m] /= erBinvAs;

			// Update status flags
			varStatus[s] = BASIC;
			if (basicVars[r] < n) {
				if (fabs(x[basicVars[r]] - xLB[basicVars[r]]) < LPG_TOL) varStatus[basicVars[r]] = NONBASIC_L;
				if (fabs(x[basicVars[r]] - xUB[basicVars[r]]) < LPG_TOL) varStatus[basicVars[r]] = NONBASIC_U;
			} else {
				if (fabs(x[basicVars[r]] - 0.00000) < LPG_TOL) varStatus[basicVars[r]] = NONBASIC_L;
				if (fabs(x[basicVars[r]] - LPG_BIG) < LPG_TOL) varStatus[basicVars[r]] = NONBASIC_U;
			}
			cBT[r] = phaseOne ? 0.0 : c[s];
			basicVars[r] = s;

		} else {
			// Degenerate iteration
			if (varStatus[s] == NONBASIC_L) { varStatus[s] = NONBASIC_U; }
			else { varStatus[s] = NONBASIC_L; }
		}
		//###DEBUG: DebugPrint("Updated basis:",varStatus,n+m);
	}
	
	//-------------------------------------------------------------------------
	// 5.	RELEASE MEMORY
	free(varStatus);
	free(basicVars);
	free(Binv);
	free(cBT);
	//free(x);
	free(pi);
	free(rc);
	free(BinvAs);
}

