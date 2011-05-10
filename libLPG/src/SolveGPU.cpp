///////////////////////////////////////////////////////////////////////////////
// LPG - libLPG																 //
// Implements RSM for the solution of LPs									 //
// ------------------------------------------------------------------------- //
//																			 //
// SolveGPU.cpp															     //
// Solve LPs on the GPGPU                              						 //
//																			 //
// (c) Iain Dunning 2011													 //
///////////////////////////////////////////////////////////////////////////////


//*****************************************************************************
// Broken until further notice!
// Due to:
//	- Column-wise A
//*****************************************************************************

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

	// Get platform and device information
	cl_int ret;
	ret = clGetPlatformIDs(1, &platformID, &numPlatforms); 
	CL_ERR_CHECK
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
	LoadKernel("C:/Users/Iain/LPG/libLPG/kernel/dual_kernel.cl",	"dual",		dual_program,		dual_kernel		);
	LoadKernel("C:/Users/Iain/LPG/libLPG/kernel/rc1_kernel.cl",		"rc1",		rc1_program,		rc1_kernel		);
	LoadKernel("C:/Users/Iain/LPG/libLPG/kernel/rc2_kernel.cl",		"rc2",		rc1_program,		rc2_kernel		);
	LoadKernel("C:/Users/Iain/LPG/libLPG/kernel/binvas_kernel.cl",	"binvas",	binvas_program,		binvas_kernel	);
	LoadKernel("C:/Users/Iain/LPG/libLPG/kernel/tableau1_kernel.cl","tableau1",	tableau1_program,	tableau1_kernel	);
	LoadKernel("C:/Users/Iain/LPG/libLPG/kernel/tableau2_kernel.cl","tableau2",	tableau2_program,	tableau2_kernel	);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// SolveLP_G
// Given a problem in SCF, solve using RSM BOUNDED
void LPG::SolveGPU()
{
	//-------------------------------------------------------------------------
	// 1.	GENERAL SETUP
	// 1.1	Constants
	size_t m_floats  = m   * sizeof(SCALAR);
	size_t n_floats  = n   * sizeof(SCALAR);
	size_t mm_floats = m*m * sizeof(SCALAR);
	size_t nm_floats = n*m * sizeof(SCALAR);

	const int BASIC			=  0;
	const int NONBASIC_L	= +1;
	const int NONBASIC_U	= -1;

	cl_int ret = 0;
	const int PRINT_ITER_EVERY = 100;

	// 1.2	Workgroup sizes for dual, rc, binvas, tableau
	size_t problemSize, localSize;
	ret = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &localSize, NULL);
	size_t bestLocalSize = (localSize < 32) ? localSize : 32;
	size_t gpu_m, gpu_n;
	if (m % bestLocalSize == 0) { gpu_m = m; }
	else { gpu_m = m + (bestLocalSize - (m % bestLocalSize)); }
	if (n % bestLocalSize == 0) { gpu_n = n; }
	else { gpu_n = n + (bestLocalSize - (n % bestLocalSize)); }

	//-------------------------------------------------------------------------
	// 2.		ALLOCATE MEMORY
	// 2.1		In main system RAM
	int*	varStatus	= (int*)	malloc(sizeof(int)*(n+m)		);
	int*	basicVars	= (int*)	malloc(sizeof(int)*m			);
	SCALAR*	Binv		= (SCALAR*)	malloc(mm_floats				);
	SCALAR* cBT			= (SCALAR*) malloc(m_floats					);
	SCALAR*	x			= (SCALAR*)	malloc(sizeof(SCALAR)*(n+m)		);
	SCALAR* pi			= (SCALAR*) malloc(m_floats					);
	SCALAR* rc			= (SCALAR*) malloc(n_floats					);
	SCALAR* BinvAs		= (SCALAR*) malloc(m_floats					);

	// 2.2		On the GPU
	cl_mem cBT_gpu		= clCreateBuffer(context, CL_MEM_READ_WRITE,  m_floats, NULL, &ret); CL_ERR_CHECK
	cl_mem Binv_gpu		= clCreateBuffer(context, CL_MEM_READ_WRITE, mm_floats, NULL, &ret); CL_ERR_CHECK
	cl_mem pi_gpu		= clCreateBuffer(context, CL_MEM_READ_WRITE,  m_floats, NULL, &ret); CL_ERR_CHECK
	
	cl_mem A_gpu		= clCreateBuffer(context, CL_MEM_READ_WRITE, nm_floats, NULL, &ret); CL_ERR_CHECK
	cl_mem c_gpu		= clCreateBuffer(context, CL_MEM_READ_WRITE,  n_floats, NULL, &ret); CL_ERR_CHECK 
	cl_mem rc_gpu		= clCreateBuffer(context, CL_MEM_READ_WRITE,  n_floats, NULL, &ret); CL_ERR_CHECK

	cl_mem BinvAs_gpu	= clCreateBuffer(context, CL_MEM_READ_WRITE,  m_floats, NULL, &ret); CL_ERR_CHECK
	
	//-------------------------------------------------------------------------
	// 3.		INITIALISE MEMORY
	// 3.1		Initial values of variables
	// 3.1.1	Real variables
	for (int i = 0; i < n; i++) {
		SCALAR absLB = fabs(xLB[i]), absUB = fabs(xUB[i]);
		x[i]		 = (absLB < absUB) ? xLB[i]		: xUB[i];
		varStatus[i] = (absLB < absUB) ? NONBASIC_L : NONBASIC_U;
	}
	// 3.1.2	Artificial variables
	for (int i = n; i < n+m; i++) {
		x[i] = b[i-n];
		for (int i2 = 0; i2 < n; i2++) x[i] -= A[i2 + (i-n)*n]*x[i2];
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
	for (int i = n; i < n+m; i++)	cBT[i-n] = +1.0;

	// 3.3	Kernels
	// 3.3.1	dual_kernel - computes piT = cBT * Binv
	//			SCALAR *cBT, SCALAR *Binv, SCALAR *piT, int m
	ret = clSetKernelArg(dual_kernel,	0, sizeof(cl_mem),	(void *)&cBT_gpu);	CL_ERR_CHECK
	ret = clSetKernelArg(dual_kernel,	1, sizeof(cl_mem),	(void *)&Binv_gpu); CL_ERR_CHECK
	ret = clSetKernelArg(dual_kernel,	2, sizeof(cl_mem),	(void *)&pi_gpu);	CL_ERR_CHECK
	ret = clSetKernelArg(dual_kernel,	3, sizeof(int),		(void *)&m);		CL_ERR_CHECK

	// 3.3.2	rc1_kernel - computes rcT = 0 - piT*A
	//			SCALAR *rc, SCALAR *piT, SCALAR *A, int m, int n
	ret = clSetKernelArg(rc1_kernel,	0, sizeof(cl_mem),	(void *)&rc_gpu);	CL_ERR_CHECK
	ret = clSetKernelArg(rc1_kernel,	1, sizeof(cl_mem),	(void *)&pi_gpu);	CL_ERR_CHECK
	ret = clSetKernelArg(rc1_kernel,	2, sizeof(cl_mem),	(void *)&A_gpu);	CL_ERR_CHECK
	ret = clSetKernelArg(rc1_kernel,	3, sizeof(int),		(void *)&m);		CL_ERR_CHECK
	ret = clSetKernelArg(rc1_kernel,	4, sizeof(int),		(void *)&n);		CL_ERR_CHECK

	// 3.3.3	rc2_kernel - computes rcT = cT - piT*A
	//			SCALAR *rc, SCALAR *c, SCALAR *piT, SCALAR *A, int m, int n
	ret = clSetKernelArg(rc2_kernel,	0, sizeof(cl_mem),	(void *)&rc_gpu);	CL_ERR_CHECK
	ret = clSetKernelArg(rc2_kernel,	1, sizeof(cl_mem),	(void *)&c_gpu);	CL_ERR_CHECK
	ret = clSetKernelArg(rc2_kernel,	2, sizeof(cl_mem),	(void *)&pi_gpu);	CL_ERR_CHECK
	ret = clSetKernelArg(rc2_kernel,	3, sizeof(cl_mem),	(void *)&A_gpu);	CL_ERR_CHECK
	ret = clSetKernelArg(rc2_kernel,	4, sizeof(int),		(void *)&m);		CL_ERR_CHECK
	ret = clSetKernelArg(rc2_kernel,	5, sizeof(int),		(void *)&n);		CL_ERR_CHECK

	// 3.3.5  binvas_kernel - computes BinvAs = Binv * As
	//	      SCALAR *BinvAs, SCALAR *Binv, SCALAR *A, int m, int n, int s
	ret = clSetKernelArg(binvas_kernel, 0, sizeof(cl_mem), (void *)&BinvAs_gpu); CL_ERR_CHECK
	ret = clSetKernelArg(binvas_kernel, 1, sizeof(cl_mem), (void *)&Binv_gpu);	CL_ERR_CHECK
	ret = clSetKernelArg(binvas_kernel, 2, sizeof(cl_mem), (void *)&A_gpu);		CL_ERR_CHECK
	ret = clSetKernelArg(binvas_kernel, 3, sizeof(int),	   (void *)&m);			CL_ERR_CHECK
	ret = clSetKernelArg(binvas_kernel, 4, sizeof(int),	   (void *)&n);			CL_ERR_CHECK
	// s done later

	// 3.3.6  tableau1_kernel - update the basis, all rows except r
	//	      SCALAR *Binv, SCALAR *BinvAs, int m, int r
	ret = clSetKernelArg(tableau1_kernel, 0, sizeof(cl_mem), (void *)&Binv_gpu); CL_ERR_CHECK
	ret = clSetKernelArg(tableau1_kernel, 1, sizeof(cl_mem), (void *)&BinvAs_gpu); CL_ERR_CHECK
	ret = clSetKernelArg(tableau1_kernel, 2, sizeof(cl_mem), (void *)&m); CL_ERR_CHECK
	// r done later
	
	// 3.3.7  tableau2_kernel - update the basis, only r
	//	      SCALAR *Binv, SCALAR *BinvAs, int m, int r
	ret = clSetKernelArg(tableau2_kernel, 0, sizeof(cl_mem), (void *)&Binv_gpu); CL_ERR_CHECK
	ret = clSetKernelArg(tableau2_kernel, 1, sizeof(cl_mem), (void *)&BinvAs_gpu); CL_ERR_CHECK
	ret = clSetKernelArg(tableau2_kernel, 2, sizeof(cl_mem), (void *)&m); CL_ERR_CHECK
	// r done later
	
	// 3.4	Memory on GPU
	ret = clEnqueueWriteBuffer(commandQueue, A_gpu,		CL_TRUE, 0,			nm_floats,	A,			0, NULL, NULL); CL_ERR_CHECK
	ret = clEnqueueWriteBuffer(commandQueue, c_gpu,		CL_TRUE, 0,			 n_floats,	c,			0, NULL, NULL); CL_ERR_CHECK
	ret = clEnqueueWriteBuffer(commandQueue, Binv_gpu,	CL_TRUE, 0,		    mm_floats,	Binv,		0, NULL, NULL); CL_ERR_CHECK
	ret = clEnqueueWriteBuffer(commandQueue, cBT_gpu,	CL_TRUE, 0,			 m_floats,	cBT,		0, NULL, NULL); CL_ERR_CHECK

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
				SCALAR z_one = 0.0;
				for (int i = n; i < n+m; i++) z_one += x[i];
				printf("\t[phase one] z = %.5f\n", z_one);
			} else {
				SCALAR z_two = 0.0;
				for (int i = 0; i < n; i++) z_two += x[i]*c[i];
				printf("\t[phase two] z = %.5f\n", z_two);
			}
		}
		//---------------------------------------------------------------------

		//---------------------------------------------------------------------
		// STEP ONE: DUALS AND REDUCED COSTS
		// piT = cbT Binv
		problemSize = gpu_m;
		localSize = bestLocalSize;
		ret = clEnqueueNDRangeKernel(commandQueue, dual_kernel, 1, NULL, &problemSize, &localSize, 0, NULL, NULL); CL_ERR_CHECK

		problemSize = gpu_n;
		localSize = bestLocalSize;
		// P1: rc = 0 - A^T pi
		if ( phaseOne) ret = clEnqueueNDRangeKernel(commandQueue, rc1_kernel, 1, NULL, &problemSize, &localSize, 0, NULL, NULL); CL_ERR_CHECK
		// P2: rc = c - A^T pi
		if (!phaseOne) ret = clEnqueueNDRangeKernel(commandQueue, rc2_kernel, 1, NULL, &problemSize, &localSize, 0, NULL, NULL); CL_ERR_CHECK
		// Pull back from GPU
		//###TODO: Don't do this!
		ret = clEnqueueReadBuffer(commandQueue, rc_gpu, CL_TRUE, 0, n_floats, rc, 0, NULL, NULL); CL_ERR_CHECK
		//###DEBUG: DebugPrint("rc[]",rc,n); 
		//---------------------------------------------------------------------


		//---------------------------------------------------------------------
		// STEP TWO: CHECK OPTIMALITY, PICK EV
		SCALAR minRC = -LPG_TOL;
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
				z = 0;	for (int i = 0; i < m; i++) z += cBT[i] * x[basicVars[i]];
				if (z > LPG_TOL) {
					printf("\tPhase 1 objective: z = %.3f > 0 -> infeasible!\n", z);
					status = LPG_INFEASIBLE;
					break;
				} else {
					printf("\tTransitioning to phase 2\n");
					phaseOne = false;
					for (int i = 0; i < m; i++) {
						cBT[i] = (basicVars[i] < n) ? (c[basicVars[i]]) : (0);
					}
					ret = clEnqueueWriteBuffer(commandQueue, cBT_gpu, CL_TRUE, 0, m_floats, cBT, 0, NULL, NULL); CL_ERR_CHECK
					continue;
				}
			} else {
				printf("\tOptimality in Phase 2!\n");
				status = LPG_OPTIMAL;
				z = 0;
				for (int i = 0; i < n; i++) {
					//x_ans[i] = x[i];
					z += c[i] * x[i];
				}
				break;
			}
		}
		//---------------------------------------------------------------------

		//---------------------------------------------------------------------
		// STEP THREE: CALCULATE BINVAS
		problemSize = gpu_m;
		localSize = bestLocalSize;
		ret = clSetKernelArg(binvas_kernel, 5, sizeof(int), (void *)&s);
		ret = clEnqueueNDRangeKernel(commandQueue, binvas_kernel, 1, NULL, &problemSize, &localSize, 0, NULL, NULL); CL_ERR_CHECK
		// Pull back from GPU
		//###TODO: Don't do this!
		ret = clEnqueueReadBuffer(commandQueue, BinvAs_gpu, CL_TRUE, 0, m_floats, BinvAs, 0, NULL, NULL); CL_ERR_CHECK
		//###DEBUG: DebugPrint("BinvAs",BinvAs,m);
		//---------------------------------------------------------------------

		//---------------------------------------------------------------------
		// STEP FOUR: MIN RATIO TEST
		SCALAR minRatio = LPG_BIG, ratio = 0.0;
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
				assert(minRatio > -LPG_TOL); //###ERR
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
				assert(minRatio > -LPG_TOL); //###ERR
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

			problemSize = gpu_m;
			localSize = bestLocalSize;
			ret = clSetKernelArg(tableau1_kernel, 3, sizeof(int), (void *)&r);
			ret = clEnqueueNDRangeKernel(commandQueue, tableau1_kernel, 1, NULL, &problemSize, &localSize, 0, NULL, NULL); CL_ERR_CHECK

			problemSize = 1;
			localSize = 1;
			ret = clSetKernelArg(tableau2_kernel, 3, sizeof(int), (void *)&r);
			ret = clEnqueueNDRangeKernel(commandQueue, tableau2_kernel, 1, NULL, &problemSize, &localSize, 0, NULL, NULL); CL_ERR_CHECK

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

			// Push cBT
			//###TODO: Don't do this!
			ret = clEnqueueWriteBuffer(commandQueue, cBT_gpu, CL_TRUE, 0, m_floats, cBT, 0, NULL, NULL); CL_ERR_CHECK
		} else {
			// Degenerate iteration
			if (varStatus[s] == NONBASIC_L) { varStatus[s] = NONBASIC_U; }
			else { varStatus[s] = NONBASIC_L; }
		}
		//###DEBUG DebugPrint("Updated basis:",varStatus,n+m);
	}
	
	//-------------------------------------------------------------------------
	// 5.	RELEASE MEMORY
	free(varStatus);	free(basicVars);	free(Binv);
	free(cBT);			free(x);			free(pi);
	free(rc);			free(BinvAs);
	ret = clReleaseMemObject(cBT_gpu);		CL_ERR_CHECK
	ret = clReleaseMemObject(Binv_gpu);		CL_ERR_CHECK
	ret = clReleaseMemObject(pi_gpu);		CL_ERR_CHECK
	ret = clReleaseMemObject(A_gpu);		CL_ERR_CHECK
	ret = clReleaseMemObject(c_gpu);		CL_ERR_CHECK
	ret = clReleaseMemObject(rc_gpu);		CL_ERR_CHECK
	ret = clReleaseMemObject(BinvAs_gpu);	CL_ERR_CHECK
}
//-----------------------------------------------------------------------------

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