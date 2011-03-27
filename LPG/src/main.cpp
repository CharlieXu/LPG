///////////////////////////////////////////////////////////////////////////////
// LPG - LPG																 //
// CLI to the libLPG library												 //
// ------------------------------------------------------------------------- //
//																			 //
// main.cpp																	 //
// Everything!			                              						 //
//																			 //
// (c) Iain Dunning 2011													 //
///////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
// Standard Lib Includes
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <fstream>
using namespace std;
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// CoinUtils
#include "CoinMpsIO.hpp"
#include "CoinLpIO.hpp"
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// libLPG
#include "libLPG.hpp"
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// The problem
int m, n;
CoinPackedMatrix* sparseA;
double *A, *b, *c;
double *xLB, *xUB;
// The solution
double* x;
double z;
int status;
// Flag
bool isLoaded = false;
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Prototypes
void DisplayMenu();
void LoadFromMPS(const char* providedFilename = "", bool silentMode = false);
void LoadFromLP (const char* providedFilename = "", bool silentMode = false);
void SolveCPU();
void SolveGPU();
void FreeIfNeeded();
void RunTestSuite(bool useGPU = false);
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Configuration
char dataFolderPath[255];
char testcasesFilename[255];

//-----------------------------------------------------------------------------
int main(int argc, char *argv[]) {

	printf("LPG - Command Line Interface\n");
	
	FILE* readConfigFile = NULL;
	readConfigFile = fopen("config.txt", "r");
	if (readConfigFile != NULL) {
		printf("Reading configuration file...\n");
		// LINE 1 - dataFolderPath
		fgets(dataFolderPath, 255, readConfigFile);
		dataFolderPath[strlen(dataFolderPath) - 1] = '\0';
		printf("Will look for .LP/.MPS files in path: %s\n", dataFolderPath);
		// LINE 2 - testcasesFilename
		fgets(testcasesFilename, 255, readConfigFile);
		testcasesFilename[strlen(testcasesFilename) - 1] = '\0';
		printf("List of problems to solve is in file: %s\n", testcasesFilename);
	} else {
		// File not found, create it with defaults
		printf("Configuration file config.txt was not found!\n");
		printf("Default configuration will be used, and written to config.txt\n");
		FILE* writeConfigFile = NULL;
		writeConfigFile = fopen("config.txt", "w");
		// LINE 1 - dataFolderPath
		fprintf(writeConfigFile, "data/\n");
		sprintf(dataFolderPath, "data/");
		printf("Will look for .LP/.MPS files in path: %s\n", dataFolderPath);
		// LINE 2 - testcasesFilename
		fprintf(writeConfigFile, "testcases.txt\n");
		sprintf(testcasesFilename, "testcases.txt");
		printf("List of problems to solve is in file: %s\n", testcasesFilename);
		// EOF
		fprintf(writeConfigFile,"#");
		fclose(writeConfigFile);
	}


	// Are there any arguments?
	if (argc > 1) {

		printf("Command line mode\n");

		char command = argv[1][0];
		if		(command == 'M') { LoadFromMPS(argv[2]); } 
		else if (command == 'L') { LoadFromLP(argv[2]);  }
		else					 { return -1;			 }

		char solver = argv[3][0];
		if		(solver == 'C') { SolveCPU(); }
		else if (solver == 'G') { SolveGPU(); }
		else					{ return -1;  }

	} else {

		printf("Interactive mode\n");
		DisplayMenu();
		// Wait for commands
		bool done = false;
		while (!done) {
			char command;
			scanf("%c", &command);
			switch (command) {
				// L - load LP file
				case 'L': LoadFromLP();		break;

				// M - load MPS file
				case 'M': LoadFromMPS();	break;

				// C - solve problem with CPU
				case 'C': SolveCPU();		break;

				// G - solve problem with GPU
				case 'G': SolveGPU();		break;

				// T - run test problems on CPU
				case 'T': RunTestSuite(false);	break;

				// Y - run test problems on GPU
				case 'Y': RunTestSuite(true);	break;

				// X - quit
				case 'X': done = true;		break;

				
			}
		}
	}

	return 0;
}

//-----------------------------------------------------------------------------
void DisplayMenu()
{
	printf("Commands:\n");
	printf("L filename            Load filename, where filename is an LP file.   \n");
	printf("M filename            Load filename, where filename is an MPS file.  \n");
	printf("C                     Solve loaded problem with CPU solver.          \n");
	printf("G                     Solve loaded problem with GPU solver.          \n");
	printf("T                     Run through test suite with CPU solver.        \n");
	printf("Y                     Run through test suite with GPU solver.        \n");
	printf("X                     Quit.                                          \n");
	printf("---------------------------------------------------------------------\n");
}

//-----------------------------------------------------------------------------
void LoadFromMPS(const char* providedFilename, bool silentMode)
{
	FreeIfNeeded();

	char filename[100];
	if (strlen(providedFilename) > 0) {
		strcpy(filename, providedFilename);
	} else {
		scanf("%s",filename);
	}
	if (!silentMode) printf("Loading MPS file %s...\n",filename);

	char filepathname[100] = "";
	strcat(filepathname, "C:/Users/Iain/LPG/TestProbs/");
	strcat(filepathname, filename);

	if (!silentMode) printf("LoadFromMPS(filename=%s)\n", filename);
	if (!silentMode) printf("CoinMpsIO...\n");

	CoinMpsIO mpsIO;
	int status = mpsIO.readMps(filepathname);
	if (status == -1) return;

	if (!silentMode) printf("Convert to internal form...\n");
	ConvertFromGeneralFormToInternal(	*mpsIO.getMatrixByCol(),	mpsIO.getColLower(), mpsIO.getColUpper(),
										mpsIO.getObjCoefficients(),	mpsIO.getRowLower(), mpsIO.getRowUpper(),
										m, n, sparseA, A, b, c, xLB, xUB,
										silentMode);
	x = (double*)malloc(sizeof(double)*n);
	isLoaded = true;
	z = 0.0;
	status = LPG_UNKNOWN;
	if (!silentMode) printf("Finished loading!\n");
}
//-----------------------------------------------------------------------------
void LoadFromLP(const char* providedFilename, bool silentMode)
{
	FreeIfNeeded();

	char filename[100];
	if (strlen(providedFilename) > 0) {
		strcpy(filename, providedFilename);
	} else {
		scanf("%s",filename);
	}
	if (!silentMode) printf("Loading LP file %s...\n",filename);


	char filepathname[100] = "";
	strcat(filepathname, "C:/Users/Iain/LPG/TestProbs/");
	strcat(filepathname, filename);

	if (!silentMode) printf("LoadFromLP(filename=%s)\n", filename);
	if (!silentMode) printf("CoinLpIO...\n");
	CoinLpIO lpIO;
	lpIO.readLp(filepathname);

	if (!silentMode) printf("Convert to internal form...\n");
	ConvertFromGeneralFormToInternal(	*lpIO.getMatrixByCol(),		lpIO.getColLower(), lpIO.getColUpper(),
										lpIO.getObjCoefficients(),	lpIO.getRowLower(), lpIO.getRowUpper(),
										m, n, sparseA, A, b, c, xLB, xUB);
	x = (double*)malloc(sizeof(double)*n);
	isLoaded = true;
	z = 0.0;
	status = LPG_UNKNOWN;
	if (!silentMode) printf("Finished loading!\n");
}
//-----------------------------------------------------------------------------
void FreeIfNeeded() {
	if (isLoaded) {
		free(A);
		free(b);
		free(c);
		free(xLB);
		free(xUB);
		free(x);
		isLoaded = false;
	}
}
//-----------------------------------------------------------------------------
void RunTestSuite(bool useGPU) {

	// Prepare for GPUing...
	if (useGPU) {
		printf("RunTestSuite is intialising GPU and kernels...\n");
		InitGPU();
		InitKernels();
		printf("Done!\n");
	}

	// Load test data
	std::vector<std::string> fileNames;
	std::vector<double> results;
	std::vector<double> expected;
	std::vector<double> times;
	std::vector<int> problemSizeM;

	int numCases = 0;
	std::ifstream testData("testcases.txt");

	testData >> numCases;
	for (int i = 0; i < numCases; i++) {
		std::string filename;
		double expResult;
		testData >> filename >> expResult;
		fileNames.push_back(filename);
		expected.push_back(expResult);
	}
	testData.close();

	// Run tests
	for (int i = 0; i < numCases; i++) {
		LoadFromMPS(fileNames[i].c_str(), true);
		problemSizeM.push_back(m);
		clock_t timePreSolve = clock();
		if (!useGPU) SolveLP_C(m,n,A,b,c,xLB,xUB,z,x,status);
		if ( useGPU) SolveLP_G(m,n,A,b,c,xLB,xUB,z,x,status);
		clock_t timePostSolve = clock();
		times.push_back((timePostSolve-timePreSolve)/(CLOCKS_PER_SEC*1.0));
		results.push_back(z);
	}
	
	// Results table
	printf("RESULTS\n");
	printf("%15s  %10s  %10s  %7s  %6s\n", "Name", "Result", "Expected", "Time", "NumRow");
	for (int i = 0; i < numCases; i++) {
		printf("%15s  %10.3f  %10.3f  %7.3f  %6d\n", fileNames[i].c_str(), results[i], expected[i], times[i], problemSizeM[i]);
	}

	// Finish up GPU
	if (useGPU) {
		printf("RunTestSuite is shutting down GPU...\n");
		FreeGPUandKernels();
		printf("Done!\n");
	}
}


//-----------------------------------------------------------------------------
void SolveCPU() 
{
	// Solve it
	clock_t timePreSolve = clock();
	SolveLP_C(m,n,A,b,c,xLB,xUB,z,x,status);
	clock_t timePostSolve = clock();
	printf("Time for SolveLP_C was %f seconds\n", (timePostSolve-timePreSolve)/(CLOCKS_PER_SEC*1.0));

	// Display results
	printf("Objective function value: z = %f\n", z);
	//printf("Non-zero variables:\n");
	//for (int i = 0; i < n; i++) if (fabs(x[i]) > LPG_TOL) printf("x[%d] = %f\n", i, x[i]);
}
//-----------------------------------------------------------------------------
void SolveGPU()
{
	// Get ready
	clock_t timePreInit = clock();
	InitGPU();
	InitKernels();
	clock_t timePostInit = clock();
	printf("Time for InitGPU and InitKernels was %f seconds\n", (timePostInit-timePreInit)/(CLOCKS_PER_SEC*1.0));

	// Solve it
	clock_t timePreSolve = clock();
	SolveLP_G(m,n,A,b,c,xLB,xUB,z,x,status);
	clock_t timePostSolve = clock();
	printf("Time for SolveLP_G was %f seconds\n", (timePostSolve-timePreSolve)/(CLOCKS_PER_SEC*1.0));

	// Display results
	printf("Objective function value: z = %f\n", z);
	//printf("Non-zero variables:\n");
	//for (int i = 0; i < n; i++) if (fabs(x[i]) > LPG_TOL) printf("x[%d] = %f\n", i, x[i]);

	// Shut it down
	clock_t timePreFree = clock();
	FreeGPUandKernels();
	clock_t timePostFree = clock();
	printf("Time for FreeGPUandKernels was %f seconds\n", (timePostFree-timePreFree)/(CLOCKS_PER_SEC*1.0));
}
//-----------------------------------------------------------------------------
