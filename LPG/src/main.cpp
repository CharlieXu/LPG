///////////////////////////////////////////////////////////////////////////////
// LPG - LPG																 //
// CLI to the libLPG library												 //
// ------------------------------------------------------------------------- //
//																			 //
// main.cpp																	 //
// Main interface		                              						 //
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
#include <vector>
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// libLPG
#include "libLPG.hpp"
LPG* model = NULL;
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Prototypes
void ProcessConfig();
void DisplayMenu();
void Load(char fileType, const char* providedFilename = "");
void Solve(char type);
void RunTestSuite(bool useGPU = false);
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Configuration
char dataFolderPath[255];
char testcasesFilename[255];
//-----------------------------------------------------------------------------

int main(int argc, char *argv[]) {

	printf("LPG - Command Line Interface V1.0\n");

	ProcessConfig();

	// Are there any arguments?
	if (argc > 1) {

		// COMMAND LINE MODE
		printf("Command line mode\n");
		char command = argv[1][0];
		if		(command == 'M') { Load(command,argv[2]);	} 
		else if (command == 'L') { Load(command,argv[2]);	}
		else					 { return -1;				}

		char solver = argv[3][0];
		if		(solver == 'C')	{ Solve(solver);	}
		else if (solver == 'G')	{ Solve(solver);	}
		else					{ return -1;	}

	} else {

		// INTERACTIVE MODE
		printf("Interactive mode\n");
		
		DisplayMenu();
		
		bool done = false;
		while (!done) {
			char command;
			scanf("%c", &command);
			switch (command) {
				case 'L': Load(command);		break;
				case 'M': Load(command);		break;
				case 'C': Solve(command);		break; // CPU
				case 'G': Solve(command);		break; // GPU
				case 'T': RunTestSuite(false);	break; // CPU
				case 'Y': RunTestSuite(true);	break; // GPU
				case 'X': done = true;			break;
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
void Load(char command, const char* providedFilename)
{
	if (model != NULL) delete model;

	char filename[255];
	// Filename is optional - will scanf if not there
	if (strlen(providedFilename) > 0) {
		strcpy(filename, providedFilename);
	} else {
		scanf("%s",filename);
	}

	char filepathAndName[255] = "";
	strcat(filepathAndName, dataFolderPath);
	strcat(filepathAndName, filename);

	model = new LPG();
	if (command == 'L') model->LoadLP (filepathAndName);
	if (command == 'M') model->LoadMPS(filepathAndName);
}

//-----------------------------------------------------------------------------
void RunTestSuite(bool useGPU) {
	
	// Create a LPG instance
	model = new LPG(useGPU, false);

	// Load test data
	std::vector<std::string> fileNames;
	std::vector<double> results;
	std::vector<double> expected;
	std::vector<double> times;
	std::vector<int> problemSizeM;

	int numCases = 0;
	std::ifstream testData(testcasesFilename);

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
		std::string fullFileName(fileNames[i]);
		fullFileName.insert(0,dataFolderPath);
		model->LoadMPS(fullFileName.c_str());
		problemSizeM.push_back(model->m);
		clock_t timePreSolve = clock();
		if (!useGPU) model->SolveCPU();
		if ( useGPU) model->SolveGPU();
		clock_t timePostSolve = clock();
		times.push_back((timePostSolve-timePreSolve)/(CLOCKS_PER_SEC*1.0));
		results.push_back(model->z);
	}
	
	// Results table
	printf("RESULTS\n");
	printf("%20s  %10s  %10s  %7s  %6s\n", "Name", "Result", "Expected", "Time", "NumRow");
	for (int i = 0; i < numCases; i++) {
		printf("%20s  %10.3f  %10.3f  %7.3f  %6d\n", fileNames[i].c_str(), results[i], expected[i], times[i], problemSizeM[i]);
	}

	// Delete LPG instance
	delete model;
}


//-----------------------------------------------------------------------------
void Solve(char type) 
{
	// GPU-only: prepare GPU (may be done already)
	if (type == 'G') model->InitGPU();
	if (type == 'G') model->InitKernels();

	// Solve it
	clock_t timePreSolve = clock();
	if (type == 'C') model->SolveCPU();
	if (type == 'G') model->SolveGPU();
	clock_t timePostSolve = clock();
	printf("Solve took %f s\n", (timePostSolve-timePreSolve)/(CLOCKS_PER_SEC*1.0));

	// Display results
	printf("Objective function value: z = %f\n", model->z);
	//printf("Non-zero variables:\n");
	//for (int i = 0; i < n; i++) if (fabs(x[i]) > LPG_TOL) printf("x[%d] = %f\n", i, x[i]);
}
//-----------------------------------------------------------------------------


void ProcessConfig() {
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
}