// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// SplitData.cpp : Defines the entry point for the console application.
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <omp.h>

#include "timer.h"
#include "utils.h"
#include "functions.h"
#include "split_file.h"

using namespace functions;

int main(int argc, char* argv[])
{
	if (argc < 3) {
		std::cout << "Need to specify input file name and number of files to split." << std::endl;
		std::exit(0);
	}
	std::string filename(argv[1]);
	std::string numsplit(argv[2]);
	int n = std::stoi(numsplit);
	char sorted = 'n';
	if (argc >=4 ){
		sorted = argv[3][0];
	}

	// record time for reading file and writing files
	HighResTimer timer;
	timer.start();

	size_t N;
	if (sorted == 's' || sorted == 'S') {
		N = split_data_sorted(filename, n);
	}
	else {
		N = split_datafile(filename, n);
	}

	std::cout << "Read and wrote " << N << " lines in " << timer.seconds_from_start() << " seconds." << std::endl;

	return 0;
}
