// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstdio>
#include <cstdint>
#include <random>
#include <omp.h>

#include "utils.h"
#include "timer.h"
#include "split_file.h"

// OpenMP 3.0 supports unsigned for-loop iterator, so can replace std::int64_t by size_t
using std::int64_t;

size_t split_datafile(const std::string dataname, const unsigned int n)
{
	ifstream ifs(dataname);
	if (!ifs.is_open()) {
		throw runtime_error("Data file " + dataname + " cannot be opened.");
	}

	size_t buffer_size = 256 * 1024;
	struct FileBuffer {
		char buf[256 * 1024];
	};
	std::vector<FileBuffer> buffer(n);

	std::vector<ofstream> ofs(n);
	for (int i = 0; i < n; i++) {
		ofs[i].rdbuf()->pubsetbuf(buffer[i].buf, buffer_size);
		std::string filename = dataname + std::to_string(n) + '_' + std::to_string(i + 1);
		ofs[i].open(filename);
	}

	// uniform random generator
	std::random_device rdv;
	std::mt19937_64 rng(rdv());		
	std::uniform_int_distribution<size_t> unif(0, n - 1);

	std::string line;
	size_t count = 0;
	while (std::getline(ifs, line)) {
		int i = unif(rng);
		//int i = count % n;
		ofs[i] << line << '\n';
		count++;
	}

	// close all files
	ifs.close();
	for (int i = 0; i < n; i++) {
		ofs[i].close();
	}

	return count;
}

size_t split_data_sorted(const std::string dataname, const unsigned int n)
{
	ifstream ifs(dataname);
	if (!ifs.is_open()) {
		throw runtime_error("Data file " + dataname + " cannot be opened.");
	}

	// read lines and sort lexicographically
	std::vector<std::string> alllines;

	std::string line;
	size_t count = 0;
	while (std::getline(ifs, line)) {
		alllines.push_back(line.substr(0,2));
		count++;
	}
	ifs.close();

	std::vector<size_t> indices(count);
	std::vector<size_t> file_id(count, 0);

	std::iota(indices.begin(), indices.end(), 0);

	//std::sort(alllines.begin(), alllines.end());
	//std::sort(alllines.begin(), alllines.end(), [](string s1, string s2) {return (s1[0] > s2[0]); });
	std::sort(indices.begin(), indices.end(), [&alllines](size_t i1, size_t i2) {return (alllines[i1] > alllines[i2]); });

	size_t quotient = count / n;
	for (size_t i = 0; i < n*quotient; i++) {
		file_id[indices[i]] = i / quotient;
	}

	// reopen the input file and then write to output files
	size_t buffer_size = 256 * 1024;
	struct FileBuffer {
		char buf[256 * 1024];
	};
	std::vector<FileBuffer> buffer(n);

	std::vector<ofstream> ofs(n);
	for (int i = 0; i < n; i++) {
		ofs[i].rdbuf()->pubsetbuf(buffer[i].buf, buffer_size);
		std::string filename = dataname + std::to_string(n) + "_s0_" + std::to_string(i + 1);
		ofs[i].open(filename);
	}

	ifs.open(dataname);
	count = 0;
	while (std::getline(ifs, line)) {
		ofs[file_id[count]] << line << '\n';
		count++;
	}

	// close all files
	ifs.close();
	for (int i = 0; i < n; i++) {
		ofs[i].close();
	}

	return count;
}