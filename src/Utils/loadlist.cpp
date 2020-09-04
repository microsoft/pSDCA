// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <omp.h>

#include "utils.h"
#include "timer.h"

// OpenMP 3.0 supports unsigned for-loop iterator, so can replace std::int64_t by size_t
using std::int64_t;
using namespace std;

// load datafile with line format: <label>[:<weight>] [<colidx>[:<value>]] ...
template <typename T>
size_t load_datafile(const list<string> filelist, vector<T> &labels, vector<T> &weights,
	vector<T> &values, vector<size_t> &colidx, vector<size_t> &rowptr,
	const bool add_bias, const bool disp_timing, const bool binary, const size_t start_colidx,
	const bool skip_first)
{
	// Work with new vectors and then swap them with output vectors in the end
	vector<T> _labels;
	vector<T> _weights;
	vector<T> _values;
	vector<size_t> _colidx;
	vector<size_t> _rowptr;

	string line;
	T lbl, wgh, val;
	size_t idx, max_idx;
	const char *pStart;
	char *pEnd;

	// record time for reading file and post-processing
	HighResTimer timer;
	timer.start();

	for (auto & filename : filelist)
	{
		ifstream ifs(filename);
		if (!ifs.is_open()) {
			throw runtime_error("Data file " + filename + " cannot be opened.");
		}

		try {
			while (getline(ifs, line))
			{
				// pStart now has a null-character '\0' appended at the end as c_str()
				pStart = line.c_str();

				// first read the label, and possibly a weight for the example
				lbl = (T)strtod(pStart, &pEnd);
				if (*pEnd == ':') {
					wgh = (T)strtod(pEnd + 1, &pEnd);	// pEnd + 1 to skip ':' 
				}
				else {
					wgh = 1.0f;
				}

				if (binary) { lbl = lbl > 0.5 ? 1 : -1; }	// use {+1, -1} for binary labels
				_labels.push_back(lbl);
				_weights.push_back(wgh);
				_rowptr.push_back(_values.size());

				// some file has redundant information about dimensionality of feature vector
				if (skip_first) {
					idx = strtoul(pEnd, &pEnd, 10);	// idx is unsigned long int with base 10
				}

				// read features in example. test for the null character '\0' at the line end 
				max_idx = 0;
				while (*pEnd)	// if not reaching end of line '\0'
				{
					idx = strtoul(pEnd, &pEnd, 10);	// idx is unsigned long int with base 10
					if (*pEnd == ':') {
						val = (T)strtod(pEnd + 1, &pEnd);	// pEnd + 1 to skip ':' 
					}
					else {
						val = 1.0f;
					}
					_values.push_back(val);
					_colidx.push_back(idx);
					max_idx = idx > max_idx ? idx : max_idx;

					// skip white space, only necessary at end of each line to move to '\0'
					while (isspace(*pEnd)) { pEnd++; }
				}

				if (add_bias)	// only add bias after processing all features in the example
				{
					_values.push_back(1.0f);			// can reset value in SparseMatrixCSR
					_colidx.push_back(max_idx + 1);		// will reset after reading all data
				}
			}
		}
		catch (...)
		{
			throw runtime_error("Something wrong when reading data file: " + filename);
		}
		ifs.close();
	}
	// set the last element of row pointers in SparseMatrixCSR format
	_rowptr.push_back(_values.size());

	float t_loading = timer.seconds_from_last();

	// make sure column index starts with 0, as needed by sparse matrix operations
	size_t num_vals = _colidx.size();
	size_t min_idx = *min_element(_colidx.begin(), _colidx.end());
	// for distributed computing, it is necessary to specify?!
	if (min_idx < start_colidx) {
		throw runtime_error("Error in locading data file: min_idx smaller than start_colidx.");
	}
	if (start_colidx > 0)
	{
#pragma omp parallel for
		for (int64_t i = 0; i < num_vals; i++) {
			_colidx[i] -= start_colidx;
		}
	}
	// reset bias index for all examples to be the same last index
	size_t num_examples = _labels.size();
	max_idx = *max_element(_colidx.begin(), _colidx.end());
	if (add_bias)
	{
#pragma omp parallel for
		for (int64_t i = 0; i < num_examples; i++) {
			size_t bias_idx = _rowptr[i + 1] - 1;
			_colidx[bias_idx] = max_idx;	// this already include bias indexing
		}
	}

	// swap vectors to avoid reallocate memory, and clean up content of input vectors
	labels.swap(_labels);
	weights.swap(_weights);
	values.swap(_values);
	colidx.swap(_colidx);
	rowptr.swap(_rowptr);

	// display timing if disp_timing == true
	float t_process = timer.seconds_from_last();
	if (disp_timing) {
		std::cout << std::endl << "Loading datafiles:" << std::endl;
		for (auto & filename : filelist) {
			std::cout << "  " << filename << std::endl;
		}
		std::cout << "Time for loading datafiles: " << t_loading << " seconds." << std::endl;
		std::cout << "Time for post-processing:  " << t_process << " seconds." << std::endl;
		std::cout << "Number of examples: " << num_examples << std::endl;
		std::cout << "Number of features: " << max_idx + 1 << std::endl;
		std::cout << "Number of nonzeros: " << values.size() << std::endl;
		std::cout << std::endl;
	}

	return num_examples;
}

// Instantiate function templates with float and double types
template size_t load_datafile<float>(const list<string> filelist, vector<float> &labels, vector<float> &weights,
	vector<float> &vals, vector<size_t> &cidx, vector<size_t> &rptr,
	const bool add_bias, const bool disp_timing, const bool binary, const size_t start_colidx, const bool skip_first);
template size_t load_datafile<double>(const list<string> filelist, vector<double> &labels, vector<double> &weights,
	vector<double> &vals, vector<size_t> &cidx, vector<size_t> &rptr,
	const bool add_bias, const bool disp_timing, const bool binary, const size_t start_colidx, const bool skip_first);