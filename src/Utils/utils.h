// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include<list>
#include<string>
#include<vector>

using namespace std;

// generate a random permutation vector for indices 0, 1, 2, ..., N-1
std::vector<size_t> random_permutation(const size_t N, const int seed = -1);

// Load datafile with line format: <label>[:<weight>] [<colidx>[:<value>]] ...
// By default LIVSVM colidx starts with 1, but SparseMatrixCSR starts with 0
template <typename T>
size_t load_datafile(const string filename, vector<T> &labels, vector<T> &weights,
	vector<T> &values, vector<size_t> &colidx, vector<size_t> &rowptr,
	const bool add_bias = false, const bool disp_timing = false, const bool binary = false, 
	const size_t start_colidx = 1, const bool skip_first = false);

// Need to make some change to read criteo dataset (skip the second entry for each row)
template <typename T>
size_t load_datafile(const list<string> filelist, vector<T> &labels, vector<T> &weights,
	vector<T> &values, vector<size_t> &colidx, vector<size_t> &rowptr,
	const bool add_bias = false, const bool disp_timing = false, const bool binary = false,
	const size_t start_colidx = 1, const bool skip_first = false);
