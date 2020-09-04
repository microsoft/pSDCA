// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <vector>
#include <numeric>
#include <random>
#include <algorithm>

#include "utils.h"

// generate a random permutation vector for indices 0, 1, 2, ..., N-1
std::vector<size_t> random_permutation(const size_t N, const int seed)
{
	std::vector<size_t> p(N);
	std::iota(p.begin(), p.end(), 0);
	std::random_device rdev;
	unsigned int rand_seed = seed > 0 ? seed : rdev();
	std::shuffle(p.begin(), p.end(), std::default_random_engine(rand_seed));
	return std::move(p);
}