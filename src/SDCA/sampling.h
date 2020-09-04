// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace randalgms {

	// an abstract class to serve as interface to random index generators 
	class IRandomIndexGenerator
	{
	protected: 
		size_t _n;					// range of number of indices 0, 1, 2, ..., n-1 
		std::mt19937_64 _rng;		// random number generator, seeded in constructor

	public: 
		IRandomIndexGenerator(const size_t n) : _n(n) { 
			std::random_device rd; 
			_rng.seed(rd()); 
		};
		virtual size_t operator()() = 0;
		virtual size_t next() = 0;
	};

	class RandIndex_Unifom : public IRandomIndexGenerator
	{
	private: 
		std::uniform_int_distribution<size_t> _unif;

	public: 
		RandIndex_Unifom(const size_t n) : IRandomIndexGenerator(n), _unif(0, n - 1) {};
		
		size_t operator()() override { return _unif(_rng); };
		size_t next() override { return _unif(_rng); };
	};

	class RandIndex_Permut : public IRandomIndexGenerator
	{
	private:
		std::vector<size_t> _indices;
		size_t _count;

	public:
		RandIndex_Permut(const size_t n) : IRandomIndexGenerator(n), _indices(n), _count(0) {
			std::iota(_indices.begin(), _indices.end(), 0);				// 0, 1, ..., n-1
			std::shuffle(_indices.begin(), _indices.end(), _rng);		// random permutation
		};

		size_t operator()() override { return this->next(); };

		size_t next() override {
			if (_count == _n) {
				std::shuffle(_indices.begin(), _indices.end(), _rng);	// random permutation
				_count = 0;												// reset _count = 0
			}
			return _indices[_count++]; 
		};
	};

	class RandIndex_Weighted : public IRandomIndexGenerator
	{
	private:
		std::discrete_distribution<size_t> _distr;

	public:
		RandIndex_Weighted(const std::vector<float> weights)
			:IRandomIndexGenerator(weights.size()), _distr(weights.begin(), weights.end()) {};
		RandIndex_Weighted(const std::vector<double> weights)
			:IRandomIndexGenerator(weights.size()), _distr(weights.begin(), weights.end()) {};
		
		size_t operator()() override { return _distr(_rng); };
		size_t next() override { return _distr(_rng); };
	};
}