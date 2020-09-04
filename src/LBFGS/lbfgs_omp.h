// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <functional>

#include "regu_loss.h"

using namespace functions;

namespace distropt {

	struct lbfgs_params {
		int    max_itrs = 100;
		int    m_memory = 20;
		double eps_grad = 1e-8;
		double btls_rho = 0.5;
		double btls_dec = 1e-4;
		int    btls_max = 10;
		bool   btls_ada = false;
		bool   display = true;
		bool   fileoutput = false;
		std::string filename = "temp.txt";

		// is this a good place to put it? probably not, good to leave with the application
		//void parse(int argc, char* argv[]);
	};

	int lbfgs_omp(std::function<double(const Vector&)> fval, std::function<double(const Vector&, Vector &)> grad, 
		          const Vector &x0, Vector &x, const lbfgs_params &params);
}