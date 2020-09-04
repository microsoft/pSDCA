// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include<iostream>
//#include<functional>

#include "utils.h"
#include "randalgms.h"
#include "lbfgs_omp.h"

using namespace std;
using namespace distropt;

int main_test_lbfgs(int argc, char* argv[])
//int main(int argc, char* argv[])
{
	if (argc < 2) {
		std::cout << "Need to specify training data file." << std::endl;
		std::exit(0);
	}
	string data_file = argv[1];

	// read training and testing files
	vector<spmatT> labels;
	vector<spmatT> weights;
	vector<spmatT> values;
	vector<size_t> colidx;
	vector<size_t> rowptr;

	std::cout << "Loading training data ... " << std::endl;
	// news20 actually has column index starting from 0, rcv1 and most others start with 1. Need to revise code!
	//size_t n_examples = load_datafile(data_file, labels, weights, values, colidx, rowptr, false, true, true, 0);
	size_t n_examples = load_datafile(data_file, labels, weights, values, colidx, rowptr, false, true, true);
	SparseMatrixCSR X(values, colidx, rowptr, false);
	Vector y(labels);

	X.normalize_rows();

	// -------------------------------------------------------------------------------------
	lbfgs_params params;
	params.max_itrs = 100;
	params.eps_grad = 1e-8;
	params.m_memory = 20;
	params.btls_rho = 0.5;
	params.btls_dec = 1e-4;
	params.btls_max = 20;
	params.btls_ada = false;

	double lambda = 1.0e-5;

	size_t dim = X.ncols();
	Vector w0(dim), w(dim);

	// -------------------------------------------------------------------------------------
	RegularizedLoss reguloss(X, y, 'l', lambda, '2');

	// Replace std::bind solution by using lambdas
	//using namespace std::placeholders;
	//auto fval = std::bind(&RegularizedLoss::regu_loss, &reguloss, _1);
	//auto grad = std::bind(&RegularizedLoss::regu_grad, &reguloss, _1, _2);

	auto fval = [&reguloss](const Vector &x) {return reguloss.regu_loss(x); };
	auto grad = [&reguloss](const Vector &x, Vector &g) {return reguloss.regu_grad(x, g); };

	lbfgs_omp(fval, grad, w0, w, params);

	// -------------------------------------------------------------------------------------
	float train_err = binary_error_rate(X, y, w);
	std::cout << "Training error rate = " << train_err * 100 << " %" << std::endl;

	return 0;
}

