// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <string>
#include <iostream>
#include <algorithm>
#include <random>
#include <numeric>
#include <stdexcept>
#include <ctime>
#include <math.h>

#include "utils.h"
#include "randalgms.h"

using namespace std;
using namespace randalgms;

//int main_test_sdca(int argc, char* argv[])
int main(int argc, char* argv[])
{
	if (argc < 2) {
		std::cout << "Need to specify training data file." << std::endl;
		std::exit(0);
	}
	string trainingfile = argv[1];
	string testingfile;
	if (argc >= 3){ testingfile = argv[2]; }

	// read training and testing files
	vector<spmatT> labels;
	vector<spmatT> weights;
	vector<spmatT> values;
	vector<size_t> colidx;
	vector<size_t> rowptr;

	std::cout << "Loading training data ... " << std::endl;
	size_t n_examples = load_datafile(trainingfile, labels, weights, values, colidx, rowptr, false, true, true);
	SparseMatrixCSR X(values, colidx, rowptr, false);
	Vector y(labels);

	size_t dim = X.ncols();

	SparseMatrixCSR *X_test = nullptr;
	Vector *y_test = nullptr;
	if (!testingfile.empty()){
		std::cout << "Loading test data ... " << std::endl;
		load_datafile(testingfile, labels, weights, values, colidx, rowptr, false, true, true);
		X_test = new SparseMatrixCSR(values, colidx, rowptr, false);
		y_test = new Vector(labels);

		dim = X.ncols() >= X_test->ncols() ? X.ncols() : X_test->ncols();
		X.reset_ncols(dim);
		X_test->reset_ncols(dim);
	}

	// training using SDCA and display error rates
	Vector w(dim), a(X.nrows());
	ISmoothLoss *f = new LogisticLoss();
	//ISmoothLoss *f = new SmoothedHinge();
	//ISmoothLoss *f = new SquareLoss();
	IUnitRegularizer *g = new SquaredL2Norm();
	//IUnitRegularizer *g = new ElasticNet(1);
	OffsetQuadratic q(dim);
	//IUnitRegularizer *g = &q;
	q.update_offset(w);

	double lambda = 1.0e-5;
	int max_epoch = 100;
	double eps_gap = 1e-8;

	X.normalize_rows();
	sdca(X, y, *f, lambda, *g, max_epoch, eps_gap, w, a, 'p', 'd');
	a.fill_zero();
	sdca(X, y, *f, lambda, *g, max_epoch, eps_gap, w, a, 'p', 'p');
	a.fill_zero();
	sdca(X, y, *f, lambda, *g, max_epoch, eps_gap, w, a, 'p', 'b');

	/*
	q.update_offset(w);
	a.fill_zero();
	sdca(X, y, *f, lambda, *g, max_epoch, eps_gap, w, a, 'u', 'd');
	// shuffle the row indices and do it again
	std::vector<size_t> idx(X.nrows());
	std::iota(std::begin(idx), std::end(idx), 0);
	std::shuffle(std::begin(idx), std::end(idx), std::default_random_engine(0));
	SubSparseMatrixCSR Xshfl(X, idx);
	Vector yshfl(y, idx);
	// w.fill_zero();	// no need to clean w because only a is used to initialize SDCA
	a.fill_zero();
	Vector ashfl(a, idx);
	sdca(Xshfl, yshfl, *f, lambda, *g, max_epoch, eps_gap, w, ashfl, 'w', 'd');
	*/

	float train_err = binary_error_rate(X, y, w);
	std::cout << "Training error rate = " << train_err * 100 << " %" << std::endl;
	if (!testingfile.empty()) {
		float test_err = binary_error_rate(*X_test, *y_test, w);
		std::cout << "Testing error rate  = " << test_err * 100 << " %" << std::endl;
	}

	// do not forget to delete object pointers!
	if (X_test != nullptr) { delete X_test; }
	if (y_test != nullptr) { delete y_test; }
	delete f;
	if (g->symbol() != 'q') delete g;

	return 0;
}

