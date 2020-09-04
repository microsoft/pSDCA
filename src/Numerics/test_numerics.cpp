// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <numeric>
#include <algorithm>

#include "utils.h"
#include "timer.h"
#include "spmatrix.h"
#include "floattype.h"

using Vector = numerics::Vector<floatT>;
using SubVector = numerics::SubVector<floatT>;
using SparseMatrixCSR = numerics::SparseMatrixCSR<spmatT, floatT>;
using SubSparseMatrixCSR = numerics::SubSparseMatrixCSR<spmatT, floatT>;

int main(int argc, char* argv[])
{
	//string filename = "C:/Data/LIBSVM/splice1percent100_1";
	//string filename = "C:/Data/LIBSVM/ads1percent100_1";
	//string filename = "C:/Data/LIBSVM/rcv1_test.binary";
	string filename = "C:/Data/LIBSVM/rcv1_train.binary";
	//string filename = "C:/Data/LIBSVM/news20.binary";
	//string filename = "C:/Data/LIBSVM/covtype.libsvm.binary.scale";

	vector<spmatT> labels;
	vector<spmatT> weights;
	vector<spmatT> values;
	vector<size_t> colidx;
	vector<size_t> rowptr;

	load_datafile(filename, labels, weights, values, colidx, rowptr, false, true, true);

	SparseMatrixCSR X(values, colidx, rowptr, false, false);
	Vector y(labels);
	size_t N = X.nrows();
	size_t D = X.ncols();
	std::cout << "Sparse matrix with " << N << " rows, " << D << " cols, " << X.nnzs() << " nnzs." << std::endl;

	// test matrix-vector multiplication -----------------------------------------
	Vector v(D), Xv(N), u(N), XTu(D);
	u.fill_rand(0, 1);
	v.fill_rand(0, 1);

	HighResTimer timer;
	X.aAxby(1, v, 0, Xv);
	float t_elpsd = timer.seconds_from_last();
	std::cout << "Matrix-Vector multiplications took " << t_elpsd << " seconds)." << std::endl;

	timer.reset();
	X.aATxby(1, u, 0, XTu);
	t_elpsd = timer.seconds_from_last();
	std::cout << "Matrix.T-Vector multiplications took " << t_elpsd << " seconds)." << std::endl;

	// ---------------------------------------------------------------------------
	// test submatrix splitting with random indices, prepare for DSCOVR algorithms
	size_t m = 10;
	size_t K = 20;
	std::vector<size_t> row_ids = random_permutation(N);
	std::vector<size_t> col_ids = random_permutation(D);

	std::vector<SparseMatrixCSR*> XiT(m);
	std::vector<std::vector<SubSparseMatrixCSR*>> XiTk(m);
	for (size_t i = 0; i < m; i++) { XiTk[i].resize(K); }

	size_t row_stride = N / m;
	size_t row_remain = N % m;
	size_t col_stride = D / K;
	size_t col_remain = D % K;
	SubSparseMatrixCSR Xi(X, 0, 1);				// temporary initialization
	size_t row_start = 0;
	for (size_t i = 0; i < m; i++) {
		size_t n_rows = i < row_remain ? row_stride + 1 : row_stride;
		std::vector<size_t> subrows(row_ids.begin() + row_start, row_ids.begin() + (row_start + n_rows));
		Xi.recast(X, subrows);
		XiT[i] = new SparseMatrixCSR(Xi, true);
		row_start += n_rows;

		size_t col_start = 0;
		for (size_t k = 0; k < K; k++) {
			size_t n_cols = k < col_remain ? col_stride + 1 : col_stride;
			std::vector<size_t> subcols(col_ids.begin() + col_start, col_ids.begin() + (col_start + n_cols));
			XiTk[i][k] = new SubSparseMatrixCSR(*XiT[i], subcols);	// actually rows of XiT after transpose
			col_start += n_cols;
		}
	}

	// test matrix vector multiplications using the split submatrices
	Vector Xv2(N), Xv3(N), uperm(N), XTu2(D);
	SubVector Xiv(Xv2, 0, N);
	u.post_permute(row_ids, uperm);
	SubVector ui(uperm, 0, N);
	row_start = 0;
	for (size_t i = 0; i < m; i++) {
		Xiv.recast(Xv2, row_start, XiT[i]->ncols());
		XiT[i]->aATxby(1, v, 0, Xiv);
		ui.recast(uperm, row_start, XiT[i]->ncols());
		XiT[i]->aAxby(1, ui, 1, XTu2);
		row_start += Xiv.length();
	}
	std::cout << "||X*v|| = " << Xv.norm2() << std::endl;
	std::cout << "||X1*v; ...; Xm*v|| = " << Xv2.norm2() << std::endl;
	Xv2.pre_permute(row_ids, Xv3);
	Vector::axpy(-1, Xv, Xv3);
	std::cout << "||X*v - [X1*v; ...; Xm*v]|| = " << Xv3.norm2() << std::endl;

	std::cout << "||X'*u|| = " << XTu.norm2() << std::endl;
	std::cout << "||X1'*u1 + ... + Xm'*um|| = " << XTu2.norm2() << std::endl;
	Vector::axpy(-1, XTu, XTu2);
	std::cout << "||X'*u - (X1'*u1 + ... + Xm'*um)|| = " << XTu2.norm2() << std::endl;

	// delete pointers allocated with "new"
	for (size_t i = 0; i < m; i++) {
		delete XiT[i];
		for (size_t k = 0; k < K; k++) {
			delete XiTk[i][k];
		}
	}
	return 0;
}
