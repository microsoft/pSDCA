// Numerics.cpp : Defines the entry point for the console application.

#include <string>
#include <iostream>
#include <stdexcept>
#include <ctime>
#include <math.h>

#include "spmatrix.h"
#include "utils.h"

using namespace std;
using namespace Numerics;

//int main_test_sparsematrix(int argc, char* argv[])
int main(int argc, char* argv[])
{
	//string filename = "C:/Data/LIBSVMdata/splice1percent100_1";
	string filename = "C:/Data/LIBSVMdata/ads1percent100_1";
	//string filename = "C:/Data/LIBSVMdata/rcv1_test.binary";
	//string filename = "C:/Data/LIBSVMdata/rcv1_train.binary";
	//string filename = "C:/Data/LIBSVMdata/news20.binary";
	//string filename = "C:/Data/LIBSVMdata/covtype.libsvm.binary.scale";

	vector<float>  vals, labl;
	vector<size_t> cidx;
	vector<size_t> rptr;

	clock_t time;
	time = clock();
	Load_LIBSVM_file(filename, vals, cidx, rptr, labl);
	time = clock() - time;
	std::cout << "Loading file took " << ((float)time) / CLOCKS_PER_SEC << " seconds)." << std::endl;

	SparseMatrixCSR A(vals, cidx, rptr);
	std::cout << A.Rows() << ' ' << A.Cols() << ' ' << A.NNZs() << std::endl;

	Vector y(labl);
	Vector x(A.Cols());

	time = clock();
	A.aAxby(1, x, 0, y);
	time = clock() - time;
	std::cout << "Matrix-Vector multiplications took " << ((float)time) / CLOCKS_PER_SEC << " seconds)." << std::endl;

	time = clock();
	A.aATxby(1, y, 0, x);
	time = clock() - time;
	std::cout << "Matrix.T-vector multiplications took " << ((float)time) / CLOCKS_PER_SEC << " seconds)." << std::endl;

	return 0;
}
