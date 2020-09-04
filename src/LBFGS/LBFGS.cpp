// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <mpi.h>

#include "floattype.h"
#include "utils.h"
#include "randalgms.h"
#include "lbfgs_omp.h"

using namespace functions;
using namespace distropt;

int main(int argc, char* argv[])
{
	// In our implementation, only the main thread calls MPI, so we use MPI_THREAD_FUNNELED
	int mpi_thread_provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread_provided);

	MPI_Comm comm_world = MPI_COMM_WORLD;
	int mpi_size, mpi_rank;
	MPI_Comm_size(comm_world, &mpi_size);
	MPI_Comm_rank(comm_world, &mpi_rank);

	//---------------------------------------------------------------------------------------------
	// parse the input arguments
	std::string data_file;
	std::string init_mthd = "none";
	char loss_type, regu_type;
	double lambda, l1weight = 0;	// default l1 weight set to zero
	double rownorm = 1;				// default to normalize rows of X
	lbfgs_params params;

	int must_count = 0;
	for (int i = 1; i < argc; i += 2) {
		std::string optstr(argv[i]);
		std::string valstr(argv[i + 1]);
		if (optstr == "-data") {
			data_file = valstr;
			must_count++;
		}
		else if (optstr == "-loss") {
			loss_type = valstr[0];
			must_count++;
		}
		else if (optstr == "-lambda") {
			lambda = std::stof(valstr);
			must_count++;
		}
		else if (optstr == "-regu") {
			regu_type = valstr[0];
			must_count++;
		}
		else if (optstr == "-init") {
			init_mthd = valstr;
		}
		else if (optstr == "-maxitrs") {
			params.max_itrs = std::stoi(valstr);
		}
		else if (optstr == "-epsgrad") {
			params.eps_grad = std::stof(valstr);
		}
		else if (optstr == "-memory") {
			params.m_memory = std::stoi(valstr);
		}
		else if (optstr == "-lsrho") {
			params.btls_rho = std::stof(valstr);
		}
		else if (optstr == "-lsdec") {
			params.btls_dec = std::stof(valstr);
		}
		else if (optstr == "-lsmax") {
			params.btls_max = std::stoi(valstr);
		}
		else if (optstr == "-lsada") {
			params.btls_ada = (std::stoi(valstr) > 0);
		}
		else if (optstr == "-display") {
			params.display = (std::stoi(valstr) > 0);
		}
		else {
			std::cout << "LBFGS: Invalid arguments, please try again." << std::endl;
			std::exit(0);
		}
	}
	if (must_count < 4) {
		std::cout << "LBFGS arguments: -data <string> -loss <char> -lambda <double> -regu <char>" << std::endl;
		std::cout << "                 -init <string> -maxitrs <int> -epsgrad <double> -memory <int>" << std::endl;
		std::cout << "                 -lsrho <double> -lsdec <double> -lsmax <int> -lsada <bool>" << std::endl;
		std::cout << "                 -display <int>" << std::endl;
		std::cout << "The first 4 options must be given, others can use default values." << std::endl;
		std::exit(0);
	}

	// generate the output file name reflecting the dataset, algorithm, loss+regu, and lambda value
	std::string data_name = data_file.substr(data_file.find_last_of("/\\") + 1);
	std::stringstream sslambda;
	sslambda << std::scientific << std::setprecision(1) << lambda;
	std::string slambda = sslambda.str();
	slambda.erase(slambda.find_first_of("."), 2);
	std::string lsada = params.btls_ada ? "ada_" : "_";
	params.filename = data_name + "_" + std::string(1, loss_type) + std::string(1, regu_type) + "_lbfgs"
		+ std::to_string(params.m_memory) + lsada
		+ std::to_string(mpi_size) + "_" + slambda + "_" + init_mthd;

	//---------------------------------------------------------------------------------------------
	// read local data file
	string myfile = data_file + '_' + std::to_string(mpi_rank + 1);		// file labels start with 1

	std::vector<spmatT> labels;
	std::vector<spmatT> weights;
	std::vector<spmatT> values;
	std::vector<size_t> colidx;
	std::vector<size_t> rowptr;

	//std::cout << "Loading training data ... " << std::endl;
	double time_start = MPI_Wtime();
	size_t n_examples = load_datafile(myfile, labels, weights, values, colidx, rowptr, false, false, true);

	SparseMatrixCSR X(values, colidx, rowptr, false);
	Vector y(labels);
	if (rownorm > 0) { X.normalize_rows(rownorm); }

	MPI_Barrier(comm_world);

	if (mpi_rank == 0) {
		std::cout << "Loading files took " << MPI_Wtime() - time_start << " sec" << std::endl;
	}

	// use collective communications to get nTotalSamples and nAllFeatures
	size_t nSamples = X.nrows();
	size_t nFeatures = X.ncols();
	size_t nnzs = X.nnzs();
	size_t N, D, NZ;

	time_start = MPI_Wtime();
	MPI_Allreduce(&nSamples, &N, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm_world);
	MPI_Allreduce(&nFeatures, &D, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm_world);
	MPI_Allreduce(&nnzs, &NZ, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm_world);

	// make sure everyone have the same feature dimension
	X.reset_ncols(D);

	if (mpi_rank == 0) {
		std::cout << "MPI_Allreduce took " << MPI_Wtime() - time_start << " sec" << std::endl;
		std::cout << "N  = " << N << std::endl;
		std::cout << "D  = " << D << std::endl;
		std::cout << "NZ = " << NZ << std::endl;
	}

	//---------------------------------------------------------------------------------------------
	// construct loss function and regularization

	RegularizedLoss reguloss(X, y, loss_type, lambda, '2');

	auto localfval = [&reguloss](const Vector &w) { return reguloss.regu_loss(w); };
	auto localgrad = [&reguloss](const Vector &w, Vector &g) { return reguloss.regu_grad(w, g); };

	auto fval = [&reguloss, &comm_world, N, lambda](const Vector &w) {
		double sumloss = reguloss.sum_loss(w);
		MPI_Allreduce(MPI_IN_PLACE, &sumloss, 1, MPI_DOUBLE, MPI_SUM, comm_world);
		return sumloss / N + (lambda / 2)*w.dot(w);
	};

	auto grad = [&reguloss, &comm_world, N, lambda](const Vector &w, Vector &g) {
		double sumloss = reguloss.sum_loss(w);
		MPI_Allreduce(MPI_IN_PLACE, &sumloss, 1, MPI_DOUBLE, MPI_SUM, comm_world);
		reguloss.sum_grad(w, g);
		MPI_Allreduce(MPI_IN_PLACE, g.data(), g.length(), MPI_VECTOR_TYPE, MPI_SUM, comm_world);
		g.scale(1.0 / N);
		g.axpy(lambda, w);
		return sumloss / N + (lambda / 2)*w.dot(w);
	};

	//---------------------------------------------------------------------------------------------
	// compute initial condition using different methods
	Vector w(D), a(X.nrows());
	if (init_mthd == "sdca") {
		ISmoothLoss *f = nullptr;
		switch (loss_type)
		{
		case 'L':
		case 'l':
			f = new LogisticLoss();
			break;
		case 'S':
		case 's':
			f = new SmoothedHinge();
			break;
		case '2':
			f = new SquareLoss();
			break;
		default:
			throw std::runtime_error("Loss function type " + std::to_string(loss_type) + " not defined.");
		}
		SquaredL2Norm g;

		randalgms::sdca(X, y, *f, lambda, g, 20, 1e-8, w, a, 'p', 'd', params.display);
		delete f;
	}
	else if (init_mthd == "lbfgs") {
		params.display = false;
		lbfgs_omp(localfval, localgrad, w, w, params);
	}
	else {}

	// Allreduce to compute the sum and then the average of local solutions
	int w_len = int(w.length());
	MPI_Allreduce(MPI_IN_PLACE, w.data(), w_len, MPI_VECTOR_TYPE, MPI_SUM, comm_world);
	// need to compute the average as starting points for all machines
	w.scale(1.0 / mpi_size);


	// LBFGS algorithm ------------------------------------------------------------------------------
	params.display = mpi_rank == 0 ? true : false;
	params.fileoutput = mpi_rank == 0 ? true : false;

	MPI_Barrier(comm_world);
	lbfgs_omp(fval, grad, w, w, params);

	// always call MPI_Finalize()
	MPI_Finalize();
	return 0;
}
