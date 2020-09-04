// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <stdexcept>
#include <string>

#include "randalgms.h"
#include "sampling.h"
#include "timer.h"
#include "display.h"

namespace randalgms {

	// RPCD algorithm for solving CoCoA local optimization problems
	int rpcd_cocoa(const SparseMatrixCSR &X, const Vector &y, const ISmoothLoss &f, const double lambda_N, const double sigma,
		const int max_epoch, const Vector &Xw, const Vector &a0, Vector &a1, const char opt_sample, const bool display)

	{
		size_t n = X.nrows();
		size_t d = X.ncols();

		if (y.length() != n || Xw.length() != n || a0.length() != n || a1.length() != n) {
			throw std::runtime_error("RPCD(CoCoA): Input/output matrix and vector dimensions do not match.");
		}

		// start tracking time elapsed.
		HighResTimer timer;

		// initialize primal-dual variables and working space variables
		Vector L(n);		// vector to store row-wise Lipschitz constant
		X.row_sqrd2norms(L);

		// construct the random index generator after vector L is computed
		IRandomIndexGenerator *randidx;
		switch (opt_sample) {
		case 'u':	// i.i.d. uniform sampling
			randidx = new RandIndex_Unifom(n);
			break;
		case 'w':	// i.i.d. weighted sampling proportional to L[i]
			randidx = new RandIndex_Weighted(L.to_vector());
			break;
		case 'p':	// random permutations for each epoch
			randidx = new RandIndex_Permut(n);
			break;
		default:
			throw std::runtime_error("RPCD(CoCoA): sampling method " + std::string(1, opt_sample) + " is not defined.");
		}

		// scale L for convenience of computation: L[i] = ||X_i||^2 * sigma / (lambda * n)
		L.scale(sigma / lambda_N);			

		// need to test if initial a is feasible using f.conj_feasible() return true or false
		a1.copy(a0);
		if (!f.conj_feasible(a1, y)) {
			a1.fill_zero();
		}
		double Da = f.sum_conjugate(a1, y);		// objective value with initial a1 = a0 
		Da /= n;

		double t_elpsd = timer.seconds_from_start();
		if (display) {
			display_header("RPCD(CoCoA) with options: sample(" + std::string(1, opt_sample) + ").");
			display_progress(0, 0, Da, t_elpsd);
		}

		Vector XTda(d);		// X(a-a0), needed for computing gradient of quadratic (L/2)||X(a-a0)||^2
		size_t i;
		double ai, yi, Li, gi, ai1;		// elements of dual variables a1 and Xw
		int epoch = 0;
		while (epoch < max_epoch)
		{
			// iterate over data once for each epoch
			for (int64_t iter = 0; iter < n; iter++)
			{
				i = randidx->next();

				ai = a1[i];
				yi = y[i];
				Li = L[i];

				gi = -Xw[i] + (sigma / lambda_N)*X.row_dot(i, XTda);

				if (Li > 0)
				{
					ai1 = f.conj_prox(1.0f / Li, ai - gi / Li, yi);
				}

				if (ai1 != ai)
				{
					X.row_add(i, ai1 - ai, XTda);
					a1[i] = ai1;
				}
			}

			epoch++;

			Da = f.sum_conjugate(a1, y) - Xw.dot(a1) + 0.5*(sigma / lambda_N)*pow(XTda.norm2(), 2);
			Da /= n;

			t_elpsd = timer.seconds_from_start();
			if (display) {
				display_progress(epoch, 0, Da, t_elpsd);
			}
		}

		// delete pointers initialized with "new"
		delete randidx;

		return epoch;
	}
}