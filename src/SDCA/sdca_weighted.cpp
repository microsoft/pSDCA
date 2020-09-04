// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <stdexcept>
#include <string>

#include "randalgms.h"
#include "sampling.h"
#include "timer.h"
#include "display.h"

namespace randalgms {

	// SDCA need the regularization function to have strongly convex parameter 1 (unit strong convexity)
	int sdca(const SparseMatrixCSR &X, const Vector &y, const Vector &theta, const ISmoothLoss &f, const double lambda,
		const IUnitRegularizer &g, const int max_epoch, const double eps_gap, Vector &w, Vector &a,
		const char opt_sample, const char opt_update, const bool display)
	{
		size_t n = X.nrows();
		size_t d = X.ncols();

		if (y.length() != n || theta.length() != n || w.length() != d || a.length() != n) {
			throw std::runtime_error("SDCA: Input/output matrix and vector dimensions do not match.");
		}

		// start tracking time elapsed.
		HighResTimer timer;

		// initialize primal-dual variables and working space variables
		Vector L(n);							// vector to store row-wise Lipschitz constant
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
			throw std::runtime_error("SDCA: sampling method " + std::string(1, opt_sample) + " is not defined.");
		}

		double gammainv = f.smoothness();		// f is 1/gamma smooth
		double lambda_n = lambda*n;				// sdca suppose g has unit strong convexity
		L.scale(1.0 / lambda_n);				// scale L for easy computation: L[i] = ||X_i||^2 / (lambda * n)
		Vector::elem_multiply(L, theta, L);		// modifications for update in the weighted case

		// need to test if initial a is feasible using f.conj_feasible() return true or false
		if (!f.conj_feasible(a, y)) {
			a.fill_zero();
		}
		// initial dual variable a is used (not w) because SDCA is fundamentally a dual algoithm
		Vector v(d);							// intermediate variable to compute next w
		Vector Xw(n);							// vector to store X * w
		Vector::elem_multiply(theta, a, Xw);	// borrow memory of Xw for computing v (not affecting later)
		X.aATxby(-1.0 / lambda_n, Xw, 0, v);	// v = - (X' * (theta.*a)) / (lambda * n)
		g.conj_grad(v, w);						// w = \nabla g*(v) determined by a
		X.aAxby(1.0f, w, 0, Xw);				// Xw = X * w
												// compute primal and dual objective values, and duality gap
		double Pw = f.sum_loss(theta, Xw, y) / n + lambda*g(w);
		double Da = -f.sum_conjugate(theta, a, y) / n - lambda*g.conjugate(v, w);
		double Pw_Da = Pw - Da;

		double t_elpsd = timer.seconds_from_start();
		if (display) {
			display_header("SDCA with options: sample(" + std::string(1, opt_sample)
				+ "), update(" + std::string(1, opt_update) + ").");
			display_progress(0, Pw, Da, t_elpsd);
		}

		// SDCA main loop
		size_t i;
		double ai, yi, Li, Xiw, ai1, da;		// elements of dual variables a1 and Xw
		double cfp = 2;
		int epoch = 0;
		while (epoch < max_epoch)
		{
			// iterate over data once for each epoch
			for (int64_t iter = 0; iter < n; iter++)
			{
				i = randidx->next();

				ai = a[i];
				yi = y[i];
				Li = L[i];

				// compute Xi'*w, exploit structure of g
				switch (g.symbol()) {
				case 'q':
					Xiw = X.row_dot(i, v) + X.row_dot(i, *g.offset());
					break;
				default:
					Xiw = X.row_dot(i, w);
					break;
				}

				da = 0;
				if (Li > 0)
				{
					switch (opt_update) {
					case 'p':	// primal update, use primal function gradient for update
						da = (f.derivative(Xiw, yi) - ai) / (1.0 + Li*gammainv);
						ai1 = ai + da;
						break;
					case 'd':	// dual update, use dual (conjugate) function prox mapping
					default:
						ai1 = f.conj_prox(1.0f / Li, ai + Xiw / Li, yi);
						da = ai1 - ai;
						break;
					}
				}

				if (da != 0)
				{
					switch (g.symbol()) {
					case '2':	// regularization g is unit L2 norm squared 
						X.row_add(i, - theta[i] * da / lambda_n, w);
						break;
					case 'q':
						X.row_add(i, - theta[i] * da / lambda_n, v);
						break;
					case 'e':	// regularization g is elastic net (unit L2 + L1)
						X.row_add(i, - theta[i] * da / lambda_n, v);
						X.row_sst(i, g.l1penalty(), v, w);
						break;
					default:	// generic update w = \nabla g*(v)
						X.row_add(i, - theta[i] * da / lambda_n, v);
						g.conj_grad(v, w);
						break;
					}

					a[i] = ai1;
				}
			}

			epoch++;

			// re-calculate v if necessary to combat numerical error accumulation (useful for async?)
			//X.aATxby(-1.0 / lambda_n, a, 0, v);	// here need to multiply by theta as well!
			//g.conj_grad(v, w);

			// compute primal and dual objective values and duality gap
			if (g.symbol() == 'q') {
				Vector::elem_add(*g.offset(), v, w);
			}
			X.aAxby(1.0f, w, 0, Xw);
			Pw = f.sum_loss(theta, Xw, y) / n + lambda*g(w);
			if (g.symbol() == '2') {
				Da = -f.sum_conjugate(theta, a, y) / n - lambda*g.conjugate(w);
			}
			else {
				Da = -f.sum_conjugate(theta, a, y) / n - lambda*g.conjugate(v);
			}

			t_elpsd = timer.seconds_from_start();
			if (display) {
				display_progress(epoch, Pw, Da, t_elpsd);
			}

			Pw_Da = Pw - Da;
			if (Pw_Da < eps_gap) {
				break;
			}
		}

		// delete pointers initialized with "new"
		delete randidx;

		return epoch;
	}
}

