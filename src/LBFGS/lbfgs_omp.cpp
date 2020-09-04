// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include<deque>
#include<string>
#include<iostream>
#include<fstream>
#include<iomanip>

#include "timer.h"
#include "lbfgs_omp.h"

namespace distropt
{
	// formatted output: "iter  pcg_itrs   primal_obj    newton_dec    time" 
	void formatted_output(std::ostream &ofs, const int iters, const double stepsize, const int lscnt, const double f, const double gnorm, const double t)
	{
		ofs << std::setw(3) << iters
			<< std::scientific << std::setprecision(3)
			<< std::setw(12) << stepsize
			<< std::setw(4) << lscnt
			<< std::fixed << std::setprecision(12)
			<< std::setw(17) << f
			<< std::scientific << std::setprecision(3)
			<< std::setw(12) << gnorm
			<< std::fixed << std::setprecision(3)
			<< std::setw(9) << t
			<< std::endl;
	}

	int lbfgs_omp(std::function<double(const Vector&)> fval, std::function<double(const Vector&, Vector &)> grad,
		          const Vector &x0, Vector &x, const lbfgs_params &params)
	{
		// need more careful check, including function signature? may not be possible
		assert(x0.length() == x.length());

		size_t d = x0.length();
		int m = params.m_memory;

		// construct variables and working space
		Vector g(d), p(d), x_try(d);				// gradient, search direction, and next trial
		Vector x_pre(d), g_pre(d), dx(d), dg(d);	// vectors for L-BFGS memory
		std::deque<Vector> s, y;
		std::deque<double> rho;
		std::vector<double> alpha(m);

		// timing and recording
		double t_elapsed;
		HighResTimer timer;
		std::vector<double> stepsizes, objvals, gradnorms, time;
		std::vector<int> nls;

		if (params.display) {
			std::cout << std::endl << "iter   stepsize  ls         f(x)       |grad(x)|     time" << std::endl;
		}

		// L-BFGS iterations
		x.copy(x0);
		int k = 0, lscnt = 0;
		double stepsize = 1.0;

		// compute gradient at x
		double fx = grad(x, g);
		double g_norm = g.norm2();
		t_elapsed = timer.seconds_from_start();

		// record progress
		stepsizes.push_back(1.0);
		objvals.push_back(fx);
		gradnorms.push_back(g_norm);
		time.push_back(t_elapsed);
		nls.push_back(0);

		if (params.display)
		{
			formatted_output(std::cout, k, stepsize, lscnt, fx, g_norm, t_elapsed);
		}

		for (k = 1; k <= params.max_itrs; k++)
		{
			// update previous vectors
			x_pre.copy(x);
			g_pre.copy(g);

			// compute the limited-memory-matrix-vector product
			p.copy(g);
			for (int i = 0; i < s.size(); i++)
			{
				alpha[i] = rho[i] * s[i].dot(p);
				p.axpy(-alpha[i], y[i]);
			}
			double gamma = s.size() > 0 ? s[0].dot(y[0]) / (y[0].dot(y[0])) : 1.0;
			p.scale(gamma);
			for (int i = s.size() - 1; i > -1; i--)
			{
				double beta = rho[i] * y[i].dot(p);
				p.axpy(alpha[i] - beta, s[i]);
			}

			// backtracking line search (for nonconvex functions, need Wolfe conditions)
			stepsize = params.btls_ada ? stepsize / params.btls_rho : 1.0;
			double gdp = g.dot(p);
			for (lscnt = 0; lscnt < params.btls_max; lscnt++)
			{
				x_try.copy(x);
				x_try.axpy(-stepsize, p);
				if (fval(x_try) < fx - params.btls_dec*stepsize*gdp) {
					break;
				}
				stepsize *= params.btls_rho;
			}
			x.copy(x_try);

			// compute function value and gradient at new point
			fx = grad(x, g);
			g_norm = g.norm2();
			t_elapsed = timer.seconds_from_start();

			// record progress
			stepsizes.push_back(stepsize);
			objvals.push_back(fx);
			gradnorms.push_back(g_norm);
			time.push_back(t_elapsed);
			nls.push_back(lscnt);

			if (params.display)
			{
				formatted_output(std::cout, k, stepsize, lscnt, fx, g_norm, t_elapsed);
			}

			// stop criteria: norm of gradient 
			if (g_norm < params.eps_grad)
			{
				break;
			}

			// remove old items from queues if k > m
			if (k > m)
			{
				s.pop_back();
				y.pop_back();
				rho.pop_back();
			}

			// add new s and y to queue 
			Vector::elem_subtract(x, x_pre, dx);
			Vector::elem_subtract(g, g_pre, dg);
			s.push_front(dx);
			y.push_front(dg);
			rho.push_front(1.0 / dx.dot(dg));
		}

		//---------------------------------------------------------------------
		// write progress record in a file
		if (params.fileoutput) {
			std::ofstream ofs(params.filename, std::ofstream::out);
			ofs << "iter   stepsize  ls         f(x)       |grad(x)|     time" << std::endl;
			for (int i = 0; i < objvals.size(); i++) {
				formatted_output(ofs, i, stepsizes[i], nls[i], objvals[i], gradnorms[i], time[i]);
			}
			ofs.close();
		}

		// return number of iterations performed
		return k;
	}
}