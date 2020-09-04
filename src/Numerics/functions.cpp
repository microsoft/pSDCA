// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <assert.h>
#include <math.h>
#include <omp.h>
#include <cstdint>
#include <iostream>

// OpenMP 3.0 supports unsigned for-loop iterator, so can replace std::int64_t by size_t
#include "functions.h"

using std::int64_t;

namespace functions
{
	// mse for linear regression (1/n)*||X*w - y||^2 where n is length of y
	floatT regression_mse(const SparseMatrixCSR &X, const Vector &y, const Vector &w)
	{
		assert(X.nrows() == y.length() && X.ncols() == w.length());
		Vector yp(y);
		X.aAxby(1, w, -1, yp);	// yp = X * w - y
		return pow(yp.norm2(), 2) / yp.length();
	}

	// return number of prediction errors where sign(Xi'*w) != yi
	size_t binary_error_count(const SparseMatrixCSR &X, const Vector &y, const Vector &w)
	{
		assert(X.nrows() == y.length() && X.ncols() == w.length());
		Vector yp(y.length());
		X.aAxby(1, w, 0, yp);	// yp = X * w
		Vector::elem_multiply(y, yp, yp);
		return yp.count_negative();
	}

	// return error rate: number of errors (with sign(Xi'*w) != yi) divided by length of y
	floatT binary_error_rate(const SparseMatrixCSR &X, const Vector &y, const Vector &w)
	{
		assert(X.nrows() == y.length() && X.ncols() == w.length());
		Vector yp(y.length());
		X.aAxby(1, w, 0, yp);	// yp = X * w
		yp.to_sign();			// only keep signs in {-1, 0, +1}
		return (1 - yp.dot(y) / y.length()) / 2;
	}

	// Member functions of SquareLoss class: f(t) = (1/2)*(t-b)^2 ---------------------
	floatT SquareLoss::loss(const floatT Xiw, const floatT yi) const
	{
		return 0.5f * pow(Xiw - yi, 2);
	}

	void SquareLoss::loss(const Vector &Xw, const Vector y, Vector &loss) const
	{
		assert(Xw.length() == y.length() && y.length() == loss.length());
		size_t len = y.length();

		#pragma omp parallel for 
		for (int64_t i = 0; i < len; i++)
		{
			loss[i] = 0.5f * pow(Xw[i] - y[i], 2);
		}
	}

	// return F(Xw) = sum_i 0.5*(Xw[i]-y[i])^2
	floatT SquareLoss::sum_loss(const Vector &Xw, const Vector &y) const
	{
		assert(Xw.length() == y.length());
		size_t len = y.length();
		floatT sum = 0;

		#pragma omp parallel for reduction(+:sum)
		for (int64_t i = 0; i < len; i++)
		{
			sum += 0.5f * pow(Xw[i] - y[i], 2);
		}
		return sum;
	}

	// return F(Xw) = sum_i 0.5*(Xw[i]-y[i])^2
	floatT SquareLoss::sum_loss(const Vector &theta, const Vector &Xw, const Vector &y) const
	{
		assert(Xw.length() == y.length() && theta.length() == y.length());
		size_t len = y.length();
		floatT sum = 0;

		#pragma omp parallel for reduction(+:sum)
		for (int64_t i = 0; i < len; i++)
		{
			sum += 0.5f * pow(Xw[i] - y[i], 2) * theta[i];
		}
		return sum;
	}

	// return di = f'(Xiw) = Xiw - yi
	floatT SquareLoss::derivative(const floatT Xiw, const floatT yi) const
	{
		return Xiw - yi;
	}

	// return d = F'(Xw) = Xw - y
	void SquareLoss::derivative(const Vector &Xw, const Vector &y, Vector &d) const
	{
		assert(Xw.length() == y.length() && y.length() == d.length());
		Vector::elem_subtract(Xw, y, d);
	}

	floatT SquareLoss::second_derivative(const floatT Xiw, const floatT yi) const
	{
		return 1;
	}

	void SquareLoss::second_derivative(const Vector &Xw, const Vector &y, Vector &d) const
	{
		assert(Xw.length() == y.length() && y.length() == d.length());
		d.fill(1);
	}

	floatT SquareLoss::conjugate(const floatT ai, const floatT yi) const
	{
		return 0.5f * ai*ai + yi*ai;
	}

	void SquareLoss::conjugate(const Vector &a, const Vector &y, Vector &conj) const
	{
		assert(a.length() == y.length() && y.length() == conj.length());
		size_t len = y.length();

		#pragma omp parallel for 
		for (int64_t i = 0; i < len; i++)
		{
			conj[i] = 0.5f * a[i] * a[i] + y[i] * a[i];
		}
	}

	// return f*(a) = (1/2)*||a||^2 + a'*y
	floatT SquareLoss::sum_conjugate(const Vector &a, const Vector &y) const
	{
		assert(a.length() == y.length());
		return 0.5f * a.dot(a) + a.dot(y);
	}

	floatT SquareLoss::sum_conjugate(const Vector &theta, const Vector &a, const Vector &y) const
	{
		assert(a.length() == y.length() && theta.length() == y.length());
		size_t len = y.length();

		floatT sum = 0;
		#pragma omp parallel for reduction(+:sum)
		for (int64_t i = 0; i < len; i++)
		{
			sum += (0.5f * a[i] * a[i] + y[i] * a[i]) * theta[i];
		}
		return sum;
	}

	floatT SquareLoss::conj_derivative(const floatT ai, const floatT yi) const
	{
		return ai + yi;
	}

	void SquareLoss::conj_derivative(const Vector &a, const Vector &y, Vector &cd) const
	{
		assert(a.length() == y.length() && y.length() == cd.length());
		Vector::elem_add(a, y, cd);
	}

	// Return argmin_b {sigma*(f*)(b) + (1/2)*||b - a||_2^2} 
	floatT SquareLoss::conj_prox(const floatT sigma, const floatT ai, const floatT yi) const
	{
		return (ai - sigma*yi) / (1 + sigma);
	}

	// Return argmin_b {sigma*(f*)(b) + (1/2)*||b - a||_2^2} 
	void SquareLoss::conj_prox(const floatT sigma, const Vector &a, const Vector &y, Vector &u) const
	{
		assert(a.length() == y.length() && y.length() == u.length());
		size_t len = a.length();

		#pragma omp parallel for 
		for (int64_t i = 0; i < len; i++) 
		{
			u[i] = (a[i] - sigma*y[i]) / (1 + sigma);
		}
	}

	// Member functions of LogisticLoss class ------------------------------------------

	LogisticLoss::LogisticLoss(const floatT deltabd, const floatT epsilon, const int maxiter)
	{
		assert(deltabd > 0 && deltabd < 1 && epsilon > 0 && maxiter > 0);
		_deltabd = deltabd;
		_epsilon = epsilon;
		_maxiter = maxiter;
	
		// conjugate function value when dual variable is within deltabd to boundary of [-1,0]
		_xentrpy = deltabd*log(deltabd) + (1 - deltabd)*log1p(-deltabd);
		_xderiva = log1p(-deltabd) - log(deltabd);
	}

	floatT LogisticLoss::loss(const floatT Xiw, const floatT yi) const
	{
		floatT yxwi = yi * Xiw;
		return yxwi > 50 ? 0 : yxwi < -50 ? -yxwi : log1p(exp(-yxwi));
	}

	void LogisticLoss::loss(const Vector &Xw, const Vector y, Vector &loss) const
	{
		assert(Xw.length() == y.length() && y.length() == loss.length());
		size_t len = y.length();

		#pragma omp parallel for 
		for (int64_t i = 0; i < len; i++)
		{
			loss[i] = this->loss(Xw[i], y[i]);
		}
	}

	// Return sum_i  log(1+exp(-y[i]*Xw[i])). Need to parallelize using OpenMP
	floatT LogisticLoss::sum_loss(const Vector &Xw, const Vector &y) const
	{
		assert(Xw.length() == y.length());
		size_t len = y.length();
		double sum = 0;

		#pragma omp parallel for reduction(+:sum)
		for (int64_t i = 0; i < len; i++)
		{
			sum += this->loss(Xw[i], y[i]);
		}
		return floatT(sum);
	}

	floatT LogisticLoss::sum_loss(const Vector &theta, const Vector &Xw, const Vector &y) const
	{
		assert(Xw.length() == y.length() && theta.length() == y.length());
		size_t len = y.length();
		double sum = 0;

		#pragma omp parallel for reduction(+:sum)
		for (int64_t i = 0; i < len; i++)
		{
			sum += this->loss(Xw[i], y[i]) * theta[i];
		}
		return floatT(sum);
	}

	floatT LogisticLoss::derivative(const floatT Xiw, const floatT yi) const
	{
		return -yi / (1 + exp(yi*Xiw));
	}

	// d[i] = -y[i]/(1+exp(y[i]*Xw[i]) so that gradient is expressed as sum_i d[i]*X[i,:]
	void LogisticLoss::derivative(const Vector &Xw, const Vector &y, Vector &d) const
	{
		assert(Xw.length() == y.length() && Xw.length() == y.length());
		size_t len = y.length();

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++)
		{
			d[i] = -y[i] / (1 + exp(y[i] * Xw[i]));
		}
	}

	floatT LogisticLoss::second_derivative(const floatT Xiw, const floatT yi) const
	{
		return 1.0 / (2 + exp(yi*Xiw) + exp(-yi*Xiw));
	}

	// d[i] = -y[i]/(1+exp(y[i]*Xw[i]) so that gradient is expressed as sum_i d[i]*X[i,:]
	void LogisticLoss::second_derivative(const Vector &Xw, const Vector &y, Vector &d) const
	{
		assert(Xw.length() == y.length() && Xw.length() == y.length());
		size_t len = y.length();

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++)
		{
			d[i] = 1.0 / (2 + exp(y[i] * Xw[i]) + exp(-y[i] * Xw[i]));
		}
	}

	bool LogisticLoss::conj_feasible(const Vector &a, const Vector &y) const
	{
		assert(a.length() == y.length());
		//Vector ya(y.length());
		//Vector::elem_multiply(y, a, ya);
		//return (ya.max() <= 0 && ya.min() >= -1);

		size_t len = y.length();
		for (size_t i = 0; i < len; i++)
		{
			if (!this->conj_feasible(a[i], y[i])) { return false; }
		}
		return true;
	}

	floatT LogisticLoss::conjugate(const floatT ai, const floatT yi) const
	{
		// Need to check if all a_j belongs to [-1, 0], but can ignore if initialized this way 
		floatT t = yi*ai;
		assert(t >= -1 && t <= 0);
		return t > -_deltabd ? _xentrpy : t < -1.0 + _deltabd ? _xentrpy : (-t) * log(-t) + (1.0 + t) * log1p(t);
	}

	void LogisticLoss::conjugate(const Vector &a, const Vector &y, Vector &conj) const
	{
		assert(a.length() == y.length() && y.length() == conj.length());
		size_t len = y.length();

		#pragma omp parallel for 
		for (int64_t i = 0; i < len; i++)
		{
			conj[i] = this->conjugate(a[i], y[i]);
		}
	}

	// Return (f*)(a) sum_i (-ya[i])*log(-ya[i]) + (1+ya[i])*log(1+ya[i]) with ya[i] in [-1, 0]
	floatT LogisticLoss::sum_conjugate(const Vector &a, const Vector &y) const
	{
		assert(a.length() == y.length());
		size_t len = a.length();
		double sum = 0;

		#pragma omp parallel for reduction(+:sum)
		for (int64_t i = 0; i < len; i++)
		{
			sum += this->conjugate(a[i], y[i]);
		}
		return floatT(sum);
	}

	floatT LogisticLoss::sum_conjugate(const Vector &theta, const Vector &a, const Vector &y) const
	{
		assert(a.length() == y.length() && theta.length() == y.length());
		size_t len = a.length();
		double sum = 0;

		#pragma omp parallel for reduction(+:sum)
		for (int64_t i = 0; i < len; i++)
		{
			sum += this->conjugate(a[i], y[i]) * theta[i];
		}
		return floatT(sum);
	}

	floatT LogisticLoss::conj_derivative(const floatT ai, const floatT yi) const
	{
		floatT t = yi*ai;
		assert(t >= -1 && t <= 0);
		return t > -_deltabd ? yi*_xderiva : t < -1.0 + _deltabd ? -yi*_xderiva : yi*(log1p(t) - log(-t));
	}

	void LogisticLoss::conj_derivative(const Vector &a, const Vector &y, Vector &cd) const
	{
		assert(a.length() == y.length() && y.length() == cd.length());
		size_t len = y.length();

		#pragma omp parallel for 
		for (int64_t i = 0; i < len; i++)
		{
			cd[i] = this->conj_derivative(a[i], y[i]);
		}
	}

	// Return argmin_b {sigma*(f*)(b) + (1/2)*(b - a)^2} 
	floatT LogisticLoss::conj_prox(const floatT sigma, const floatT ai, const floatT yi) const
	{
		// using double here is critical for numerical stability
		double lb = _deltabd - 1.0, ub = -_deltabd;
		double epsilon = _epsilon;
		int maxiter = _maxiter;

		double yai = ai*yi;					// need to convert sign first to use one code
		double bi = fmin(ub, fmax(lb, yai));
		double f = 0, df = 0;
		for (int k = 0; k < maxiter; ++k)
		{
			f = bi - yai + sigma * log((1.0 + bi) / (-bi));
			if (fabs(f) < epsilon) break;
			df = 1.0 - sigma / (bi * (1.0 + bi));
			bi -= f / df;
			bi = fmin(ub, fmax(lb, bi));	// critical for convergence
		}
		return floatT(bi * yi);				
	}

	// Return argmin_b {sigma*(f*)(b) + (1/2)*||b - a||_2^2} 
	void LogisticLoss::conj_prox(const floatT sigma, const Vector &a, const Vector &y, Vector &b) const
	{
		assert(a.length() == y.length() && y.length() == b.length());
		size_t len = y.length();

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++)
		{
			b[i] = this->conj_prox(sigma, a[i], y[i]);
		}
	}

	// Member functions of SmoothedHinge class ------------------------------------------

	floatT SmoothedHinge::loss(const floatT Xiw, const floatT yi) const
	{
		floatT yXwi = yi * Xiw;
		floatT t = 1 - yXwi;
		return yXwi >= 1 ? 0 : t >= _delta ? t - _delta / 2 : t*t / (2 * _delta);
	}

	void SmoothedHinge::loss(const Vector &Xw, const Vector y, Vector &loss) const
	{
		assert(Xw.length() == y.length() && y.length() == loss.length());
		size_t len = y.length();

		#pragma omp parallel for 
		for (int64_t i = 0; i < len; i++)
		{
			loss[i] = this->loss(Xw[i], y[i]);
		}
	}

	floatT SmoothedHinge::sum_loss(const Vector &Xw, const Vector &y) const
	{
		assert(Xw.length() == y.length());
		size_t len = y.length();
		floatT sum = 0;

		#pragma omp parallel for reduction(+:sum)
		for (int64_t i = 0; i < len; ++i)
		{
			sum += this->loss(Xw[i], y[i]);
		}
		return sum;
	}

	floatT SmoothedHinge::sum_loss(const Vector &theta, const Vector &Xw, const Vector &y) const
	{
		assert(Xw.length() == y.length() && theta.length() == y.length());
		size_t len = y.length();
		floatT sum = 0;

		#pragma omp parallel for reduction(+:sum)
		for (int64_t i = 0; i < len; ++i)
		{
			sum += this->loss(Xw[i], y[i]) * theta[i];
		}
		return sum;
	}

	floatT SmoothedHinge::derivative(const floatT Xiw, const floatT yi) const
	{
		floatT yXwi = yi * Xiw;
		floatT di = yXwi>1 ? 0 : yXwi < 1 - _delta ? -1 : (yXwi - 1) / _delta;
		return yi*di;
	}

	// Compute d[i] so that gradient is expressed as sum_i d[i]*X[i,:]
	void SmoothedHinge::derivative(const Vector &Xw, const Vector &y, Vector &d) const
	{
		assert(Xw.length() == y.length() && Xw.length() == y.length());
		size_t len = y.length();

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++)
		{
			d[i] = this->derivative(Xw[i], y[i]);
		}
	}

	floatT SmoothedHinge::second_derivative(const floatT Xiw, const floatT yi) const
	{
		floatT yXwi = yi * Xiw;
		floatT di = yXwi>1 ? 0 : yXwi < 1 - _delta ? 0 : 1.0 / _delta;
		return yi*di;
	}

	// Compute d[i] so that gradient is expressed as sum_i d[i]*X[i,:]
	void SmoothedHinge::second_derivative(const Vector &Xw, const Vector &y, Vector &d) const
	{
		assert(Xw.length() == y.length() && Xw.length() == y.length());
		size_t len = y.length();

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++)
		{
			d[i] = this->second_derivative(Xw[i], y[i]);
		}
	}

	bool SmoothedHinge::conj_feasible(const Vector &a, const Vector &y) const
	{
		assert(a.length() == y.length());
		//Vector ya(y.length());
		//Vector::elem_multiply(y, a, ya);
		//return (ya.max() <= 0 && ya.min() >= -1);

		size_t len = y.length();
		for (size_t i = 0; i < len; i++)
		{
			if (!this->conj_feasible(a[i], y[i])) { return false; }
		}
		return true;
	}

	floatT SmoothedHinge::conjugate(const floatT ai, const floatT yi) const
	{
		floatT yai = yi*ai;
		assert(yai >= -1 && yai <= 0);
		return yai + (_delta / 2)*(yai*yai);
	}

	void SmoothedHinge::conjugate(const Vector &a, const Vector &y, Vector &conj) const
	{
		assert(a.length() == y.length() && y.length() == conj.length());
		size_t len = y.length();

		#pragma omp parallel for 
		for (int64_t i = 0; i < len; i++)
		{
			conj[i] = this->conjugate(a[i], y[i]);
		}
	}

	floatT SmoothedHinge::sum_conjugate(const Vector &a, const Vector &y) const
	{
		assert(a.length() == y.length());
		size_t len = y.length();
		floatT sum = 0;

		#pragma omp parallel for reduction(+:sum) 
		for (int64_t i = 0; i < len; i++)
		{
			floatT yai = y[i] * a[i];
			assert(yai >= -1 && yai <= 0);
			sum += yai + (_delta / 2)*(yai*yai);
			//sum += this->conjugate(a[i], y[i]);
		}
		return sum;
	}

	floatT SmoothedHinge::sum_conjugate(const Vector &theta, const Vector &a, const Vector &y) const
	{
		assert(a.length() == y.length() && theta.length() == y.length());
		size_t len = y.length();
		floatT sum = 0;

		#pragma omp parallel for reduction(+:sum) 
		for (int64_t i = 0; i < len; i++)
		{
			floatT yai = y[i] * a[i];
			assert(yai >= -1 && yai <= 0);
			sum += (yai + (_delta / 2)*(yai*yai)) * theta[i];
			//sum += this->conjugate(a[i], y[i]) * theta[i];
		}
		return sum;
	}

	floatT SmoothedHinge::conj_derivative(const floatT ai, const floatT yi) const
	{
		floatT yai = yi*ai;
		assert(yai >= -1 && yai <= 0);
		return _delta*ai + yi;
	}

	void SmoothedHinge::conj_derivative(const Vector &a, const Vector &y, Vector &cd) const
	{
		assert(a.length() == y.length() && y.length() == cd.length());
		cd.copy(y);
		Vector::axpy(_delta, a, cd);
	}

	// Return argmin_b {sigma*(f*)(b) + (1/2)*||b - a||_2^2} 
	floatT SmoothedHinge::conj_prox(const floatT sigma, const floatT ai, const floatT yi) const
	{
		// should not assert yi*ai in [-1, 0] here because ai can be arbitrary
		floatT bi = (yi*ai - sigma) / (1 + sigma*_delta);
		return yi*(bi > 0 ? 0 : bi < -1 ? -1 : bi);
	}

	// Return argmin_b {sigma*(f*)(b) + (1/2)*||b - a||_2^2} 
	void SmoothedHinge::conj_prox(const floatT sigma, const Vector &a, const Vector &y, Vector &b) const
	{
		assert(a.length() == y.length() && y.length() == b.length());
		size_t len = y.length();
		floatT delta = _delta;

		#pragma omp parallel for default(none) shared(a,y,b,len,delta)
		for (int64_t i = 0; i < len; i++)
		{
			floatT bi = (y[i]*a[i] - sigma) / (1 + sigma*delta);
			b[i] = y[i] * (bi > 0 ? 0 : bi < -1 ? -1 : bi);
		}
	}

	// Member functions of SquaredL2Norm class ------------------------------------------

	floatT SquaredL2Norm::operator()(const Vector &w) const 
	//floatT SquaredL2Norm::value(const Vector &w) const
	{
		return 0.5f * pow(w.norm2(), 2);
	}

	// Compute f*(v) = max_w {v'*w - f(w)}
	floatT SquaredL2Norm::conjugate(const Vector &v) const
	{ 
		return 0.5f *pow(v.norm2(), 2);
	}

	// Compute f*(v) = max_w {v'*w - f(w)}, with maxing w already computed
	floatT SquaredL2Norm::conjugate(const Vector &v, const Vector &w) const
	{
		assert(v.length() == w.length());
		return this->conjugate(v);
	}

	// prox() operator computes u = argmin_u { tau*f(u) + (1/2)*||u-w||_2^2 }
	void SquaredL2Norm::prox(const floatT tau, const Vector &w, Vector &u) const
	{
		assert(w.length() == u.length());
		if (&w != &u) { u.copy(w); }
		u.scale(1 / (1 + tau));
	}

	// Given vector v, compute g = argmax_w {v'*w - f(w)} = argmin_w {-v'*w + f(w)}
	void SquaredL2Norm::conj_grad(const Vector &v, Vector &w) const
	{
		assert(v.length() == w.length());
		if (&w != &v) { w.copy(v); }
	}

	// Member functions of ElasticNet class ------------------------------------------

	//floatT ElasticNet::value(const Vector &w) const
	floatT ElasticNet::operator()(const Vector &w) const 
	{
		return 0.5f * pow(w.norm2(), 2) + _lambda1 * w.norm1();
	}

	// Compute f*(v) = max_w {v'*w - f(w)}
	floatT ElasticNet::conjugate(const Vector &v) const
	{
		Vector w(v.length());
		this->conj_grad(v, w);
		return v.dot(w) - (*this)(w);
	}

	// Compute f*(v) = max_w {v'*w - f(w)}, with maxing w already computed
	floatT ElasticNet::conjugate(const Vector &v, const Vector &w) const
	{
		assert(v.length() == w.length());
		return v.dot(w) - (*this)(w);
	}

	// Compute u = argmin_u { tau*f(u) + (1/2)*||u-w||_2^2 }
	// Note that we allow u and w being the same Vector, that is, allow &w==&u
	void ElasticNet::prox(const floatT tau, const Vector &w, Vector &u) const
	{
		assert(w.length() == u.length());
		floatT tl2 = 1 + tau;
		floatT tl1 = tau * _lambda1;
		size_t len = w.length();

		#pragma omp parallel for
		for(int64_t i=0; i<len; i++)
		{
			floatT vneg = w[i] + tl1;
			floatT vpos = w[i] - tl1;
			u[i] = vpos > 0 ? vpos / tl2 : vneg < 0 ? vneg / tl2 : 0;
		}
	}

	// Given vector v, compute g = argmax_w {v'*w - f(w)} = argmin_w {-v'*w + f(w)}
	void ElasticNet::conj_grad(const Vector &v, Vector &w) const
	{
		size_t len = v.length();
		floatT lambda1 = _lambda1;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++)
		{
			floatT vneg = v[i] + lambda1;
			floatT vpos = v[i] - lambda1;
			w[i] = vpos > 0 ? vpos : vneg < 0 ? vneg : 0;
		}
	}

	// Member functions of UnitQuadratic class ------------------------------------------

	OffsetQuadratic::OffsetQuadratic(const size_t dim)
	{
		_w0 = new Vector(dim);
		_dw = new Vector(dim);
	}

	OffsetQuadratic::~OffsetQuadratic()
	{
		if (_w0 != nullptr) delete _w0;
		if (_dw != nullptr) delete _dw;
	}

	floatT OffsetQuadratic::operator()(const Vector &w) const
	{
		Vector::elem_subtract(w, *_w0, *_dw);
		//Vector _dw(w);
		//Vector::axpy(-1, *_w0, *_dw);
		return 0.5f * pow(_dw->norm2(), 2);
	}

	floatT OffsetQuadratic::conjugate(const Vector &v) const
	{
		return 0.5f * pow(v.norm2(), 2) + v.dot(*_w0);
	}

	floatT OffsetQuadratic::conjugate(const Vector &v, const Vector &w) const
	{
		assert(v.length() == w.length());
		// return v.dot(w) - (*this)(w);	
		// but it's easier to call directly
		return this->conjugate(v);
	}

	void OffsetQuadratic::prox(const floatT tau, const Vector &w, Vector &u) const
	{
		assert(w.length() == u.length());
		if (&w != &u) { u.copy(w); }
		Vector::axpy(tau, *_w0, u);
		u.scale(1.0f / (1.0f + tau));
	}

	// Given vector v, compute g = argmax_w {v'*w - f(w)} = argmin_w {-v'*w + f(w)}
	// this one does not exploit sparsity, can be computationally very expensive
	// see efficient implementation in sdca
	void OffsetQuadratic::conj_grad(const Vector &v, Vector &w) const
	{
		assert(v.length() == w.length());
		//if (&w != &v) { w.copy(v); }
		//Vector::axpy(1.0f, *_w0, w);
		Vector::elem_add(*_w0, v, w);
	}
}