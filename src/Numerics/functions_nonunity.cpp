#include <assert.h>
#include <math.h>
#include <omp.h>
#include <cstdint>

// OpenMP 3.0 supports unsigned for-loop iterator, so can replace std::int64_t by size_t
#include "functions.h"

using std::int64_t;

namespace randalgms
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
	inline floatT SquareLoss::loss(const float Xiw, const floatT yi) const
	{
		return pow(Xiw - yi, 2) / 2;
	}

	void SquareLoss::loss(const Vector &Xw, const Vector y, Vector &loss) const
	{
		assert(Xw.length() == y.length() && y.length() == loss.length());
		size_t len = y.length();

		#pragma omp parallel for 
		for (int64_t i = 0; i < len; i++)
		{
			loss[i] = pow(Xw[i] - y[i], 2) / 2;
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
			floatT zi = Xw[i] - y[i];
			sum += zi*zi / 2;
		}
		return sum;
	}

	// return di = f'(Xiw) = Xiw - yi
	inline floatT SquareLoss::derivative(const floatT Xiw, const floatT yi) const
	{
		return Xiw - yi;
	}

	// return d = F'(Xw) = Xw - y
	void SquareLoss::derivative(const Vector &Xw, const Vector &y, Vector &d) const
	{
		assert(Xw.length() == y.length() && y.length() == d.length());
		Vector::elem_subtract(Xw, y, d);
	}

	inline floatT SquareLoss::conjugate(const floatT &ai, const floatT yi) const
	{
		return ai*ai / 2 + yi*ai;
	}

	void SquareLoss::conjugate(const Vector &a, const Vector &y, Vector &conj) const
	{
		assert(a.length() == y.length() && y.length() == conj.length());
		size_t len = y.length();

		#pragma omp parallel for 
		for (int64_t i = 0; i < len; i++)
		{
			conj[i] = a[i] * a[i] / 2 + y[i] * a[i];
		}
	}

	// return f*(a) = (1/2)*||a||^2 + a'*y
	floatT SquareLoss::sum_conjugate(const Vector &a, const Vector &y) const
	{
		assert(a.length() == y.length());
		return a.dot(a) / 2 + a.dot(y);
	}

	// Return argmin_b {sigma*(f*)(b) + (1/2)*||b - a||_2^2} 
	inline floatT SquareLoss::conj_prox(floatT sigma, const floatT ai, const floatT yi) const
	{
		return (ai - sigma*yi) / (1 + sigma);
	}

	// Return argmin_b {sigma*(f*)(b) + (1/2)*||b - a||_2^2} 
	void SquareLoss::conj_prox(floatT sigma, const Vector &a, const Vector &y, Vector &u) const
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

	LogisticLoss::LogisticLoss(const floatT min_log, const floatT epsilon, const int maxiter)
	{
		assert(min_log > 0 && epsilon > 0 && maxiter > 0);
		_min_log = min_log;
		_epsilon = epsilon;
		_maxiter = maxiter;
	}

	inline floatT LogisticLoss::loss(const float Xiw, const floatT yi) const
	{
		floatT yxwi = yi * Xiw;
		return yxwi > 50 ? 0 : yxwi < -50 ? -yxwi : log(1 + exp(-yxwi));
	}

	void   LogisticLoss::loss(const Vector &Xw, const Vector y, Vector &loss) const
	{
		assert(Xw.length() == y.length() && y.length() == loss.length());
		size_t len = y.length();

		#pragma omp parallel for 
		for (int64_t i = 0; i < len; i++)
		{
			floatT yxwi = y[i] * Xw[i];
			loss[i] = yxwi > 50 ? 0 : yxwi < -50 ? -yxwi : log(1 + exp(-yxwi));
		}
	}

	// Return sum_i  log(1+exp(-y[i]*Xw[i])). Need to parallelize using OpenMP
	floatT LogisticLoss::sum_loss(const Vector &Xw, const Vector &y) const
	{
		assert(Xw.length() == y.length());
		size_t len = y.length();
		floatT sum = 0;

		#pragma omp parallel for reduction(+:sum)
		for (int64_t i = 0; i < len; i++)
		{
			floatT yxwi = y[i] * Xw[i];
			sum += yxwi > 50 ? 0 : yxwi < -50 ? -yxwi : log(1 + exp(-yxwi));
		}
		return sum;
	}

	inline floatT LogisticLoss::derivative(const floatT Xiw, const floatT yi) const
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

	inline floatT LogisticLoss::conjugate(const floatT &ai, const floatT yi) const
	{
		assert(yi*ai >= -1 && yi*ai <= 0);	// yi*ai in [-1, 0], may not be necessary here

		// clip may not work here, especially for float 1e-12 - 1 = -1, log(1+t)=log(0)
		// better approach: should directly bound entropy argument from zero!
		double t = fmin(-_min_log, fmax(_min_log - 1, yi*ai));	// clip [-1 + min_log, - min_log]
		return floatT((-t) * log(-t) + (1 + t) * log(1 + t));
	}

	void LogisticLoss::conjugate(const Vector &a, const Vector &y, Vector &conj) const
	{
		assert(a.length() == y.length() && y.length() == conj.length());
		size_t len = y.length();
		double lb = _min_log - 1, ub = -_min_log;

		#pragma omp parallel for 
		for (int64_t i = 0; i < len; i++)
		{
			assert(y[i] * a[i] >= -1 && y[i] * a[i] <= 0);
			double t = fmin(ub, fmax(lb, y[i] * a[i]));
			conj[i] = floatT((-t) * log(-t) + (1 + t) * log(1 + t));
		}
	}

	// Return (f*)(a) sum_i (-ya[i])*log(-ya[i]) + (1+ya[i])*log(1+ya[i]) with ya[i] in [-1, 0]
	floatT LogisticLoss::sum_conjugate(const Vector &a, const Vector &y) const
	{
		// Need to check if all a_j belongs to [-1, 0], but can ignore if initialized this way 
		assert(a.length() == y.length());
		size_t len = a.length();
		double lb = _min_log - 1, ub = -_min_log;
		double sum = 0;

		// this code may have more numerical error accumulation !!?
		#pragma omp parallel for reduction(+:sum)
		for (int64_t i = 0; i < len; i++)
		{
			assert(y[i] * a[i] >= -1 && y[i] * a[i] <= 0);
			double t = fmin(ub, fmax(lb, y[i] * a[i]));
			sum += (-t) * log(-t) + (1 + t) * log(1 + t);
		}
		return floatT(sum);
	}

	// Return argmin_b {sigma*(f*)(b) + (1/2)*(b - a)^2} 
	floatT LogisticLoss::conj_prox(floatT sigma, const floatT ai, const floatT yi) const
	{
		floatT lb = _min_log - 1, ub = -_min_log;
		floatT epsilon = _epsilon;
		int maxiter = _maxiter;

		double yai = ai*yi;					// need to convert sign first to use one code
		double bi = fmin(ub, fmax(lb, yai));
		double f = 0, df = 0;
		for (int k = 0; k < maxiter; ++k)
		{
			f = bi - yai + sigma * log((1 + bi) / (-bi));
			if (fabs(f) < epsilon) break;
			df = 1.0 - sigma / (bi * (1 + bi));
			bi -= f / df;
			bi = fmin(ub, fmax(lb, bi));	// critical for convergence
		}
		return floatT(bi * yi);				
	}

	// Return argmin_b {sigma*(f*)(b) + (1/2)*||b - a||_2^2} 
	void LogisticLoss::conj_prox(floatT sigma, const Vector &a, const Vector &y, Vector &b) const
	{
		// Need to check if all a_j belongs to [-1, 0], but can ignore because always clipped
		assert(a.length() == y.length());
		Vector ya(y.length());
		Vector::elem_multiply(y, a, ya);	// u[i] = y[i]*a[i]. Here y[i] need to be in {-1, +1};

		floatT lb = _min_log - 1, ub = -_min_log;
		floatT epsil = _epsilon;
		int maxiter = _maxiter;
		size_t len = y.length();

		#pragma omp parallel for default(none) shared(y,ya,b,sigma,lb,ub,epsilon,maxiter,len)
		for (int64_t i = 0; i < len; i++)
		{
			double bi = fmin(ub, fmax(lb, ya[i]));
			double f = 0, df = 0;
			for (int k = 0; k < maxiter; ++k)
			{
				f = bi - ya[i] + sigma * log((1 + bi) / (-bi));
				if (fabs(f) < epsil) break;
				df = 1.0 - sigma / (bi * (1 + bi));
				bi -= f / df;
				bi = fmin(ub, fmax(lb, bi));	// critical for convergence
			}
			b[i] = floatT(bi * y[i]);			
		}
	}

	// Member functions of SmoothedHinge class ------------------------------------------

	inline floatT SmoothedHinge::loss(const float Xiw, const floatT yi) const
	{
		floatT yXwi = yi * Xiw;
		floatT t = 1 - yXwi;
		return yXwi >= 1 ? 0 : t >= _delta ? t - _delta / 2 : t*t / (2 * _delta);
	}

	void SmoothedHinge::loss(const Vector &Xw, const Vector y, Vector &loss) const
	{
		assert(Xw.length() == y.length() && y.length() == loss.length());
		size_t len = y.length();
		floatT delta = _delta;

		#pragma omp parallel for 
		for (int64_t i = 0; i < len; i++)
		{
			floatT yXwi = y[i] * Xw[i];
			floatT t = 1 - yXwi;
			loss[i] = yXwi >= 1 ? 0 : t >= delta ? t - delta / 2 : t*t / (2 * delta);
		}
	}

	floatT SmoothedHinge::sum_loss(const Vector &Xw, const Vector &y) const
	{
		assert(Xw.length() == y.length());
		size_t len = y.length();
		floatT delta = _delta;
		floatT sum = 0;

		#pragma omp parallel for reduction(+:sum)
		for (int64_t i = 0; i < len; ++i)
		{
			floatT yXwi = y[i] * Xw[i];
			floatT t = 1.0f - yXwi;
			sum += yXwi >= 1 ? 0 : t >= delta ? t - delta / 2 : t*t / (2 * delta);
		}
		return sum;
	}

	inline floatT SmoothedHinge::derivative(const floatT Xiw, const floatT yi) const
	{
		floatT yXwi = yi * Xiw;
		return yXwi>1 ? 0 : yXwi < 1 - _delta ? -1 : (yXwi - 1) / _delta;
	}

	// Compute d[i] so that gradient is expressed as sum_i d[i]*X[i,:]
	void SmoothedHinge::derivative(const Vector &Xw, const Vector &y, Vector &d) const
	{
		assert(Xw.length() == y.length() && Xw.length() == y.length());
		size_t len = y.length();
		floatT delta = _delta;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++)
		{
			floatT yXwi = y[i] * Xw[i];
			d[i] = yXwi>1 ? 0 : yXwi < 1 - delta ? -1 : (yXwi - 1) / delta;
		}
	}

	inline floatT SmoothedHinge::conjugate(const floatT &ai, const floatT yi) const
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
			floatT yai = y[i] * a[i];
			assert(yai >= -1 && yai <= 0);
			conj[i] = yai + (_delta / 2)*(yai*yai);
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
		}
		return sum;
	}

	// Return argmin_b {sigma*(f*)(b) + (1/2)*||b - a||_2^2} 
	inline floatT SmoothedHinge::conj_prox(floatT sigma, const floatT ai, const floatT yi) const
	{
		// should not assert yi*ai in [-1, 0] here because ai can be arbitrary
		floatT bi = (yi*ai - sigma) / (1 + sigma*_delta);
		return yi*(bi > 0 ? 0 : bi < -1 ? -1 : bi);
	}

	// Return argmin_b {sigma*(f*)(b) + (1/2)*||b - a||_2^2} 
	void SmoothedHinge::conj_prox(floatT sigma, const Vector &a, const Vector &y, Vector &b) const
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

	SquaredL2Norm::SquaredL2Norm(const floatT lambda) 
	{ 
		assert(lambda > 0); 
		_lambda = lambda; 
		if (lambda == 1.0f) {
			_is_unity = true;
		}
		else {
			_is_unity = false;
		}
	}

	SquaredL2Norm::SquaredL2Norm(const SquaredL2Norm &g, bool unitize)
	{
		if (unitize) {
			_lambda = 1.0f;
			_is_unity = true;
		}
		else {
			_lambda = g._lambda;
			_is_unity = g._is_unity;
		}
	}

	inline floatT SquaredL2Norm::unitize()
	{
		floatT previous_lambda = _lambda;
		_lambda = 1.0f;
		_is_unity = true;
		return previous_lambda;
	}

	floatT SquaredL2Norm::operator()(const Vector &w) const 
	//floatT SquaredL2Norm::value(const Vector &w) const
	{
		return (0.5f *_lambda)*pow(w.norm2(), 2);
	}

	// Compute f*(v) = max_w {v'*w - f(w)}
	floatT SquaredL2Norm::conjugate(const Vector &v) const
	{ 
		return (0.5f / _lambda)*pow(v.norm2(), 2);
	}

	// Compute f*(v) = max_w {v'*w - f(w)}, with maxing w already computed
	floatT SquaredL2Norm::conjugate(const Vector &v, const Vector &w) const
	{
		return this->conjugate(v);
	}

	// prox() operator computes u = argmin_u { tau*f(u) + (1/2)*||u-w||_2^2 }
	void SquaredL2Norm::prox(const floatT tau, const Vector &w, Vector &u) const
	{
		if (&w != &u) { u.copy(w); }
		u.scale(1 / (1 + tau*_lambda));
	}

	// Given vector v, compute g = argmax_w {v'*w - f(w)} = argmin_w {-v'*w + f(w)}
	void SquaredL2Norm::conj_grad(const Vector &v, Vector &w) const
	{
		if (&w != &v) { w.copy(v); }
		if (!_is_unity) {
			w.scale(1.0f / _lambda);
		}
	}

	// Member functions of ElasticNet class ------------------------------------------

	ElasticNet::ElasticNet(const floatT lambda1, const floatT lambda2)
	{ 
		assert(lambda1 >= 0 && lambda2 > 0); 
		_lambda1 = lambda1; 
		_lambda2 = lambda2; 
		if (lambda2 == 1.0f) {
			_is_unity = true;
		}
		else {
			_is_unity = false;
		}
	}

	ElasticNet::ElasticNet(const ElasticNet &g, bool unitize)
	{
		if (unitize) {
			_lambda1 = g._lambda1 / g._lambda2;
			_lambda2 = 1.0f;
			_is_unity = true;
		}
		else {
			_lambda1 = g._lambda1;
			_lambda2 = g._lambda2;
			_is_unity = g._is_unity;
		}
	}

	inline floatT ElasticNet::unitize()
	{
		floatT previous_lambda2 = _lambda2;
		_lambda1 = _lambda1 / _lambda2;
		_lambda2 = 1.0f;
		_is_unity = true;
		return previous_lambda2;
	}


	//floatT ElasticNet::value(const Vector &w) const
	floatT ElasticNet::operator()(const Vector &w) const 
	{
		return (_lambda2 / 2)*pow(w.norm2(), 2) + _lambda1*w.norm1();
	}

	// Compute f*(v) = max_w {v'*w - f(w)}
	floatT ElasticNet::conjugate(const Vector &v) const
	{
		Vector w(v.length());
		this->conj_grad(v, w);
		return v.dot(w) - (*this)(v);
	}

	// Compute f*(v) = max_w {v'*w - f(w)}, with maxing w already computed
	floatT ElasticNet::conjugate(const Vector &v, const Vector &w) const
	{
		assert(v.length() == w.length());
		return v.dot(w) - (*this)(v);
	}

	// Compute u = argmin_u { tau*f(u) + (1/2)*||u-w||_2^2 }
	// Note that we allow u and w being the same Vector, that is, allow &w==&u
	void ElasticNet::prox(const floatT tau, const Vector &w, Vector &u) const
	{
		floatT tl2 = 1 + tau * _lambda2;
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
	void ElasticNet::conj_grad(const Vector &v, Vector &g) const
	{
		size_t len = v.length();
		floatT lambda1 = _lambda1;
		floatT lambda2 = _lambda2;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++)
		{
			floatT vneg = v[i] + lambda1;
			floatT vpos = v[i] - lambda1;
			g[i] = vpos > 0 ? vpos / lambda2 : vneg < 0 ? vneg / lambda2 : 0;
		}
	}
}