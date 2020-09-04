// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <assert.h>
#include <math.h> 
#include <algorithm>
#include <functional>
#include <random>
#include <cstdint>
#include <omp.h>

#include "dnvector.h"

// OpenMP 3.0 supports unsigned for-loop iterator, so can replace std::int64_t by size_t
using std::int64_t;

namespace numerics {

	// Construct an all-zero vector
	template <typename T> 
	Vector<T>::Vector(const size_t n)
	{
		_length = n;
		_elements = new T[n];
		memset(_elements, 0, n * sizeof(T));
		_need_cleanup = true;
	}

	// Constructor from another Vector
	template <typename T>
	Vector<T>::Vector(const Vector<T> &v)
	{
		_length = v._length;
		_elements = new T[_length];
		memcpy(_elements, v._elements, _length * sizeof(T));
		_need_cleanup = true;
	}

	template <typename T>
	Vector<T>::Vector(const std::vector<float> &v)
	{
		_length = v.size();
		_elements = new T[_length];
		// cannot use memcpy() here since T may not be float
		//memcpy(_elements, v.data(), _length * sizeof(T));	
		for (size_t i = 0; i < _length; i++) {
			_elements[i] = v[i];
		}
		_need_cleanup = true;
	}

	template <typename T>
	Vector<T>::Vector(const std::vector<double> &v)
	{
		_length = v.size();
		_elements = new T[_length];
		// cannot use memcpy() here since T may not be double
		//memcpy(_elements, v.data(), _length * sizeof(T));
		for (size_t i = 0; i < _length; i++) {
			_elements[i] = v[i];
		}
		_need_cleanup = true;
	}

	template <typename T>
	Vector<T>::Vector(const Vector<T> &v, const std::vector<size_t> &indices)
	{
		assert(*std::min_element(std::begin(indices), std::end(indices)) >= 0
			&& *std::max_element(std::begin(indices), std::end(indices)) < v.length());

		_length = indices.size();
		_elements = new T[_length];
		for (size_t i = 0; i < _length; i++) {
			_elements[i] = v[indices[i]];
		}
		_need_cleanup = true;
	}
	
	// Destructor, also called by derived class SubVector without cleaning up memory
	template <typename T>
	Vector<T>::~Vector()
	{
		if (_need_cleanup) delete[] _elements;
	}

	// convert to std::vector<T>, with move return after c++11
	template <typename T>
	std::vector<T> Vector<T>::to_vector()
	{
		std::vector<T> v(_length);
		memcpy(v.data(), _elements, _length * sizeof(T));
		return std::move(v);
	}

	// find pre-permutation vector v[p[i]] = this[i]
	template <typename T>
	void Vector<T>::pre_permute(const std::vector<size_t> &p, Vector<T> &v)
	{
		assert(v._length == _length && p.size() == _length);
		assert(*std::min_element(std::begin(p), std::end(p)) >= 0
			&& *std::max_element(std::begin(p), std::end(p)) < v.length());

		auto elem = _elements;
		auto len = _length;
		for (size_t i = 0; i < len; i++) {
			v[p[i]] = elem[i];
		}
	}

	// find post-permutation vector v[i] = this[p[i]]
	template <typename T>
	void Vector<T>::post_permute(const std::vector<size_t> &p, Vector<T> &v)
	{
		assert(v._length == _length && p.size() == _length);
		assert(*std::min_element(std::begin(p), std::end(p)) >= 0
			&& *std::max_element(std::begin(p), std::end(p)) < v.length());

		auto elem = _elements;
		auto len = _length;
		for (size_t i = 0; i < len; i++) {
			v[i] = elem[p[i]];
		}
	}

	// Copy from part of a longer vector x
	template <typename T>
	void Vector<T>::copy(const Vector<T> &v, const size_t start_idx)
	{
		assert(v._length >= start_idx + _length);
		memcpy(_elements, v._elements + start_idx, _length*sizeof(T));
	}

	// vector concatenation: z[] = [ x[]; y[] ] 
	template <typename T>
	void Vector<T>::concat(const Vector<T> &x, const Vector<T> &y)
	{
		assert(x._length + y._length == _length);
		memcpy(_elements, x._elements, x._length*sizeof(T));
		memcpy(_elements + x._length, y._elements, y._length*sizeof(T));
	}

	// split vector into two z[] = [ x[]; y[] ] 
	template <typename T>
	void Vector<T>::split(Vector<T> &x, Vector<T> &y) const
	{
		assert(x._length + y._length == _length);
		memcpy(x._elements, _elements, x._length * sizeof(T));
		memcpy(y._elements, _elements + x._length, y._length * sizeof(T));
	}

	// Set every element to zero
	template <typename T>
	void Vector<T>::fill_zero()
	{
		memset(_elements, 0, _length*sizeof(T));
	}

	// Fill Vector with random numbers in the interval [a, b)
	template <typename T>
	void Vector<T>::fill_rand(const T a, const T b)
	{
		std::mt19937_64 reng(std::random_device{}());
		std::uniform_real_distribution<T> unif(a, b);
		auto rgen = std::bind(unif, reng);
		std::generate(_elements, _elements + _length, rgen);
	}

	// Fill every element with a constant 
	template <typename T>
	void Vector<T>::fill(const T c)
	{
		auto elem = _elements;
		auto len = _length;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++) {
			elem[i] = c;
		}
	}

	// Add a constant to each element
	template <typename T>
	void Vector<T>::add(const T c)
	{
		auto elem = _elements;
		auto len = _length;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++) {
			elem[i] += c;
		}
	}

	// Return sum of elements
	template <typename T>
	T Vector<T>::sum() const
	{
		auto elem = _elements;
		auto len = _length;
		T s = 0;

		#pragma omp parallel for reduction(+:s)
		for (int64_t i = 0; i < len; i++) {
			s += elem[i];
		}
		return s;
	}

	// Return the maximum value. No OpenMP used
	template <typename T>
	T Vector<T>::max() const
	{
		auto elem = _elements;
		auto len = _length;
		T a = elem[0];
		for (size_t i = 1; i < len; i++) {
			if (elem[i] > a) a = elem[i];
		}
		return a;
	}

	// Return the minimum value. No OpenMP used
	template <typename T>
	T Vector<T>::min() const
	{
		auto elem = _elements;
		auto len = _length;
		T a = elem[0];
		for (size_t i = 1; i < len; i++) {
			if (elem[i] < a) a = elem[i];
		}
		return a;
	}

	// Return 1-norm: sum_i abs(x[i])
	template <typename T>
	T Vector<T>::norm1() const
	{
		auto elem = _elements;
		auto len = _length;
		T a = 0;

		#pragma omp parallel for reduction(+:a)
		for (int64_t i = 0; i < len; i++) {
			a += fabs(elem[i]);
		}
		return a;
	}

	// Returns 2-norm (Euclidean norm) of the vector
	template <typename T>
	T Vector<T>::norm2() const
	{
		auto elem = _elements;
		auto len = _length;
		T sum = 0;

		#pragma omp parallel for reduction(+:sum)
		for (int64_t i = 0; i < len; i++) {
			sum += pow(elem[i], 2);
		}
		return sqrt(sum);
	}

	// Return infinity norm: max_i abs(x[i])
	template <typename T>
	T Vector<T>::normInf() const
	{
		auto elem = _elements;
		auto len = _length;
		T a = 0, xi_abs;
		for (size_t i = 0; i < len; i++)
		{
			xi_abs = fabs(elem[i]);
			if (xi_abs > a) a = xi_abs;
		}
		return a;
	}

	// Return a vector of signs in {-1, 0, +1}
	template <typename T>
	void Vector<T>::sign(Vector<T> &s) const
	{
		assert(_length == s._length);
		auto elem = _elements;
		auto s_elem = s._elements;
		auto len = _length;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++) {
			s_elem[i] = elem[i] > 0 ? 1.0f : elem[i] < 0 ? -1 : 0.0f;
		}
	}

	// Convert to vector of signs in {-1, 0, +1}
	template <typename T>
	void Vector<T>::to_sign()
	{
		auto elem = _elements;
		auto len = _length;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++) {
			elem[i] = elem[i] > 0 ? 1.0f : elem[i] < 0 ? -1.0f : 0.0f;
		}
	}

	// Count number of positive elements
	template <typename T>
	size_t Vector<T>::count_positive() const
	{
		auto elem = _elements;
		auto len = _length;
		size_t count = 0;

		#pragma omp parallel for reduction(+:count)
		for (int64_t i = 0; i < len; i++) {
			count += elem[i] > 0 ? 1 : 0;
		}
		return count;
	}

	// Count number of negative elements
	template <typename T>
	size_t Vector<T>::count_negative() const
	{
		auto elem = _elements;
		auto len = _length;
		size_t count = 0;

		#pragma omp parallel for reduction(+:count)
		for (int64_t i = 0; i < len; i++) {
			count += elem[i] < 0 ? 1 : 0;
		}
		return count;
	}

	// Scale itself by a constant alpha
	template <typename T>
	void Vector<T>::scale(const T alpha)
	{
		auto elem = _elements;
		auto len = _length;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++) {
			elem[i] *= alpha;
		}
	}

	// Return inner product with vector x
	template <typename T>
	T Vector<T>::dot(const Vector<T> &x) const
	{
		assert(_length == x._length);
		auto elem = _elements;
		auto x_elem = x._elements;
		auto len = x._length;
		T a = 0;

		#pragma omp parallel for reduction(+:a)
		for (int64_t i = 0; i < len; i++) {
			a += elem[i] * x_elem[i];
		}
		return a;
	}

	template <typename T>
	T Vector<T>::dot(const Vector<T> &x, const Vector<T> &y)
	{
		return x.dot(y);
	}

	// y = alpha * x + y
	template <typename T>
	void Vector<T>::axpy(const T alpha, const Vector<T> &x)
	{
		assert(x._length == _length);
		auto x_elem = x._elements;
		auto y_elem = _elements;
		auto len = _length;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++) {
			y_elem[i] += alpha*x_elem[i];
		}
	}

	// y = alpha * x + y
	template <typename T>
	void Vector<T>::axpy(const T alpha, const Vector<T> &x, Vector<T> &y)
	{
		assert(x._length == y._length);
		auto x_elem = x._elements;
		auto y_elem = y._elements;
		auto len = x._length;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++) {
			y_elem[i] += alpha*x_elem[i];
		}
	}

	/* A tricky mistake, this always shadowed by the previous static method!!!
	// this = alpha * x + y
	template <typename T>
	void Vector<T>::axpy(const T alpha, const Vector<T> &x, const Vector<T> &y)
	{
		this->copy(y);
		this->axpy(alpha, x);
	}
	*/

	// Elementwise absolute value: vabs[i] = abs( v[i] )
	template <typename T>
	void Vector<T>::elem_abs(Vector<T> &v) const
	{
		assert(v._length == _length);
		auto elem = _elements;
		auto v_elem = v._elements;
		auto len = _length;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++) {
			v_elem[i] = fabs(elem[i]);
		}
	}

	// Elementwise inversion: inv[i] = 1/v[i]
	template <typename T>
	void Vector<T>::elem_inv(Vector<T> &v) const
	{
		assert(v._length == _length);
		auto elem = _elements;
		auto v_elem = v._elements;
		auto len = _length;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++) {
			v_elem[i] = 1 / elem[i];
		}
	}

	// Elementwise square root: vsqrt[i] = sqrt( v[i] )
	template <typename T>
	void Vector<T>::elem_sqrt(Vector<T> &v) const
	{
		assert(v._length == _length);
		auto elem = _elements;
		auto v_elem = v._elements;
		auto len = _length;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++) {
			v_elem[i] = sqrt(elem[i]);
		}
	}

	// Elementwise exponential: vexp[i] = exp( v[i] )
	template <typename T>
	void Vector<T>::elem_exp(Vector<T> &v) const
	{
		assert(v._length == _length);
		auto elem = _elements;
		auto v_elem = v._elements;
		auto len = _length;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++) {
			v_elem[i] = exp(elem[i]);
		}
	}

	// Elementwise natural logarithm: vlog[i] = log( v[i] )
	template <typename T>
	void Vector<T>::elem_log(Vector<T> &v) const
	{
		assert(v._length == _length);
		auto elem = _elements;
		auto v_elem = v._elements;
		auto len = _length;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++) {
			v_elem[i] = log(elem[i]);
		}
	}

	// Elementwise exponential after minus 1: vexpm1[i] = exp(v[i]) - 1 
	template <typename T>
	void Vector<T>::elem_expm1(Vector<T> &v) const
	{
		assert(v._length == _length);
		auto elem = _elements;
		auto v_elem = v._elements;
		auto len = _length;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++) {
			v_elem[i] = expm1(elem[i]);
		}
	}

	// Elementwise natural logarithm of 1 plus: vlog1p[i] = log( 1 + v[i] )
	template <typename T>
	void Vector<T>::elem_log1p(Vector<T> &v) const
	{
		assert(v._length == _length);
		auto elem = _elements;
		auto v_elem = v._elements;
		auto len = _length;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++) {
			v_elem[i] = log1p(elem[i]);
		}
	}

	// Elememtwise hinge loss: h[i] = max{0, 1 - yXw[i]}
	template <typename T>
	void Vector<T>::elem_hinge(Vector<T> &h) const
	{
		assert(_length == h._length);
		auto elem = _elements;
		auto h_elem = h._elements;
		auto len = _length;
		T diff = 0;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++)
		{
			diff = 1 - elem[i];
			h_elem[i] = diff > 0 ? diff : 0;
		}
	}

	// Elementwise clip: b[i] = min( ub, max(lb, a[i]))
	template <typename T>
	void Vector<T>::elem_clip(const T lb, const T ub, Vector<T> &b) const
	{
		assert(_length == b._length);
		auto elem = _elements;
		auto b_elem = b._elements;
		auto len = _length;
		T val = 0;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++)
		{
			val = elem[i];
			b_elem[i] = val > ub ? ub : val < lb ? lb : val;
		}
	}

	// Elementwise addition: z[i] = x[i] + y[i]
	template <typename T>
	void Vector<T>::elem_add(const Vector<T> &x, const Vector<T> &y, Vector<T> &z)
	{
		assert(x._length == y._length && x._length == z._length);
		auto x_elem = x._elements;
		auto y_elem = y._elements;
		auto z_elem = z._elements;
		auto len = x._length;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++) {
			z_elem[i] = x_elem[i] + y_elem[i];
		}
	}

	// Elementwise subtraction: z[i] = x[i] - y[i]
	template <typename T>
	void Vector<T>::elem_subtract(const Vector<T> &x, const Vector<T> &y, Vector<T> &z)
	{
		assert(x._length == y._length && x._length == z._length);
		auto x_elem = x._elements;
		auto y_elem = y._elements;
		auto z_elem = z._elements;
		auto len = x._length;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++) {
			z_elem[i] = x_elem[i] - y_elem[i];
		}
	}

	// Elementwise multiplication: z[i] = x[i] * y[i]
	template <typename T>
	void Vector<T>::elem_multiply(const Vector<T> &x, const Vector<T> &y, Vector<T> &z)
	{
		assert(x._length == y._length && x._length == z._length);
		auto x_elem = x._elements;
		auto y_elem = y._elements;
		auto z_elem = z._elements;
		auto len = x._length;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++) {
			z_elem[i] = x_elem[i] * y_elem[i];
		}
	}

	// Elementwise division: z[i] = x[i] / y[i]
	template <typename T>
	void Vector<T>::elem_divide(const Vector<T> &x, const Vector<T> &y, Vector<T> &z)
	{
		assert(x._length == y._length && x._length == z._length);
		auto x_elem = x._elements;
		auto y_elem = y._elements;
		auto z_elem = z._elements;
		auto len = x._length;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++) {
			z_elem[i] = x_elem[i] / y_elem[i];
		}
	}

	// Elementwise min: z[i] = min( x[i], y[i] )
	template <typename T>
	void Vector<T>::elem_min(const Vector<T> &x, const Vector<T> &y, Vector<T> &z)
	{
		assert(x._length == y._length && x._length == z._length);
		auto x_elem = x._elements;
		auto y_elem = y._elements;
		auto z_elem = z._elements;
		auto len = x._length;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++) {
			z_elem[i] = fmin(x_elem[i], y_elem[i]);
		}
	}

	// Elementwise max: z[i] = max( x[i], y[i] )
	template <typename T>
	void Vector<T>::elem_max(const Vector<T> &x, const Vector<T> &y, Vector<T> &z)
	{
		assert(x._length == y._length && x._length == z._length);
		auto x_elem = x._elements;
		auto y_elem = y._elements;
		auto z_elem = z._elements;
		auto len = x._length;

		#pragma omp parallel for
		for (int64_t i = 0; i < len; i++) {
			z_elem[i] = fmax(x_elem[i], y_elem[i]);
		}
	}


	// Member functions of SubVector, a mask to memory in Vector
	template <typename T>
	SubVector<T>::SubVector(const Vector<T> &v, const size_t start_idx, const size_t length)
	{
		this->recast(v, start_idx, length);
	}

	template <typename T>
	void SubVector<T>::recast(const Vector<T> &v, const size_t start_idx, const size_t length)
	{
		assert(start_idx + length <= v.length());
		this->_length = length;
		this->_elements = v.data() + start_idx;
		this->_need_cleanup = false;
	}

	// instantiates template with two float types
	template class Vector<float>;
	template class Vector<double>;
	template class SubVector<float>;
	template class SubVector<double>;
}
