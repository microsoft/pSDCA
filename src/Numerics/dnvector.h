// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <cstdint>

namespace numerics {

	template <typename T> 
	class Vector
	{
	protected:
		size_t _length;			// Length of vector
		T *_elements;			// Pointer to memory storing the elements
		bool _need_cleanup;		// Indicator to delete[] elements

		// Default constructor for creating the SubVector class (masks only)
		Vector() = default;

	public:
		Vector(const size_t n);				// construct all-zero vector of length n
		Vector(const Vector<T> &v);			// copy constructor
		// the following two functions avoid the needs for using two template parameters
		Vector(const std::vector<float> &v);	// construct from std::vector<float>
		Vector(const std::vector<double> &v);	// construct from std::vector<double>
		Vector(const Vector<T> &v, const std::vector<size_t> &indices);
		// Destructor, also called by derived class SubVector without cleaning up memory
		~Vector();

		std::vector<T> to_vector();	// convert to std::vector<T>, using C++11 std::move()
		// find pre-permutation vector v[p[i]] = this[i]
		void pre_permute(const std::vector<size_t> &p, Vector<T> &v);
		// find post-permutation vector v[i] = this[p[i]]
		void post_permute(const std::vector<size_t> &p, Vector<T> &v);

		// Do not allow copy assignment constructor due to subVector cleanup issues!
		Vector<T>& operator=(const Vector<T> &v) = delete;

		// Returns length of vector
		inline size_t length() const { return _length; }
		// Return a direct pointer to the data memory (e.g., used as MPI buffe)
		inline T* data() const { return _elements; }

		// Overloading operator [], use 0-based indexing, inlined for speed
		inline T& operator[] (const size_t i) { return _elements[i]; }
		inline const T& operator[] (const size_t i) const { return _elements[i]; }
		// To remove compilation warning of converting int64_t to size_t when using OpenMP
		inline T& operator[] (const int64_t i) { return _elements[i]; }
		inline const T& operator[] (const int64_t i) const { return _elements[i]; }

		// Copy from (part of a longer) vector x
		void copy(const Vector<T> &v, const size_t start_idx=0);
		// vector concatenation: z[] = [ x[]; y[] ] 
		void concat(const Vector<T> &x, const Vector<T> &y);
		// split vector into two z[] = [ x[]; y[] ] 
		void split(Vector<T> &x, Vector<T> &y) const;
		// Set every element to zero
		void fill_zero();
		// Fill Vector with random numbers in the interval [a, b)
		void fill_rand(const T a, const T b);
		// Fill every element with a constant 
		void fill(const T c);
		// Add a constant to each element
		void add(const T c);

		// Return sum of elements
		T sum() const;
		// Return the maximum value of elements
		T max() const;
		// Return the minimum value of elements
		T min() const;
		// Return 1-norm of vector
		T norm1() const;
		// Returns 2-norm (Euclidean norm) of the vector
		T norm2() const;
		// Return the infinite norm 
		T normInf() const;

		// Return a vector of signs in {-1, 0, +1}
		void sign(Vector<T> &s) const;
		// Convert elements to signs in {-1, 0, +1}
		void to_sign(); 
		// Count number of positive elements
		size_t count_positive() const;
		// Count number of negative elements
		size_t count_negative() const;

		// Scale itself by a constant alpha
		void scale(const T alpha);
		// Return inner product with vector x
		T dot(const Vector<T> &x) const;
		static T dot(const Vector<T> &x, const Vector<T> &y);
		// y = alpha * x + y
		void axpy(const T alpha, const Vector<T> &x);
		static void axpy(const T alpha, const Vector<T> &x, Vector<T> &y);
		// this = alpha * x + y		// this is completely ignored as call always to the above static method!!! 
		//void axpy(const T alpha, const Vector<T> &x, const Vector<T> &y);

		// Elementwise absolute value: vabs[i] = abs( v[i] )
		void elem_abs(Vector<T> &vabs) const;
		// Elementwise inversion: inv[i] = 1/v[i]
		void elem_inv(Vector<T> &vinv) const;
		// Elementwise square root: vsqrt[i] = sqrt( v[i] )
		void elem_sqrt(Vector<T> &vsqrt) const;
		// Elementwise exponential: vexp[i] = exp( v[i] )
		void elem_exp(Vector<T> &vexp) const;
		// Elementwise natural logarithm: vlog[i] = log( v[i] )
		void elem_log(Vector<T> &vlog) const;
		// Elementwise exponential after minus 1: vexpm1[i] = exp(v[i]) - 1 
		void elem_expm1(Vector<T> &vexpm1) const;
		// Elementwise natural logarithm of 1 plus: vlog1p[i] = log( 1 + v[i] )
		void elem_log1p(Vector<T> &vlog1p) const;
		// Elememtwise hinge loss: h[i] = max{0, 1 - yXw[i]}
		void elem_hinge(Vector<T> &h) const;
		// Elementwise clip: b[i] = min( ub, max(lb, a[i]))
		void elem_clip(const T lb, const T ub, Vector<T> &b) const;

		// Elementwise addition: z[i] = x[i] + y[i]
		static void elem_add(const Vector<T> &x, const Vector<T> &y, Vector<T> &z);
		// Elementwise subtraction: z[i] = x[i] - y[i]
		static void elem_subtract(const Vector<T> &x, const Vector<T> &y, Vector<T> &z);
		// Elementwise multiplication: z[i] = x[i] * y[i]
		static void elem_multiply(const Vector<T> &x, const Vector<T> &y, Vector<T> &z);
		// Elementwise division: z[i] = x[i] / y[i]
		static void elem_divide(const Vector<T> &x, const Vector<T> &y, Vector<T> &z);
		// Elementwise min: z[i] = min( x[i], y[i] )
		static void elem_min(const Vector<T> &x, const Vector<T> &y, Vector<T> &z);
		// Elementwise max: z[i] = max( x[i], y[i] )
		static void elem_max(const Vector<T> &x, const Vector<T> &y, Vector<T> &z);
	};


	// This class defines a mask for a contiguous part of a Vector
	// !!! Memory problem if using SubVector after referred Vector is desconstructed!
	template <typename T>
	class SubVector final : public Vector<T>
	{
	public:
		// Use 0-based indexing for starting index
		SubVector(const Vector<T> &v, const size_t start_idx, const size_t sub_length);
		void recast(const Vector<T> &v, const size_t start_idx, const size_t sub_length);

		// no memory allocated, so do not allow copy constructor and copy-assignment operator
		SubVector() = delete;
		SubVector(const SubVector<T> &) = delete;
		SubVector& operator=(const SubVector<T> &) = delete;
	};
}
