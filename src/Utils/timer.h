// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <chrono>

using HighResClock = std::chrono::high_resolution_clock;
using FloatSeconds = std::chrono::duration<float>;

class HighResTimer
{
private: 
	HighResClock::time_point t_start, t_last;
public:
	HighResTimer() { this->start(); };
	void start() { this->reset(); };
	void reset() { 
		t_start = HighResClock::now(); 
		t_last = t_start; 
	};
	float seconds_from_start() { 
		t_last = HighResClock::now();  
		FloatSeconds t_elpsd = t_last - t_start;
		return t_elpsd.count();
	}
	float seconds_from_last() {
		auto t_now = HighResClock::now();
		FloatSeconds t_elpsd = t_now - t_last;
		t_last = t_now;
		return t_elpsd.count();
	}
};