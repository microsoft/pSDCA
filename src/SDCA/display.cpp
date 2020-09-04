// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <string>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>

#include "display.h"

namespace randalgms {

	void display_header(std::string algm_info)
	{
		std::cout << std::endl << algm_info << std::endl;
		std::cout << "epoch  primal     dual       gap        time" << std::endl;
	}

	void display_progress(int epoch, double primal_obj, double dual_obj, double time_from_start)
	{
		std::cout << std::setw(3) << epoch
			<< std::fixed << std::setprecision(6)
			<< std::setw(10) << primal_obj
			<< std::setw(10) << dual_obj
			<< std::scientific << std::setprecision(3)
			<< std::setw(12) << primal_obj - dual_obj
			<< std::fixed << std::setprecision(3)
			<< std::setw(9) << time_from_start
			<< std::endl;
	}

	void display_header_stage(std::string algm_info)
	{
		std::cout << std::endl << algm_info << std::endl;
		std::cout << "stage epoch  primal     dual       gap        time" << std::endl;
	}

	void display_progress_stage(int stage, int epoch, double primal_obj, double dual_obj, double time_from_start)
	{
		std::cout << std::setw(3) << stage
			<< std::setw(6) << epoch
			<< std::fixed << std::setprecision(6)
			<< std::setw(10) << primal_obj
			<< std::setw(10) << dual_obj
			<< std::scientific << std::setprecision(3)
			<< std::setw(12) << primal_obj - dual_obj
			<< std::fixed << std::setprecision(3)
			<< std::setw(9) << time_from_start
			<< std::endl;
	}
}
