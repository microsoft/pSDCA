// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include<string>

namespace randalgms
{
	void display_header(std::string algm_info);
	void display_progress(int epoch, double primal_obj, double dual_obj, double time_from_start);
	void display_header_stage(std::string algm_info);
	void display_progress_stage(int stage, int epoch, double primal_obj, double dual_obj, double time_from_start);
}
