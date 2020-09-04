// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <string>

// split data file into smaller files
size_t split_datafile(const std::string dataname, const unsigned int n);
size_t split_data_sorted(const std::string dataname, const unsigned int n);