// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <algorithm>
#include <vector>
#include "utils.h"

void main(int argc, char* argv[])
{
	using floatT = float;

	vector<floatT> labels;
	vector<floatT> weights;
	vector<floatT> values;
	vector<size_t> colidx;
	vector<size_t> rowptr;

	//string filename;
	//filename = argv[1];
	//filename = "C:/Data/LIBSVMdata/covtype.libsvm.binary.scale";	//  4 seconds
	//filename = "C:/Data/LIBSVMdata/splice1percent100_1";			// 12 seconds
	//filename = "C:/Data/LIBSVMdata/ads1percent100_1";				//  4 seconds
	//filename = "C:/Data/LIBSVMdata/rcv1_test.binary";				// 50 seconds
	//filename = "C:/Data/LIBSVMdata/rcv1_train.binary";			//  2 seconds
	//filename = "C:/Data/LIBSVMdata/news20.binary";				//  6 seconds
	//filename = "C:/Data/MSFT_Ads/TrainingFIDOnly2.10G";			//  6 minutes
	//filename = "//gcr/scratch/AZ-USCentral/lixiao/Data/MSFT_Ads/TrainingFIDOnly2.10G";	// 25 minutes

	const string files[] =
	{
		//"C:/Data/LIBSVMdata/ads1percent100_1",
		//"C:/Data/LIBSVMdata/rcv1_test.binary",
		//"C:/Data/LIBSVMdata/news20.binary"
		"C:/Data/part-00000.txt",
		"C:/Data/part-r-00001.txt",
	};

	for (auto & filename : files) {
		size_t n_examples = load_datafile(filename, labels, weights, values, colidx, rowptr, false, true, true);
	}

	std::list<string> filelist;
	filelist.push_back("C:/Data/part-00000.txt");
	filelist.push_back("C:/Data/part-r-00001.txt");
	size_t n_examples = load_datafile(filelist, labels, weights, values, colidx, rowptr, false, true, true);
}