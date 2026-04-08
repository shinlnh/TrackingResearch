/* 
 * Struck: Structured Output Tracking with Kernels
 * 
 * Code to accompany the paper:
 *   Struck: Structured Output Tracking with Kernels
 *   Sam Hare, Amir Saffari, Philip H. S. Torr
 *   International Conference on Computer Vision (ICCV), 2011
 * 
 * Copyright (C) 2011 Sam Hare, Oxford Brookes University, Oxford, UK
 * 
 * This file is part of Struck.
 * 
 * Struck is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Struck is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Struck.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include "Config.h"

#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

/* input by user
c_seqPath = argv[1];
c_seqName = argv[2];
c_resPath = argv[3];
c_width = argv[4];
c_height = argv[5];
*/

Config::Config(const std::string& path)
{
	SetDefaults();
	
	ifstream f(path.c_str());
	if (!f)
	{
		cout << "error: could not load config file: " << path << endl;
		return;
	}

	string featureName, kernelName;
	featureName = "haar";
	kernelName = "gaussian";

	double param = 0.2;
	FeatureKernelPair fkp;

	fkp.kernel = kKernelTypeGaussian;
	fkp.feature = kFeatureTypeHaar;
	fkp.params.push_back(param);

	features.push_back(fkp);
	
	string line, name, tmp;
	while (getline(f, line))
	{
		istringstream iss(line);
		iss >> name >> tmp;
		
		// skip invalid lines and comments
		if (iss.fail() || tmp != "=" || name[0] == '#') continue;
		
		if      (name == "seed") iss >> seed;
		else if (name == "quietMode") iss >> quietMode;
		else if (name == "debugMode") iss >> debugMode;
		else if (name == "sequenceBasePath") iss >> sequenceBasePath;
		else if (name == "sequenceName") iss >> sequenceName;
		else if (name == "resultsPath") iss >> resultsPath;
		else if (name == "frameWidth") iss >> frameWidth;
		else if (name == "frameHeight") iss >> frameHeight;
		else if (name == "seed") iss >> seed;
		else if (name == "searchRadius") iss >> searchRadius;
		else if (name == "svmC") iss >> svmC;
		else if (name == "svmBudgetSize") iss >> svmBudgetSize;
		else if (name == "startFrame") iss >> startFrame;
		else if (name == "endFrame") iss >> endFrame;
	}
	
}

void Config::SetDefaults()
{

	quietMode = false;
	debugMode = false;
	
	/*
	sequenceBasePath = "sequences";
	sequenceName = "girl";
	
	
	frameWidth = 320;
	frameHeight = 240;
	*/
	resultsPath = "./Struck/tmpStruckRes.txt";
	
	seed = 0;
	searchRadius = 30;
	svmC = 100.0;
	svmBudgetSize = 100;
	
	features.clear();
}

std::string Config::FeatureName(FeatureType f)
{
	switch (f)
	{
	case kFeatureTypeRaw:
		return "raw";
	case kFeatureTypeHaar:
		return "haar";
	case kFeatureTypeHistogram:
		return "histogram";
	default:
		return "";
	}
}

std::string Config::KernelName(KernelType k)
{
	switch (k)
	{
	case kKernelTypeLinear:
		return "linear";
	case kKernelTypeGaussian:
		return "gaussian";
	case kKernelTypeIntersection:
		return "intersection";
	case kKernelTypeChi2:
		return "chi2";
	default:
		return "";
	}
}

ostream& operator<< (ostream& out, const Config& conf)
{
	out << "config:" << endl;
	out << "  quietMode          = " << conf.quietMode << endl;
	out << "  debugMode          = " << conf.debugMode << endl;
	out << "  sequenceBasePath   = " << conf.sequenceBasePath << endl;
	out << "  sequenceName       = " << conf.sequenceName << endl;
	out << "  resultsPath        = " << conf.resultsPath << endl;
	out << "  frameWidth         = " << conf.frameWidth << endl;
	out << "  frameHeight        = " << conf.frameHeight << endl;
	out << "  seed               = " << conf.seed << endl;
	out << "  searchRadius       = " << conf.searchRadius << endl;
	out << "  svmC               = " << conf.svmC << endl;
	out << "  svmBudgetSize      = " << conf.svmBudgetSize << endl;
	out << "  startFrame      = " << conf.startFrame << endl;
	out << "  endFrame      = " << conf.endFrame << endl;
	
	for (int i = 0; i < (int)conf.features.size(); ++i)
	{
		out << "  feature " << i << endl;
		out << "    feature: " << Config::FeatureName(conf.features[i].feature) << endl;
		out << "    kernel:  " << Config::KernelName(conf.features[i].kernel) <<endl;
		if (conf.features[i].params.size() > 0)
		{
			out << "    params: ";
			for (int j = 0; j < (int)conf.features[i].params.size(); ++j)
			{
				out << " " << conf.features[i].params[j];
			}
			out << endl;
		}
	}
	
	return out;
}
