// (C) Copyright 2011, Jun Zhu (junzhu [at] cs [dot] cmu [dot] edu)

// This file is part of MedSTC.

// MedSTC is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// MedSTC is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#pragma once

class Params
{
public:
	Params(void);
public:
	~Params(void);

	void read_settings(char *filename);
	void save_settings(char *filename);
public:
	int LAG;

	float EM_CONVERGED;
	int EM_MAX_ITER;
	float INITIAL_C;
	int NTOPICS;
	int NLABELS;
	int NFOLDS;
	float DELTA_ELL;
	float LAMBDA;
	float RHO;
	int PRIMALSVM;

	int VAR_MAX_ITER;
	float VAR_CONVERGED;

	int SVM_ALGTYPE;       // the algorithm type for SVM
	int SUPERVISED;

	char *train_filename;   // the file names of training & testing data sets
	char *test_filename;
};
