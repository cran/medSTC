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

#include "params.h"
#include <string>
#include <stdlib.h>
#include <stdio.h>
//using namespace std;

Params::Params(void)
{
	train_filename = new char[512];
	test_filename = new char[512];
	SUPERVISED = 0;
	PRIMALSVM = 1;
	DELTA_ELL = 360;
}

Params::~Params(void)
{
	delete[] train_filename;
	delete[] test_filename;
}

void Params::read_settings(char* filename)
{
	FILE* fileptr;
	
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "supervised %d\n", &SUPERVISED);
	fscanf(fileptr, "primal svm %d\n", &PRIMALSVM);
	fscanf(fileptr, "var max iter %d\n", &VAR_MAX_ITER);
	fscanf(fileptr, "var convergence %f\n", &VAR_CONVERGED);
	fscanf(fileptr, "em max iter %d\n", &EM_MAX_ITER);
	fscanf(fileptr, "em convergence %f\n", &EM_CONVERGED);

	fscanf(fileptr, "model C %f\n", &INITIAL_C);
	fscanf(fileptr, "delta ell %f\n", &DELTA_ELL);
	fscanf(fileptr, "lambda %f\n", &LAMBDA);
	fscanf(fileptr, "rho %f\n", &RHO);
	fscanf(fileptr, "svm_alg_type %d\n", &SVM_ALGTYPE);

	fscanf(fileptr, "train_file: %s\n", train_filename);
	fscanf(fileptr, "test_file: %s\n", test_filename);
	fscanf(fileptr, "class_num: %d\n", &NLABELS);

	fclose(fileptr);
}

void Params::save_settings(char* filename)
{
	FILE* fileptr;
	
	fileptr = fopen(filename, "w");
	fprintf(fileptr, "supervised %d\n", SUPERVISED);
	fprintf(fileptr, "primal svm %d\n", PRIMALSVM);
	fprintf(fileptr, "var max iter %d\n", VAR_MAX_ITER);
	fprintf(fileptr, "var convergence %.10f\n", VAR_CONVERGED);
	fprintf(fileptr, "em max iter %d\n", EM_MAX_ITER);
	fprintf(fileptr, "em convergence %.10f\n", EM_CONVERGED);

	fprintf(fileptr, "model C %.5f\n", INITIAL_C);
	fprintf(fileptr, "delta ell %.5f\n", DELTA_ELL);
	fprintf(fileptr, "lambda %.5f\n", LAMBDA);
	fprintf(fileptr, "rho %.5f\n", RHO);
	fprintf(fileptr, "svm_alg_type %d\n", SVM_ALGTYPE);

	fprintf(fileptr, "train_file: %s\n", train_filename);
	fprintf(fileptr, "test_file: %s\n", test_filename);
	fprintf(fileptr, "class_num: %d\n", NLABELS);

	fclose(fileptr);
}
