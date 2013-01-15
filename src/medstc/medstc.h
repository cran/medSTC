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


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>

#include "utils.h"
#include "cokus.h"
#include "params.h"
#include "corpus.h"
#include "lbfgscpp.h"
#include "../svm_multiclass/svm_struct_api.h"
#include "../svm_multiclass/svm_struct_learn.h"
#include "../svm_multiclass/svm_struct_common.h"

#define myrand() (double) (((unsigned long) randomMT()) / 4294967296.)
#define NUM_INIT 1

#define MAX_ALPHA_ITER 1000
#define MIN_BETA 1e-30
#define LAG 200

class MedSTC
{
public:
	MedSTC(void);
	MedSTC(int nK,int nLabelNum,int nNumTerms,int nDim,
			double dDeltaEll,double dLambda,double dRho,double dGamma,double dC,
			double dLogLoss,double dB,double dPoisOffset,double dsvm_primalobj, 
			double **dLogProbW, double *dMu,double *dEta,char* directory);
	
	
public:
	~MedSTC(void);

public:
	int train(char* start, char* directory, Corpus* pC, Params* param);

	bool dict_learn(Corpus *pC, double **theta, double ***s, Params *param, bool bInit = true);
	double sparse_coding(char* model_dir, Corpus* pC, Params *param);
	double sparse_coding(Document *doc, const int &docIx,
						Params *param, double* theta, double **s);
	void init_phi(Document *doc, double **phi, double *theta, Params *param);

	void learn_svm(char *model_dir, const double &dC, const double &dEll);

	double fdf_beta(Corpus *pC, double **theta, double ***s, double *g);
	void get_param(double *w, const int &nK, const int &nTerms);
	void set_param(double *w, const int &nK, const int &nTerms);
	void project_beta( double *beta, const int &nTerms );
	void project_beta2( double *beta, const int &nTerms, const double &dZ, const double &epsilon );

	void save_theta(char* filename, double** gamma, int num_docs, int num_topics);
	double save_prediction(char *filename, Corpus *pC);

	int predict(double *theta);
	void predict_scores(double* scores,double *theta);
	void predictTest(Corpus* pC, Params *param);
	void loss_aug_predict(Document *doc, double *zbar_mean);
	double loss(const int &y, const int &gnd);

	void free_model();
	void save_model(char*filename, double dTime);
	void new_model(int, int, int, int, double);
	void load_model(char* model_root);

	void set_init_param(STRUCT_LEARN_PARM *struct_parm,
					LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm,
					int *alg_type);
	void svmStructSolver(char *dataFileName, Params *param, double *res);
	void outputLowDimData(char *filename, Corpus *pC, double **theta);
	void readLowDimData(char *filename, int &nDataNum);

	void write_word_assignment(FILE* f, Document* doc, double** phi);
	void get_train_filename(char *buff, char *dir, Params *param);
	void get_test_filename(char *buff, char *dir, Params *param);

	void init_param(Corpus *pC);
	void release_param();
public:
	int m_nK;
	int m_nLabelNum;
	int m_nNumTerms;
	double **m_dLogProbW;
	double m_dDeltaEll;   // adjustable 0/ell loss function
	double m_dLambda;
	double m_dRho;
	double m_dGamma;
	double m_dC;
	double m_dLogLoss;
//private:
	double *m_dMu;
	double *m_dEta;
	double m_dB;
	double m_dPoisOffset;
	int m_nDim;
	double m_dsvm_primalobj;
	char *m_directory;

// parameters for fast computing.
private:
	CLBFGSCPP *m_pLBFGS;
	double *mu_;
	double *x_;
	double *g_;
	double *diag_;
	double *sold_;
	double *initEta_;
	int *label_;
	double **theta_;
};
