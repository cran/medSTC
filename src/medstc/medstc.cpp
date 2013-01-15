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

#include "medstc.h"
#include <vector>
#include <string>
#include "../svmlight/svm_common.h"
#include "params.h"
#include "lbfgscpp.h"
#include <R.h>
//using namespace std;

MedSTC::MedSTC(void)
{
	m_dLogProbW   = NULL;
	m_dMu         = NULL;
	m_dEta        = NULL;
	m_dB          = 0;
	m_dPoisOffset = 0.001;

	m_dLambda     = 1;
	m_dGamma      = m_dLambda;
	m_dRho        = 1;

	sold_ = NULL;
	mu_   = NULL;
	x_    = NULL;
	g_    = NULL;
	diag_ = NULL;
	initEta_ = NULL;
	m_pLBFGS = NULL;
}
MedSTC::MedSTC(int nK,int nLabelNum,int nNumTerms,int nDim,
			double dDeltaEll,double dLambda,double dRho,double dGamma,double dC,
			double dLogLoss,double dB,double dPoisOffset,double dsvm_primalobj, 
			double **dLogProbW, double *dMu,double *dEta, char* directory){
				m_nK=nK;
				m_nLabelNum=nLabelNum;
				m_nNumTerms=nNumTerms;
				m_dDeltaEll=dDeltaEll;
				m_dLambda=dLambda;
				m_dRho=dRho;
				m_dGamma=dGamma;
				m_dC=dC;
				m_dLogLoss=dLogLoss;
				m_dB = dB;
				m_dPoisOffset=dPoisOffset;
				m_nDim=nDim;
				m_dsvm_primalobj=dsvm_primalobj;
				m_dLogProbW = (double**)malloc(sizeof(double*)*nNumTerms);
				m_dEta = (double*)malloc(sizeof(double) * nK * nLabelNum);
				m_dMu = (double*)malloc(sizeof(double) * nDim * nLabelNum);
				
				
				for (int i=0; i<nNumTerms; i++) {
					m_dLogProbW[i] = (double*)malloc(sizeof(double)*nK);
					for (int j=0; j<nK; j++) 
						m_dLogProbW[i][j] = dLogProbW[i][j];
				}
				for ( int i=0; i<nK; i++ ) {	
					for (int j=0; j<nLabelNum; j++) 
						m_dEta[i*nLabelNum + j] = dEta[i*nLabelNum + j];
				}
				for (int i=0; i<nDim; i++)
					for (int j=0; j<nLabelNum; j++)
						m_dMu[i*nLabelNum + j] = dMu[i*nLabelNum + j];

				 m_directory = new char[512];
				 strcpy(m_directory,directory);
				 sold_ = NULL;
				 mu_   = NULL;
				 x_    = NULL;
				 g_    = NULL;
				 diag_ = NULL;
				 initEta_ = NULL;
				 m_pLBFGS = NULL;

			
			}

MedSTC::~MedSTC(void)
{
	free_model();
}

// find the loss-augmented prediction for one document.
void MedSTC::loss_aug_predict(Document *doc, double *theta)
{
	doc->lossAugLabel = -1;
	double dMaxScore = 0;
	int etaIx = 0;
	for ( int y=0; y<m_nLabelNum; y++ ) {
		double dScore = 0;
		for ( int k=0; k<m_nK; k++ ) {
			dScore += theta[k] * m_dEta[etaIx];
			etaIx ++;
		}
		dScore -= m_dB;
		dScore += loss(y, doc->gndlabel);

		if ( doc->lossAugLabel == -1 || dScore > dMaxScore ) {
			doc->lossAugLabel = y;
			dMaxScore = dScore;
		}
	}
}

double MedSTC::loss(const int &y, const int &gnd)
{
	if ( y == gnd ) return 0;
	else return m_dDeltaEll;
}

/*
* writes the word assignments line for a Document to a file
*
*/
void MedSTC::write_word_assignment(FILE* f, Document* doc, double** phi)
{
	fprintf(f, "%03d", doc->length);
	for (int n = 0; n < doc->length; n++) {
		fprintf(f, " %04d:%02d", doc->words[n], argmax(phi[n], m_nK));
	}
	fprintf(f, "\n");
	fflush(f);
}


/*
* saves the gamma parameters of the current dataset
*
*/
void MedSTC::save_theta(char* filename, double** theta, int num_docs, int num_topics)
{
	FILE* fileptr = fopen(filename, "w");

	for (int d=0; d<num_docs; d++) {
		fprintf(fileptr, "%5.10f", theta[d][0]);
		for (int k=1; k<num_topics; k++) {
			fprintf(fileptr, " %5.10f", theta[d][k]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);
}


/*
* save the prediction results and the accuracy
*/
double MedSTC::save_prediction(char *filename, Corpus *pC)
{
	double sumlikelihood = 0;
	int nterms = 0;
	double sumavglikelihood = 0;
	for ( int d=0; d<pC->num_docs; d++ ) {
		sumlikelihood += pC->docs[d].lhood;
		nterms += pC->docs[d].total;
		sumavglikelihood += pC->docs[d].lhood / pC->docs[d].total;
	}
	double perwordlikelihood1 = sumlikelihood / nterms;
	double perwordlikelihood2 = sumavglikelihood / pC->num_docs;

	int nAcc = 0;
	
	for ( int d=0; d<pC->num_docs; d++ ){
		
		if ( pC->docs[d].gndlabel == pC->docs[d].predlabel )
			nAcc += 1;
			
	}
	double dAcc = (double)nAcc / pC->num_docs;	


	FILE* fileptr;
	fileptr = fopen(filename, "w");
	fprintf(fileptr, "accuracy: %5.10f\n", dAcc );
	fprintf(fileptr, "perword likelihood1: %5.10f\n", perwordlikelihood1);
	fprintf(fileptr, "perword likelihood2: %5.10f\n", perwordlikelihood2);

	for (int d=0; d<pC->num_docs; d++)
		fprintf(fileptr, "%d\t%d\n", pC->docs[d].predlabel, pC->docs[d].gndlabel);

	fclose(fileptr);

	return dAcc;
}

// allocate memory for the fast-computing parameters.
void MedSTC::init_param(Corpus *pC)
{
	sold_ = (double*)malloc(sizeof(double)*m_nK);
	mu_   = (double*)malloc(sizeof(double)*pC->num_terms);
	x_    = (double*)malloc(sizeof(double)*m_nK*pC->num_terms);
	g_    = (double*)malloc(sizeof(double)*m_nK*pC->num_terms);
	diag_ = (double*)malloc(sizeof(double)*m_nK*pC->num_terms);
	initEta_ = (double*)malloc(sizeof(double)*(m_nK*m_nLabelNum+1));
	m_pLBFGS = new CLBFGSCPP();
}

// allocate memory for the fast-computing parameters.
void MedSTC::release_param()
{
	if ( sold_ != NULL ) free( sold_ );
	if ( mu_ != NULL ) free( mu_ );
	if ( x_ != NULL ) free( x_ );
	if ( g_ != NULL ) free( g_ );
	if ( diag_ != NULL ) free( diag_ );
	if ( initEta_ != NULL ) free( initEta_ );
	if ( m_pLBFGS != NULL ) delete m_pLBFGS;
}

void MedSTC::init_phi(Document *doc, double **phi, double *theta, 
					  Params *param)
{
	for (int n=0; n<doc->length; n++) {
		double *phiPtr = phi[n];
		for ( int k=0; k<param->NTOPICS; k++ ) {
			phiPtr[k] = 1.0 / param->NTOPICS/* + myrand() * 0.01*/;
		}
	}

	double dRatio = m_dGamma / (m_dLambda + doc->length * m_dGamma);
	for ( int k=0; k<param->NTOPICS; k++ ) {
		for ( int n=0; n<doc->length; n++ )
			theta[k] += dRatio * phi[n][k];
	}
}

/*
* learn dictionary and find optimum code.
*/
int MedSTC::train(char* start, char* directory, Corpus* pC, Params *param)
{
	m_dDeltaEll = param->DELTA_ELL;
	m_dLambda   = param->LAMBDA;
	m_dRho      = param->RHO;
	m_dGamma    = m_dLambda;
	long runtime_start = get_runtime();

	// allocate variational parameters
	double ***phi = (double***)malloc(sizeof(double**) * pC->num_docs);
	for ( int d=0; d<pC->num_docs; d++ ) {
		phi[d] = (double**)malloc(sizeof(double*)*pC->docs[d].length);
		for (int n=0; n<pC->docs[d].length; n++) {
			phi[d][n] = (double*)malloc(sizeof(double) * param->NTOPICS);
		}
	}
	double **theta = (double**)malloc(sizeof(double*)*(pC->num_docs));
	for (int d=0; d<pC->num_docs; d++) {
		theta[d] = (double*)malloc(sizeof(double) * param->NTOPICS);
	}
	for ( int d=0; d<pC->num_docs; d++ ) {
		init_phi(&(pC->docs[d]), phi[d], theta[d], param);
	}

	// initialize model
	if (strcmp(start, "random")==0) {
		new_model(pC->num_docs, pC->num_terms, param->NTOPICS, 
								param->NLABELS, param->INITIAL_C);
		init_param( pC );
	} else {
		load_model(start);
		m_dC = param->INITIAL_C;
	}
	strcpy(m_directory, directory);

	char filename[100];
	

	// run expectation maximization
	sprintf(filename, "%s/lhood.dat", directory);
	FILE* lhood_file = fopen(filename, "w");

	Document *pDoc = NULL;
	double dobj, obj_old = 1, converged = 1;
	int nIt = 0;
	while (((converged < 0) || (converged > param->EM_CONVERGED) 
		|| (nIt <= 2)) && (nIt <= param->EM_MAX_ITER))
	{

		dobj = 0;
		double dLogLoss = 0;
		for ( int d=0; d<pC->num_docs; d++ ) {
			pDoc = &(pC->docs[d]);
			dobj += sparse_coding( pDoc, d, param, theta[d], phi[d] );
			dLogLoss += m_dLogLoss;
		}

		// m-step

		dict_learn(pC, theta, phi, param, false);

		if ( param->SUPERVISED == 1 ) { // for supervised MedLDA.
			char buff[512];
			get_train_filename( buff, m_directory, param );
			outputLowDimData( buff, pC, theta );

			svmStructSolver(buff, param, m_dMu);

			if ( param->PRIMALSVM == 1 ) { // solve svm in the primal form
				for ( int d=0; d<pC->num_docs; d++ ) {
					loss_aug_predict( &(pC->docs[d]), theta[d] );
				}
			}
			dobj += m_dsvm_primalobj;
		} else ;

		// check for convergence
		converged = fabs(1 - dobj / obj_old);
		obj_old = dobj;

		// output model and lhood
		if ( param->SUPERVISED == 1 ) {
			fprintf(lhood_file, "%10.10f\t%10.10f\t%5.5e\t%.5f\n", dobj-m_dsvm_primalobj, dobj, converged, dLogLoss);
		} else {
			fprintf(lhood_file, "%10.10f\t%5.5e\t%.5f\n", dobj, converged, dLogLoss);
		}
		fflush(lhood_file);
		if ( nIt > 0 && (nIt % LAG) == 0) {
			sprintf( filename, "%s/%d", directory, nIt + 1);
			save_model( filename, -1 );
			sprintf( filename, "%s/%d.theta", directory, nIt + 1 );
			save_theta( filename, theta, pC->num_docs, m_nK );
		}
		nIt ++;
	}
	// learn the final SVM.
	if ( param->SUPERVISED == 0 ) {
		char buff[512];
		get_train_filename(buff, m_directory, param);
		outputLowDimData(buff, pC, theta);

		svmStructSolver(buff, param, m_dMu);
	}
	long runtime_end = get_runtime();
	double dTrainTime = ((double)runtime_end-(double)runtime_start) / 100.0;


	// output the final model
	sprintf( filename, "%s/final", directory);
	save_model( filename, dTrainTime );

	// output the word assignments (for visualization)
	int nNum = 0, nAcc = 0;
	sprintf(filename, "%s/word-assignments.dat", directory);
	FILE* w_asgn_file = fopen(filename, "w");
	for (int d=0; d<pC->num_docs; d++) {

		sparse_coding( &(pC->docs[d]), d, param, theta[d], phi[d] );
		write_word_assignment(w_asgn_file, &(pC->docs[d]), phi[d]);

		nNum ++;
		pC->docs[d].predlabel = predict(theta[d]);
		if ( pC->docs[d].gndlabel == pC->docs[d].predlabel ) nAcc ++;
	}
	fclose(w_asgn_file);
	fclose(lhood_file);

	sprintf(filename,"%s/train.theta",directory);
	save_theta(filename, theta, pC->num_docs, m_nK);

	for (int d=0; d<pC->num_docs; d++) {
		free( theta[d] );
		for (int n=0; n<pC->docs[d].length; n++)
			free( phi[d][n] );
		free( phi[d] );
	}
	free( theta );
	free( phi );

	return nIt;
}

/*
* update the dictionary.
*/
bool MedSTC::dict_learn(Corpus *pC, double **theta, double***s, Params *param, bool bInit /*= true*/)
{
	// run projected gradient descent.
	int opt_size = m_nK * m_nNumTerms;
	int opt_iter = 0;
	double pref = 1.0;
	double f=0, eps = 1.0e-5, xtol = 1.0e-16;
	int m = 5, iprint[2], iflag[1];

	bool diagco = false;
	iprint[0] = -1; iprint[1] = 0;
	iflag[0]=0;

#if _DEBUG
	fileptr = fopen("grad_beta.txt", "a");
#endif

	do
	{
		opt_iter ++;
		pref = f;

		get_param(x_, m_nK, m_nNumTerms);
		f = fdf_beta(pC, theta, s, g_);
		if (fabs(pref/f - 1.0) < 1e-4) break;



		try	{
			m_pLBFGS->lbfgs( opt_size, m, x_, f, g_, diagco, diag_, iprint, eps, xtol, iflag );
		} catch(ExceptionWithIflag* e) {
			Rprintf("exception in l-bfgs\n");
			break;
		}
		set_param(x_, m_nK, m_nNumTerms);

		for ( int k=0; k<m_nK; k++ ) {
			for ( int i=0; i<m_nNumTerms; i++ ) {
				mu_[i] = m_dLogProbW[i][k];
			}
			
			project_beta2( mu_, m_nNumTerms, 1.0, MIN_BETA );

			for ( int i=0; i<m_nNumTerms; i++ ) {
				m_dLogProbW[i][k] = mu_[i];
			}
		}
	} while ( iflag[0] != 0 && opt_iter < param->VAR_MAX_ITER );

	return true;
}

double MedSTC::fdf_beta(Corpus *pC, double **theta, double ***s, double *g)
{	
	memset(g, 0, sizeof(double)*m_nK*m_nNumTerms);

	Document *pDoc = NULL;
	int nWrd, x, gIx = 0, d, n, k;
	double fVal = 0, dVal = 0;
	double *pS = NULL, *bPtr = NULL;
	double **pSS = NULL;
	for ( d=0; d<pC->num_docs; d++ ) {
		pDoc = &(pC->docs[d]);
		pSS = s[d];
	
		for ( n=0; n<pDoc->length; n++ ) {
			nWrd = pDoc->words[n];   // word index
			x    = pDoc->counts[n];  // word count
			bPtr = m_dLogProbW[nWrd];

			pS = pSS[n];
			dVal = m_dPoisOffset;
			for ( k=0; k<m_nK; k++ ) {
				if ( pS[k] > 0 ) 
					dVal += pS[k] * bPtr[k];
			}

			// update function value.
			fVal += dVal - x * log(dVal);

			// update gradients.
			dVal = 1 - x/dVal;
			gIx = nWrd * m_nK;
			for ( k=0; k<m_nK; k++ ) {
				if ( pS[k] > 0 )
					g[gIx] += dVal * pS[k];
				gIx ++;
			}
		}
	}

	return fVal;
}

// project beta to simplex ( N*log(N) ).
void MedSTC::project_beta( double *beta, const int &nTerms )
{
	// copy. (mu for temp use)
	for ( int i=0; i<nTerms; i++ ) {
		mu_[i] = beta[i];
	}
	// sort m_mu.
	quickSort(mu_, 0, nTerms-1);

	// find rho.
	int rho = 0;
	double dsum = 0;
	for ( int i=0; i<nTerms; i++ ) {
		dsum += mu_[i];

		if ( mu_[i] - (dsum-1)/(i+1) > 0 )
			rho = i;
	}

	double theta = 0;
	for ( int i=0; i<=rho; i++ ) {
		theta += mu_[i];
	}
	theta = (theta-1) / (rho+1);

	for ( int i=0; i<nTerms; i++ ) {
		beta[i] = max(0.0, beta[i] - theta);
	}
}
// linear algorithm
void MedSTC::project_beta2( double *beta, const int &nTerms, 
							const double &dZ, const double &epsilon )
{
	vector<int> U(nTerms);
	for ( int i=0; i<nTerms; i++ ) {
		mu_[i] = beta[i] - epsilon;
		U[i] = i + 1;
	}
	

	/* project to a simplex. */
	double s = 0;
	int p = 0;
	while ( !U.empty() ) {
		int nSize = U.size();
		int k = U[ rand()%nSize ];

		/* partition U. */
		vector<int> G, L;
		int deltaP = 0;
		double deltaS = 0;
		for ( int i=0; i<nSize; i++ ) {
			int j = U[i];

			if ( mu_[j-1] >= mu_[k-1] ) {
				if ( j != k ) G.push_back( j );
				deltaP ++;
				deltaS += beta[j-1];
			} else L.push_back( j );
		}

		if ( s + deltaS - (p + deltaP) * mu_[k-1] < dZ ) {
			s += deltaS;
			p += deltaP;
			U = L;
		} else {
			U = G;
		}
	}

	double theta = (s - dZ) / p;
	for ( int i=0; i<nTerms; i++ ) {
		beta[i] = max(mu_[i] - theta, 0.0) + epsilon;
	}
}

void MedSTC::set_param(double *w, const int &nK, const int &nTerms)
{
	double *bPtr = NULL;
	int j = 0;
	for ( int i=0; i<nTerms; i++ ) {
		bPtr = m_dLogProbW[i];
		for ( int k=0; k<m_nK; k++ ) {
			bPtr[k] = w[j]; j++;
		}
	}
}

void MedSTC::get_param(double *w, const int &nK, const int &nTerms)
{
	double *bPtr = NULL;
	int j = 0;
	for ( int i=0; i<nTerms; i++ ) {
		bPtr = m_dLogProbW[i];
		for ( int k=0; k<m_nK; k++ ) {
			w[j] = bPtr[k]; j++;
		}
	}
}

void MedSTC::set_init_param(STRUCT_LEARN_PARM *struct_parm, LEARN_PARM *learn_parm, 
							KERNEL_PARM *kernel_parm, int *alg_type)
{
	/* set default */
	(*alg_type) = DEFAULT_ALG_TYPE;
	struct_parm->C = -0.01;
	struct_parm->slack_norm = 1;
	struct_parm->epsilon = DEFAULT_EPS;
	struct_parm->custom_argc = 0;
	struct_parm->loss_function = DEFAULT_LOSS_FCT;
	struct_parm->loss_type = DEFAULT_RESCALING;
	struct_parm->newconstretrain = 100;
	struct_parm->ccache_size = 5;
	struct_parm->batch_size = 100;
	struct_parm->delta_ell = m_dDeltaEll;

	strcpy(learn_parm->predfile, "trans_predictions");
	strcpy(learn_parm->alphafile, "");
	verbosity = 0;/*verbosity for svm_light*/
	struct_verbosity = 0; /*verbosity for struct learning portion*/
	learn_parm->biased_hyperplane = 1;
	learn_parm->remove_inconsistent = 0;
	learn_parm->skip_final_opt_check = 0;
	learn_parm->svm_maxqpsize = 10;
	learn_parm->svm_newvarsinqp = 0;
	learn_parm->svm_iter_to_shrink = -9999;
	learn_parm->maxiter = 100000;
	learn_parm->kernel_cache_size = 40;
	learn_parm->svm_c = 99999999;  /* overridden by struct_parm->C */
	learn_parm->eps = 0.001;       /* overridden by struct_parm->epsilon */
	learn_parm->transduction_posratio = -1.0;
	learn_parm->svm_costratio = 1.0;
	learn_parm->svm_costratio_unlab = 1.0;
	learn_parm->svm_unlabbound = 1E-5;
	learn_parm->epsilon_crit = 0.001;
	learn_parm->epsilon_a = 1E-10;  /* changed from 1e-15 */
	learn_parm->compute_loo = 0;
	learn_parm->rho = 1.0;
	learn_parm->xa_depth = 0;
	kernel_parm->kernel_type = 0;
	kernel_parm->poly_degree = 3;
	kernel_parm->rbf_gamma = 1.0;
	kernel_parm->coef_lin = 1;
	kernel_parm->coef_const = 1;
	strcpy(kernel_parm->custom,"empty");

	if(learn_parm->svm_iter_to_shrink == -9999) {
		learn_parm->svm_iter_to_shrink=100;
	}

	if((learn_parm->skip_final_opt_check) 
		&& (kernel_parm->kernel_type == LINEAR)) {
			learn_parm->skip_final_opt_check=0;
	}    
	parse_struct_parameters(struct_parm);
}

void MedSTC::get_train_filename(char *buff, char *dir, Params *param)
{
	sprintf(buff, "%s/ftrain_k%d_rho%d_gamma%d.txt", dir, m_nK,
		(int)(param->RHO*100), (int)(param->LAMBDA*10) );
}
void MedSTC::get_test_filename(char *buff, char *dir, Params *param)
{
	sprintf(buff, "%s/ftest_k%d_rho%d_gamma%d.txt", dir, m_nK,
		(int)(param->RHO*100), (int)(param->LAMBDA*10) );	
}
void MedSTC::svmStructSolver(char *dataFileName, Params *param, double *res)
{
	LEARN_PARM learn_parm;
	KERNEL_PARM kernel_parm;
	STRUCT_LEARN_PARM struct_parm;
	STRUCTMODEL structmodel;
	int alg_type = 2;

	/* set the parameters. */
	set_init_param(&struct_parm, &learn_parm, &kernel_parm, &alg_type);
	struct_parm.C = m_dC;

	/* read the training examples */
	SAMPLE sample = read_struct_examples(dataFileName, &struct_parm);

	if(param->SVM_ALGTYPE == 0)
		svm_learn_struct(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, NSLACK_ALG);
	//else if(alg_type == 1)
	//	svm_learn_struct(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, NSLACK_SHRINK_ALG);
	else if(param->SVM_ALGTYPE == 2) {
		struct_parm.C = m_dC * sample.n;   // Note: in n-slack formulation, C is not divided by N.
		svm_learn_struct_joint(sample, &struct_parm, &learn_parm, 
			&kernel_parm, &structmodel, ONESLACK_PRIMAL_ALG);
	} else if ( param->SVM_ALGTYPE == 3 ) {
		struct_parm.C = m_dC * sample.n;   // Note: in n-slack formulation, C is not divided by N.
		int nEtaNum = m_nLabelNum * m_nK;
		for ( int i=1; i<=nEtaNum; i++ ) {
			initEta_[i] = m_dEta[i-1];
		}
		svm_learn_struct_joint( sample, &struct_parm, &learn_parm, 
			&kernel_parm, &structmodel, ONESLACK_PRIMAL_ALG, initEta_, nEtaNum );
	} 

	/* get the optimal lagrangian multipliers. 
	*    Note: for 1-slack formulation: the "marginalization" is 
	*           needed for fast computation.
	*/
	int nVar = sample.n * m_nLabelNum;
	for ( int k=0; k<nVar; k++ ) m_dMu[k] = 0;

	if ( param->SVM_ALGTYPE == 0 ) {
		for ( int k=1; k<structmodel.svm_model->sv_num; k++ ) {
			int docnum = structmodel.svm_model->supvec[k]->orgDocNum;
			m_dMu[docnum] = structmodel.svm_model->alpha[k];
		}
	} else if ( param->SVM_ALGTYPE == 2 ) {
		for ( int k=1; k<structmodel.svm_model->sv_num; k++ ) {
			int *vecLabel = structmodel.svm_model->supvec[k]->lvec;

			double dval = structmodel.svm_model->alpha[k] / sample.n;
			for ( int d=0; d<sample.n; d++ ) {
				int label = vecLabel[d];
				m_dMu[d*m_nLabelNum + label] += dval;
			}
		}
	} else ;

	//FILE *fileptr = fopen("SVMLightSolution.txt", "a");
	// set the SVM parameters.
	m_dB = structmodel.svm_model->b;
	for ( int y=0; y<m_nLabelNum; y++ ) {
		for ( int k=0; k<m_nK; k++ ){
			int etaIx = y * m_nK + k;
			m_dEta[etaIx] = structmodel.w[etaIx+1];
		}
	}
	m_dsvm_primalobj = structmodel.primalobj;

	// free the memory
	free_struct_sample(sample);
	free_struct_model(structmodel);
}
void MedSTC::outputLowDimData(char *filename, Corpus *pC, double **theta)
{
	FILE *fileptr = fopen(filename, "w");
	for ( int d=0; d<min(pC->num_docs, 11269); d++ ) {
		int label = pC->docs[d].gndlabel;

		fprintf(fileptr, "%d %d", m_nK, label);
		for ( int k=0; k<m_nK; k++ ) {
			fprintf(fileptr, " %d:%.10f", k, theta[d][k]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);
}

/*
* allocate new model
*/
void MedSTC::new_model(int num_docs, int num_terms, int num_topics, int num_labels, double C)
{
	m_nK = num_topics;
	m_nLabelNum = num_labels;
	m_nNumTerms = num_terms;

	m_dLogProbW = (double**)malloc(sizeof(double*)*num_terms);
	m_dEta = (double*)malloc(sizeof(double) * num_topics * num_labels);
	m_dMu = (double*)malloc(sizeof(double) * num_docs * num_labels);
	for (int i=0; i<num_terms; i++) {
		m_dLogProbW[i] = (double*)malloc(sizeof(double)*num_topics);
		
		for (int j=0; j<num_topics; j++) 
			m_dLogProbW[i][j] = 1.0 / num_terms;
	}
	for ( int i=0; i<num_topics; i++ ) {	
		for (int j=0; j<num_labels; j++) 
			m_dEta[i*num_labels + j] = 0;
	}
	for (int i=0; i<num_docs; i++)
		for (int j=0; j<num_labels; j++)
			m_dMu[i*num_labels + j] = 0;

	m_nDim = num_docs;
	m_dC = C;
	m_directory = new char[512];
}

/*
* deallocate new model
*/
void MedSTC::free_model()
{
	if ( m_dLogProbW != NULL ) {
		for (int i=0; i<m_nNumTerms; i++) {
			free(m_dLogProbW[i]);
		}
		free(m_dLogProbW);
	}
	if ( m_dEta != NULL ) free(m_dEta);
	if ( m_dMu != NULL )  free(m_dMu);
	release_param();
}

/*
* save a model
*/
void MedSTC::save_model(char* model_root, double dTime)
{
	char filename[100];
	sprintf(filename, "%s.beta", model_root);
	FILE *fileptr = fopen(filename, "w");

	fprintf(fileptr, "%5.10f\n", m_dPoisOffset);
	for (int k=0; k<m_nK; k++) {
		for (int i=0; i<m_nNumTerms; i++) {
			fprintf(fileptr, " %5.10f", log(m_dLogProbW[i][k]));
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);

	sprintf(filename, "%s.eta", model_root);
	fileptr = fopen(filename, "w");
	fprintf(fileptr, "%5.10f\n", m_dB);
	for (int i=0; i<m_nK; i++) {
		// the first element is eta[k]
		for ( int k=0; k<m_nLabelNum; k++ ) {
			if ( k == m_nLabelNum-1 )
				fprintf(fileptr, "%5.10f", m_dEta[i+k*m_nK]);
			else 
				fprintf(fileptr, "%5.10f ", m_dEta[i+k*m_nK]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);

	sprintf(filename, "%s.other", model_root);
	fileptr = fopen(filename, "w");
	fprintf(fileptr, "num_topics %d\n", m_nK);
	fprintf(fileptr, "num_labels %d\n", m_nLabelNum);
	fprintf(fileptr, "num_terms %d\n", m_nNumTerms);
	fprintf(fileptr, "num_docs %d\n", m_nDim);
	fprintf(fileptr, "C %5.10f\n", m_dC);
	fprintf(fileptr, "lambda %5.10f\n", m_dLambda);
	fprintf(fileptr, "gamma %5.10f\n", m_dGamma);
	fprintf(fileptr, "rho %5.10f\n", m_dRho);
	fprintf(fileptr, "training-time (cpu-sec) %5.10f\n", dTime);
	fclose(fileptr);
}

void MedSTC::load_model(char* model_root)
{
	char filename[100];
	int num_terms, num_topics, num_labels, num_docs;
	float x, C, dLambda, dGamma, dRho;

	sprintf(filename, "%s.other", model_root);

	FILE *fileptr = fopen(filename, "r");
	fscanf(fileptr, "num_topics %d\n", &num_topics);
	fscanf(fileptr, "num_labels %d\n", &num_labels);
	fscanf(fileptr, "num_terms %d\n", &num_terms);
	fscanf(fileptr, "num_docs %d\n", &num_docs);
	fscanf(fileptr, "C %f\n", &C);
	fscanf(fileptr, "lambda %f\n", &dLambda);
	fscanf(fileptr, "gamma %f\n", &dGamma);
	fscanf(fileptr, "rho %f\n", &dRho);
	fclose(fileptr);

	new_model(num_docs, num_terms, num_topics, num_labels, C);
	m_dLambda = dLambda;
	m_dGamma  = dGamma;
	m_dRho    = dRho;

	sprintf(filename, "%s.beta", model_root);

	fileptr = fopen(filename, "r");

	fscanf(fileptr, "%lf\n", &m_dPoisOffset);
	for ( int i=0; i<m_nK; i++) {
		for ( int j=0; j<m_nNumTerms; j++) {
			fscanf(fileptr, "%f", &x);
			m_dLogProbW[j][i] = exp( x );
		}
	}
	fclose(fileptr);

	sprintf(filename, "%s.eta", model_root);

	fileptr = fopen(filename, "r");
	fscanf(fileptr, "%lf\n", &m_dB);
	for ( int i=0; i<m_nK; i++) {
		for ( int k=0; k<m_nLabelNum; k++ ) {
			fscanf(fileptr, "%f", &x);
			m_dEta[i+k*m_nK] = x;
		}
	}
	fclose(fileptr);
}

/*
* perform sparse coding with a learned dictionary.
*/
double MedSTC::sparse_coding(Document *doc, const int &docIx,
							 Params *param, double* theta, double **s )
{
	double *sPtr = NULL, *bPtr = NULL, dval = 0;
	// initialize mu & theta
	double dThetaRatio = m_dGamma / (m_dLambda + doc->length * m_dGamma);
	int svmMuIx = docIx * m_nLabelNum;
	int gndIx = doc->gndlabel * m_nK;
	int nWrd = 0, xVal = 0, k, n;
	for ( k=0; k<m_nK; k++ ) {
		theta[k] = 0;
	}
	for ( n=0; n<doc->length; n++ ) {
		mu_[n] = 0 + m_dPoisOffset;
		nWrd = doc->words[n];
		sPtr = s[n];
		bPtr = m_dLogProbW[nWrd];

		for ( k=0; k<m_nK; k++ ) {
			dval = sPtr[k];
			if ( dval > 0 ) {
				mu_[n] += dval * bPtr[k];
				theta[k] += dval;
			}
		}
	}
	// initialize theta
	for ( k=0; k<m_nK; k++ ) {
		theta[k] = theta[k] * dThetaRatio;
	}
	if ( param->SUPERVISED == 1 ) {
		if ( param->PRIMALSVM == 1 ) {
			if ( doc->lossAugLabel != -1 && doc->lossAugLabel != doc->gndlabel ) {
				int lossIx = doc->lossAugLabel * m_nK;
				double dHingeRatio = 0.5 * m_dC / (m_dLambda + doc->length * m_dGamma);
				for ( k=0; k<m_nK; k++ ) {
					theta[k] += (m_dEta[gndIx+k] - m_dEta[lossIx+k]) * dHingeRatio ;
					theta[k] = max(0.0, theta[k]); // enforce theta to be non-negative.
				}
			}
		} else {
			double dHingeRatio = 0.5 / (m_dLambda + doc->length * m_dGamma);
			for ( k=0; k<m_nK; k++ ) {
				for ( int m=0; m<m_nLabelNum; m++ ) {
					int yIx = m * m_nK;
					theta[k] += m_dMu[svmMuIx+m] * (m_dEta[gndIx+k] - m_dEta[yIx+k]) * dHingeRatio ;
				}
				theta[k] = max(0.0, theta[k]); // enforce theta to be non-negative.
			}
		}
	} else; 

	// alternating minimization over theta & s.
	double dconverged=1, fval, dobj_val, dpreVal=1;
	double mu, eta, beta, aVal, bVal, cVal, discVal, sqrtDiscVal;
	double s1, s2;

	int it = 0;
	while (((dconverged < 0) || (dconverged > param->VAR_CONVERGED) 
		|| (it <= 2)) && (it <= param->VAR_MAX_ITER))
	{
		for ( n=0; n<doc->length; n++ ) {
			nWrd = doc->words[n];
			xVal = doc->counts[n];
			bPtr = m_dLogProbW[nWrd];

			// optimize over s.
			sPtr = s[n];
			for ( k=0; k<m_nK; k++ ) {
				sold_[k] = sPtr[k];
				beta = bPtr[k];

				if ( beta > MIN_BETA ) {
					mu = mu_[n] - sold_[k] * beta;
					eta = beta + m_dRho - 2 * m_dGamma * theta[k];

					// solve the quadratic equation.
					aVal = 2 * m_dGamma * beta;
					bVal = 2 * m_dGamma * mu + beta * eta;
					cVal = mu * eta - xVal * beta;
					discVal = bVal * bVal - 4 * aVal * cVal;
					sqrtDiscVal = sqrt( discVal );
					s1 = max(0.0, (sqrtDiscVal - bVal) / (2*aVal));  // non-negative
					s2 = max(0.0, 0 - (sqrtDiscVal + bVal) / (2*aVal)); // non-negative
					sPtr[k] = max(s1, s2);
				} else {
					// solve the degenerated linear equation
					sPtr[k] = max(0.0, theta[k] - 0.5*m_dRho/m_dGamma);
				}

				// update mu.
				mu_[n] += (sPtr[k] - sold_[k]) * beta;
			}

			// update theta.
			for ( k=0; k<m_nK; k++ ) {
				theta[k] += (sPtr[k] - sold_[k]) * dThetaRatio;
				theta[k] = max(0.0, theta[k]); // enforce theta to be non-negative.
			}
		}

		// check optimality condition.
		fval = 0;
		m_dLogLoss = 0;
		for ( n=0; n<doc->length; n++ ) {
			double dval = mu_[n];
			double dLogLoss = (dval - doc->counts[n] * log(dval));
			m_dLogLoss += dLogLoss;
			fval += dLogLoss + m_dGamma * L2Dist(s[n], theta, m_nK)
					+ m_dRho * L1Norm(s[n], m_nK);
		}
		
		fval += m_dLambda * L2Norm(theta, m_nK);

		dobj_val = fval;
		if ( param->SUPERVISED == 1 ) { // compute svm objective
			if ( param->PRIMALSVM == 1 ) {
				if ( doc->lossAugLabel != -1 && doc->lossAugLabel != doc->gndlabel ) { // hinge loss
					int lossIx = doc->lossAugLabel * m_nK;
					dval = loss( doc->gndlabel, doc->lossAugLabel );
					for ( k=0; k<m_nK; k++ ) {
						dval += theta[k] * (m_dEta[lossIx+k] - m_dEta[gndIx+k]);
					}
					dobj_val += m_dC * dval;
				}
			} else {
				for ( int m=0; m<m_nLabelNum; m++ ) { 
					int yIx = m * m_nK;
					dval = loss( doc->gndlabel, m );
					for ( k=0; k<m_nK; k++ ) {
						dval += theta[k] * (m_dEta[yIx+k] - m_dEta[gndIx+k]);
					}
					dobj_val += m_dMu[svmMuIx+m] * dval;
				}
			}
		}

		dconverged = (dpreVal - dobj_val) / dpreVal;
		dpreVal = dobj_val;
		it ++;
	}

	return fval;
}

double MedSTC::sparse_coding(char* model_dir, Corpus* pC, Params *param)
{
	char model_root[512];
	sprintf(model_root, "%s/final", model_dir);
	load_model(model_root);
	init_param( pC );

	// remove unseen words
	Document* doc = NULL;
	if ( pC->num_terms > m_nNumTerms ) {
		for ( int i=0; i<pC->num_docs; i ++ ) {
			doc = &(pC->docs[i]);
			for ( int k=0; k<doc->length; k++ )
				if ( doc->words[k] >= m_nNumTerms )
					doc->words[k] = m_nNumTerms - 1;
		}
	}

	// allocate memory
	int max_length = pC->max_corpus_length();
	double **phi = (double**)malloc(sizeof(double*)*max_length);
	for (int n=0; n<max_length; n++) {
		phi[n] = (double*)malloc(sizeof(double) * m_nK);
	}
	double **theta = (double**)malloc(sizeof(double*)*(pC->num_docs));
	for (int d=0; d<pC->num_docs; d++) {
		theta[d] = (double*)malloc(sizeof(double)*m_nK);
	}
	double **avgTheta = (double**)malloc(sizeof(double*)*m_nLabelNum);
	for ( int k=0; k<m_nLabelNum; k++ ) {
		avgTheta[k] = (double*)malloc(sizeof(double)*m_nK);
		memset(avgTheta[k], 0, sizeof(double)*m_nK);
	}
	vector<vector<double> > avgWrdCode(m_nNumTerms);
	vector<int> wrdCount(m_nNumTerms, 0);
	for ( int i=0; i<m_nNumTerms; i++ ) {
		avgWrdCode[i].resize( m_nK, 0 );
	}
	vector<int> perClassDataNum(m_nLabelNum, 0);

	
	char filename[100];
	sprintf(filename, "%s/evl-slda-obj.dat", model_dir);
	FILE* fileptr = fopen(filename, "w");
	
	
	
	double dEntropy = 0, dobj = 0, dNonZeroWrdCode = 0, dNonZeroDocCode = 0;
	int nTotalWrd = 0;
	
	for (int d=0; d<pC->num_docs; d++) {

		doc = &(pC->docs[d]);
		// initialize phi.
		for (int n=0; n<doc->length; n++) {
			double *phiPtr = phi[n];
			for ( int k=0; k<m_nK; k++ ) {
				phiPtr[k] = 1.0 / m_nK;
			}
		}
		dobj = sparse_coding( doc, d, param, theta[d], phi );

		// do prediction
		doc->predlabel = predict(theta[d]);
		
		doc->scores = (double*) malloc(sizeof(double)*m_nLabelNum);;
		predict_scores(doc->scores,theta[d]);
		
		doc->lhood = dobj;
		fprintf(fileptr, "%5.5f\n", dobj);

		//dEntropy += safe_entropy( exp[d], m_nK );
		int gndLabel = doc->gndlabel;
		perClassDataNum[gndLabel] ++;
		for ( int k=0; k<m_nK; k++ ) {
			for ( int n=0; n<doc->length; n++ ) {
				//fprintf( wrdfptr, "%.10f ", phi[n][k] );
				if ( phi[n][k] > 0/*1e-10*/ ) dNonZeroWrdCode ++;
			}
			//fprintf( wrdfptr, "\n" );
			avgTheta[gndLabel][k] += theta[d][k];
			if ( theta[d][k] > 0 ) dNonZeroDocCode ++;
		}
		nTotalWrd += doc->length;
		//fprintf( wrdfptr, "\n" );
		//fflush( wrdfptr );

		dEntropy += safe_entropy( theta[d], m_nK );

		//// the average distribution of each word on the topics.
		//for ( int n=0; n<doc->length; n++ ) {
		//	int wrd = doc->words[n];
		//	wrdCount[wrd] ++;
		//	for ( int k=0; k<m_nK; k++ ) {
		//		avgWrdCode[wrd][k] += phi[n][k];
		//	}
		//}
	}
	
	fclose( fileptr );
	
	
	

	/* save theta & average theta. */
	sprintf(filename, "%s/evl-theta.dat", model_dir);
	save_theta(filename, theta, pC->num_docs, m_nK);
	sprintf(filename, "%s/evl-avgTheta.dat", model_dir);
	for ( int m=0; m<m_nLabelNum; m++ ) {
		int dataNum = perClassDataNum[m];
		for ( int k=0; k<m_nK; k++ ) {
			avgTheta[m][k] /= dataNum;
		}
	}
	printf_mat(filename, avgTheta, m_nLabelNum, m_nK);

	/* save the average topic distribution for each word. */
	sprintf(filename, "%s/evl-avgWrdCode.dat", model_dir);
	fileptr = fopen( filename, "w" );
	for ( int i=0; i<m_nNumTerms; i++ ) {
		double dNorm = wrdCount[i];
		for ( int k=0; k<m_nK; k++ ) {
			double dval = avgWrdCode[i][k];
			if ( dNorm > 0 ) dval /= dNorm;
			fprintf( fileptr, "%.10f ", dval );
		}
		fprintf( fileptr, "\n" );
	}
	fclose( fileptr );
	//printf_mat( filename, avgWrdCode, m_nNumTerms, m_nK );

	/* save the low dimension representation. */
	get_test_filename(filename, model_dir, param);
	outputLowDimData(filename, pC, theta);

	/* save the prediction performance. */
	sprintf(filename, "%s/evl-performance.dat", model_dir);
	double dAcc = save_prediction(filename, pC);

	// free memory
	for (int i=0; i<pC->num_docs; i++ ) {
		free( theta[i] );
	}
	for ( int n=0; n<max_length; n++ ) {
		free( phi[n] );
	}
	for ( int k=0; k<m_nLabelNum; k++ ) {
		free( avgTheta[k] );
	}
	free( theta );
	free( phi );
	free( avgTheta );

	return dAcc;
}

void MedSTC::predictTest(Corpus* pC, Params *param)
{
	init_param( pC );
	Document* doc = NULL;
	if ( pC->num_terms > m_nNumTerms ) {
		for ( int i=0; i<pC->num_docs; i ++ ) {
			doc = &(pC->docs[i]);
			for ( int k=0; k<doc->length; k++ )
				if ( doc->words[k] >= m_nNumTerms )
					doc->words[k] = m_nNumTerms - 1;
		}
	}
	int max_length = pC->max_corpus_length();
	double **phi = (double**)malloc(sizeof(double*)*max_length);
	for (int n=0; n<max_length; n++) {
		phi[n] = (double*)malloc(sizeof(double) * m_nK);
	}
	double **theta = (double**)malloc(sizeof(double*)*(pC->num_docs));
	for (int d=0; d<pC->num_docs; d++) {
		theta[d] = (double*)malloc(sizeof(double)*m_nK);
	}
	double **avgTheta = (double**)malloc(sizeof(double*)*m_nLabelNum);
	for ( int k=0; k<m_nLabelNum; k++ ) {
		avgTheta[k] = (double*)malloc(sizeof(double)*m_nK);
		memset(avgTheta[k], 0, sizeof(double)*m_nK);
	}
	vector<vector<double> > avgWrdCode(m_nNumTerms);
	vector<int> wrdCount(m_nNumTerms, 0);
	for ( int i=0; i<m_nNumTerms; i++ ) {
		avgWrdCode[i].resize( m_nK, 0 );
	}
	vector<int> perClassDataNum(m_nLabelNum, 0);
	double dEntropy = 0, dobj = 0, dNonZeroWrdCode = 0, dNonZeroDocCode = 0;
	int nTotalWrd = 0;
	
	for (int d=0; d<pC->num_docs; d++) {

		doc = &(pC->docs[d]);
		// initialize phi.
		for (int n=0; n<doc->length; n++) {
			double *phiPtr = phi[n];
			for ( int k=0; k<m_nK; k++ ) {
				phiPtr[k] = 1.0 / m_nK;
			}
		}
		dobj = sparse_coding( doc, d, param, theta[d], phi );

		doc->predlabel = predict(theta[d]);
		
		doc->scores = (double*) malloc(sizeof(double)*m_nLabelNum);;
		predict_scores(doc->scores,theta[d]);
		doc->lhood = dobj;
		int gndLabel = doc->gndlabel;
		perClassDataNum[gndLabel] ++;
		for ( int k=0; k<m_nK; k++ ) {
			for ( int n=0; n<doc->length; n++ ) {
				if ( phi[n][k] > 0/*1e-10*/ ) dNonZeroWrdCode ++;
			}
			
			avgTheta[gndLabel][k] += theta[d][k];
			if ( theta[d][k] > 0 ) dNonZeroDocCode ++;
		}
		nTotalWrd += doc->length;
	

		dEntropy += safe_entropy( theta[d], m_nK );

	}
	
	for (int i=0; i<pC->num_docs; i++ ) {
		free( theta[i] );
	}
	for ( int n=0; n<max_length; n++ ) {
		free( phi[n] );
	}
	for ( int k=0; k<m_nLabelNum; k++ ) {
		free( avgTheta[k] );
	}
	free( theta );
	free( phi );
	free( avgTheta );
}


void MedSTC::learn_svm(char *model_dir, const double &dC, const double &dEll)
{
	char model_root[512];
	sprintf(model_root, "%s/final", model_dir);
	load_model( model_root );
	m_dC = dC;
	m_dDeltaEll = dEll;

	Params *param = new Params();
	param->DELTA_ELL = m_dDeltaEll;
	param->LAMBDA = m_dLambda;
	param->RHO = m_dRho;
	param->INITIAL_C = m_dC;
	param->NLABELS = m_nLabelNum;
	param->NTOPICS = m_nK;
	param->SVM_ALGTYPE = 2;

	char filename[512];
	get_train_filename( filename, model_dir, param );
	svmStructSolver( filename, param, m_dMu );
	
	// for testing.
	int nDataNum = 0;
	double dAcc = 0;
	get_test_filename( filename, model_dir, param );
	readLowDimData( filename, nDataNum );

	for ( int d=0; d<nDataNum; d++ ) {
		int predLabel = predict( theta_[d] );
		if ( label_[d] == predLabel ) dAcc ++;
	}
	dAcc /= nDataNum;

	
	FILE *fileptr = fopen("overall-res.txt", "a");
	fprintf(fileptr, "setup (K: %d; C: %.3f; fold: %d; ell: %.2f; lambda: %.2f; rho: %.4f; svm_alg: %d; maxIt: %d): accuracy %.3f; avgNonZeroWrdCode: %.5f\n", 
		m_nK, m_dC, 0, dEll, m_dLambda, m_dRho, param->SVM_ALGTYPE, 0, dAcc, 0.0);
	fclose(fileptr);

	save_model( model_root, -1 );

	for ( int d=0; d<nDataNum; d++ ) {
		free( theta_[d] );
	}
	free( theta_ );
	free( label_ );
}

void MedSTC::readLowDimData(char *filename, int &nDataNum)
{
	FILE *fileptr = fopen(filename, "r");
	int nd = 0, wrd = 0, length = 0;
	double dval = 0;

	theta_ = NULL;
	label_ = NULL;
	while ((fscanf(fileptr, "%10d", &length) != EOF)) {
		theta_ = (double**) realloc(theta_, sizeof(double*)*(nd+1));
		label_ = (int*) realloc(label_, sizeof(int)*(nd+1));
		theta_[nd] = (double*) malloc(sizeof(double)*length);
		
		int nLabel;
		fscanf(fileptr, "%d", &nLabel);
		label_[nd] = nLabel;

		for ( int k=0; k<length; k++ ) {
			fscanf(fileptr, "%d:%lf", &wrd, &dval);
			theta_[nd][k] = dval;
		}
		nd ++;
	}
	fclose(fileptr);

	nDataNum = nd;
}

int MedSTC::predict(double *theta)
{
	int predlabel = -1;
	double dMaxScore = 0;
	for ( int y=0; y<m_nLabelNum; y++ )
	{
		double dScore = 0;
		for ( int k=0; k<m_nK; k++ ) {
			int etaIx = y * m_nK + k;
			dScore += theta[k] * m_dEta[etaIx];
		}
		dScore -= m_dB;

		if ( predlabel == -1 || dScore > dMaxScore ) {
			predlabel = y;
			dMaxScore = dScore;
		}
	}

	return predlabel;
}

void MedSTC::predict_scores(double * scores, double *theta)
{
	
	for ( int y=0; y<m_nLabelNum; y++ )
	{
		double dScore = 0;
		for ( int k=0; k<m_nK; k++ ) {
			int etaIx = y * m_nK + k;
			dScore += theta[k] * m_dEta[etaIx];
		}
		dScore -= m_dB;

		scores[y]=dScore;
	}
	
}
