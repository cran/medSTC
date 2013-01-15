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

#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include <map>
#include <string>
#include "../svmlight/svm_common.h"
#include "medstc.h"
#ifdef WINDOWS_VERSION
	#include <direct.h>
#endif
#include <unistd.h>
#include <dirent.h>
#include <R.h>
#include <Rdefines.h>
//using namespace std;

#define CHECK(VAL, TYPE) if (!is##TYPE(VAL)) { \
    error(#VAL " must be a(n) " #TYPE "."); \
  }

#define CHECKLEN(VAL, TYPE, LEN) if (!is##TYPE(VAL) || length(VAL) != LEN) { \
    error(#VAL " must be a length " #LEN " " #TYPE "."); \
  }

#define CHECKMATROW(VAL, TYPE, NROW) if (!isMatrix(VAL) || !is##TYPE(VAL) || NUMROWS(VAL) != NROW) { \
    error(#VAL " must be a matrix with " #NROW " rows of type " #TYPE "."); \
  }
#define NUMROWS(MAT) (INTEGER(GET_DIM(MAT))[0])
#define NUMCOLS(MAT) (INTEGER(GET_DIM(MAT))[1])

int file_exist (char *filename)
{
  struct stat   buffer;   
  return (stat (filename, &buffer) == 0);
}

void read_data_from_R(Corpus* c, SEXP documents,int nd, int labels[])
{

	int nw, word, count;
	int num_terms = 0;
	Document* docs = (Document*) malloc(sizeof(Document)*(nd));

	for (int i=0; i < nd; i++)
	{
		SEXP document = VECTOR_ELT(documents, i);
        CHECKMATROW(document, Integer, 2);
        nw = INTEGER(GET_DIM(document))[1];

        docs[i].length = nw;
        docs[i].total = 0;
        docs[i].words = (int*)malloc(sizeof(int)*nw);
		docs[i].counts = (int*)malloc(sizeof(int)*nw);
		docs[i].gndlabel = labels[i];
		docs[i].lossAugLabel = -1;
        for (int ww=0; ww< nw; ww++){
        	word = INTEGER(document)[ww * 2];

      		count = INTEGER(document)[ww * 2 + 1];
      		

			word = word - OFFSET;
			docs[i].words[ww] = word;
			docs[i].counts[ww] = count;
			docs[i].total += count;
			if (word >= num_terms) num_terms = word + 1; 

        }
	} 	
	c->docs=docs;
	c->num_docs = nd;
	c->num_terms = num_terms;
}
#ifdef WINDOWS_VERSION
char* createDir(char* prefix){

				char name[] ="tmpXXXXXX"; 
				int rez;
				char* dir= new char[512];
				rez = _mktemp_s(name, sizeof(name));
				if (rez == 0) {
       				sprintf(dir, "%s_%s", prefix,name);
       				if( _mkdir(dir) != 0 ) 
     					error("Problem creating directory %s \n",dir);
				}
				else {
        			error("Temporary model directory name %s could not be created.", dir);
				}
				return dir;		
}
#else
char* createDir(char* prefix){          
				char name[] ="tmpXXXXXX"; 
				char * rez;
				char* dir= new char[512];
				rez = mktemp(name);
				if (rez != NULL) {
       				sprintf(dir, "%s_%s", prefix,rez);
       				if( mkdir(dir,0700) != 0 ) 
     					error("Problem creating directory %s \n",dir);
				}
				else {
        			error("Temporary model directory name %s could not be created.", dir);
				}
				return dir;		
}

#endif

void removeDirectory(char * directory){
	DIR *dir;
	struct dirent *ent;
    dir = opendir (directory);
	if (dir != NULL) {
  		while ((ent = readdir (dir)) != NULL) {
  			char name[512];
  			sprintf(name, "%s/%s", directory, ent->d_name);
    		remove(name);
  		}
   		closedir (dir);
    #ifdef WINDOWS_VERSION
         		_rmdir(directory);
    #else
         	 	rmdir(directory);
    #endif
}
}


extern "C" {
SEXP medSTCTrain(SEXP documents_,
			 SEXP labels_,
			 SEXP ntopics_,
			 SEXP class_num_,
			 SEXP initial_c_,
			 SEXP lambda_,
			 SEXP rho_,
			 SEXP nfolds_,
			 SEXP delta_ell_,
			 SEXP supervised_,
			 SEXP primal_svm_,
			 SEXP var_max_iter_,
			 SEXP convergence_,
			 SEXP em_max_iter_,
			 SEXP em_convergence_,
			 SEXP svm_alg_type_,
			 SEXP output_dir_){
  			
  			GetRNGstate();
  			CHECK(documents_, NewList);
			int nd = length(documents_);
			int* labels = new int[nd];
			if (!isNull(labels_)) {
    			if (length(labels_) != nd) {
      				error("class labels must have same length as documents.");
    			}
    			CHECKLEN(labels_, Integer, nd);	
    			for (int i=0; i<nd;i++){
    				labels[i]=INTEGER(labels_)[i];
    			}
    		}
    		Params param;
			CHECKLEN(ntopics_, Integer, 1);
  			param.NTOPICS  = INTEGER(ntopics_)[0];
  			CHECKLEN(class_num_, Integer, 1);
  			param.NLABELS = INTEGER(class_num_)[0];
  			CHECKLEN(initial_c_, Real, 1);
  			param.INITIAL_C = REAL(initial_c_)[0];
			CHECKLEN(lambda_, Real, 1);
  			param.LAMBDA = REAL(lambda_)[0];
  			CHECKLEN(rho_, Real, 1);
  			param.RHO= REAL(rho_)[0];
  			CHECKLEN(nfolds_, Integer, 1);
  			param.NFOLDS = INTEGER(nfolds_)[0];
  			CHECKLEN(delta_ell_, Real, 1);
  			param.DELTA_ELL = REAL(delta_ell_)[0];
  			CHECKLEN(supervised_, Logical, 1);
  			param.SUPERVISED= LOGICAL(supervised_)[0];
  			CHECKLEN(primal_svm_, Logical, 1);
  			param.PRIMALSVM = LOGICAL(primal_svm_)[0];
  			CHECKLEN(var_max_iter_, Integer, 1);
  			param.VAR_MAX_ITER = INTEGER(var_max_iter_)[0];
  			CHECKLEN(convergence_, Real, 1);
  			param.VAR_CONVERGED = REAL(convergence_)[0];
  			CHECKLEN(em_max_iter_, Integer, 1);
  			param.EM_MAX_ITER = INTEGER(em_max_iter_)[0];
  			CHECKLEN(em_convergence_, Real, 1);
  			param.EM_CONVERGED = REAL(em_convergence_)[0];
  			CHECKLEN(svm_alg_type_, Integer, 1);
  			param.SVM_ALGTYPE = INTEGER(svm_alg_type_)[0];
  			
  			char output_dir[512];
  			sprintf(output_dir,"%s",CHAR(STRING_ELT(output_dir_,0)));
 			Corpus *c = new Corpus();
		    read_data_from_R(c,documents_,nd,labels);	
			char *  dir = new char[512];
			sprintf(dir, "%s/s%d_c%d_f%d_s%d", output_dir, param.NTOPICS, (int) param.INITIAL_C,param.NFOLDS, param.SUPERVISED);
			char* dirTemp;
			dirTemp = createDir(dir);		
			sprintf(dir,"%s",dirTemp);
			MedSTC model;
			model.train("random", dir, c, &param);
			delete c;
			SEXP retval, double_model_parameters,integer_model_parameters,dLogProbW,dMu,dEta,directory;
			PROTECT(retval = allocVector(VECSXP, 6));
			SET_VECTOR_ELT(retval, 0, double_model_parameters = allocVector(REALSXP, 9));
			SET_VECTOR_ELT(retval, 1, integer_model_parameters = allocVector(INTSXP, 4));
			SET_VECTOR_ELT(retval, 2, dLogProbW = allocMatrix(REALSXP, model.m_nNumTerms,model.m_nK));
			SET_VECTOR_ELT(retval, 3, dMu = allocVector(REALSXP, model.m_nDim * model.m_nLabelNum));
			SET_VECTOR_ELT(retval, 4, dEta = allocVector(REALSXP, model.m_nK * model.m_nLabelNum));		
			SET_VECTOR_ELT(retval, 5, directory = allocVector(STRSXP, 1));
			
			REAL(double_model_parameters)[0]=model.m_dDeltaEll;
			REAL(double_model_parameters)[1]=model.m_dLambda;
			REAL(double_model_parameters)[2]=model.m_dRho;
			REAL(double_model_parameters)[3]=model.m_dGamma;
			REAL(double_model_parameters)[4]=model.m_dC;
			REAL(double_model_parameters)[5]=model.m_dLogLoss;
			REAL(double_model_parameters)[6]=model.m_dB;
			REAL(double_model_parameters)[7]=model.m_dPoisOffset;
			REAL(double_model_parameters)[8]=model.m_dsvm_primalobj;
			
			INTEGER(integer_model_parameters)[0]=model.m_nK;
			INTEGER(integer_model_parameters)[1]=model.m_nLabelNum;
			INTEGER(integer_model_parameters)[2]=model.m_nNumTerms;
			INTEGER(integer_model_parameters)[3]=model.m_nDim;
			int i,j;
			for(i=0; i<model.m_nNumTerms;i++)
				for (j=0; j<model.m_nK; j++) 
				REAL(dLogProbW)[i+model.m_nNumTerms*j]=model.m_dLogProbW[i][j];
			for (i=0; i<model.m_nDim*model.m_nLabelNum;i++)
				REAL(dMu)[i]=model.m_dMu[i];	
			for (i=0; i<model.m_nK*model.m_nLabelNum;i++)
				REAL(dEta)[i]=model.m_dEta[i];	
			SET_STRING_ELT(directory, 0, mkChar(dir));
			removeDirectory(dir);
			UNPROTECT(1);
			return retval;
			
			}
}
extern "C" {
SEXP medSTCTest(SEXP model,SEXP documents_,SEXP labels_, 
			 SEXP ntopics_,
			 SEXP class_num_,
			 SEXP initial_c_,
			 SEXP lambda_,
			 SEXP rho_,
			 SEXP nfolds_,
			 SEXP delta_ell_,
			 SEXP supervised_,
			 SEXP primal_svm_,
			 SEXP var_max_iter_,
			 SEXP convergence_,
			 SEXP em_max_iter_,
			 SEXP em_convergence_,
			 SEXP svm_alg_type_,
			 SEXP output_dir_){

			int m_nK,m_nLabelNum,m_nNumTerms,m_nDim;
			double **m_dLogProbW;
			double m_dDeltaEll, m_dLambda,m_dRho, m_dGamma,m_dC,m_dLogLoss,m_dB,m_dPoisOffset,m_dsvm_primalobj;
			double *m_dMu;
			double *m_dEta;
			
			Params param;
			CHECKLEN(ntopics_, Integer, 1);
  			param.NTOPICS  = INTEGER(ntopics_)[0];
  			CHECKLEN(class_num_, Integer, 1);
  			param.NLABELS = INTEGER(class_num_)[0];
  			CHECKLEN(initial_c_, Real, 1);
  			param.INITIAL_C = REAL(initial_c_)[0];
			CHECKLEN(lambda_, Real, 1);
  			param.LAMBDA = REAL(lambda_)[0];
  			CHECKLEN(rho_, Real, 1);
  			param.RHO= REAL(rho_)[0];
  			CHECKLEN(nfolds_, Integer, 1);
  			param.NFOLDS = INTEGER(nfolds_)[0];
  			CHECKLEN(delta_ell_, Real, 1);
  			param.DELTA_ELL = REAL(delta_ell_)[0];
  			CHECKLEN(supervised_, Logical, 1);
  			param.SUPERVISED= LOGICAL(supervised_)[0];
  			CHECKLEN(primal_svm_, Logical, 1);
  			param.PRIMALSVM = LOGICAL(primal_svm_)[0];
  			CHECKLEN(var_max_iter_, Integer, 1);
  			param.VAR_MAX_ITER = INTEGER(var_max_iter_)[0];
  			CHECKLEN(convergence_, Real, 1);
  			param.VAR_CONVERGED = REAL(convergence_)[0];
  			CHECKLEN(em_max_iter_, Integer, 1);
  			param.EM_MAX_ITER = INTEGER(em_max_iter_)[0];
  			CHECKLEN(em_convergence_, Real, 1);
  			param.EM_CONVERGED = REAL(em_convergence_)[0];
  			CHECKLEN(svm_alg_type_, Integer, 1);
  			param.SVM_ALGTYPE = INTEGER(svm_alg_type_)[0];
  			
  			char output_dir[512];
  			sprintf(output_dir,"%s",CHAR(STRING_ELT(output_dir_,0)));
  			CHECK(documents_, NewList);
			int nd = length(documents_);
			int* labels = new int[nd];
			if (!isNull(labels_)) {
    			if (length(labels_) != nd) {
      				error("class labels must have same length as documents.");
    			}
    			CHECKLEN(labels_, Integer, nd);	
    			for (int i=0; i<nd;i++){
    				labels[i]=INTEGER(labels_)[i];
    				}
    		}
    		
    		Corpus *c = new Corpus();
			read_data_from_R(c,documents_,nd,labels);	
			char dir[512];
			sprintf(dir, "%s/s%d_c%d_f%d_s%d", output_dir, param.NTOPICS, (int) param.INITIAL_C,param.NFOLDS, param.SUPERVISED);
			char* dirTemp;
			dirTemp = createDir(dir);		
			sprintf(dir,"%s",dirTemp);	
			
					
			SEXP double_model_parameters = VECTOR_ELT(model,0);
				m_dDeltaEll = REAL(double_model_parameters)[0];
				m_dLambda = REAL(double_model_parameters)[1];
				m_dRho = REAL(double_model_parameters)[2];
				m_dGamma = REAL(double_model_parameters)[3];			
				m_dC = REAL(double_model_parameters)[4];
				m_dLogLoss = REAL(double_model_parameters)[5];
				m_dB = REAL(double_model_parameters)[6];
				m_dPoisOffset = REAL(double_model_parameters)[7];
				m_dsvm_primalobj = REAL(double_model_parameters)[8];
			SEXP integer_model_parameters = VECTOR_ELT(model,1);
				m_nK = INTEGER(integer_model_parameters)[0];
				m_nLabelNum = INTEGER(integer_model_parameters)[1];
				m_nNumTerms = INTEGER(integer_model_parameters)[2];
				m_nDim = INTEGER(integer_model_parameters)[3];
			
			m_dLogProbW = (double**)malloc(sizeof(double*)*m_nNumTerms);
			m_dEta = (double*)malloc(sizeof(double) * m_nK * m_nLabelNum);
			m_dMu = (double*)malloc(sizeof(double) * m_nDim * m_nLabelNum);
			SEXP dLogProbW = VECTOR_ELT(model,2);
				for (int i=0; i<m_nNumTerms; i++) {
					m_dLogProbW[i] = (double*)malloc(sizeof(double)*m_nK);
					for (int j=0; j<m_nK; j++) 
						m_dLogProbW[i][j] = REAL(dLogProbW)[i+m_nNumTerms*j];
				}
			SEXP dMu = VECTOR_ELT(model,3);
				for (int i=0; i<m_nDim; i++)
					for (int j=0; j<m_nLabelNum; j++)
						m_dMu[i*m_nLabelNum + j] = REAL(dMu)[i*m_nLabelNum + j];

			SEXP dEta = VECTOR_ELT(model,4);
			for ( int i=0; i<m_nK; i++ ) {	
					for (int j=0; j<m_nLabelNum; j++) 
						m_dEta[i*m_nLabelNum + j] = REAL(dEta)[i*m_nLabelNum + j];
				}
				
			
			
			MedSTC evlModel = MedSTC(m_nK,m_nLabelNum,m_nNumTerms,m_nDim,m_dDeltaEll,m_dLambda,m_dRho,m_dGamma,m_dC,
			m_dLogLoss,m_dB,m_dPoisOffset,m_dsvm_primalobj, 
			m_dLogProbW, m_dMu,m_dEta, dir);
			
	
			evlModel.predictTest(c, &param);
			
			SEXP retval;
  			PROTECT(retval = allocMatrix(REALSXP, nd, param.NLABELS));
  			for (int i=0; i < c->num_docs; i++){
  				for (int j=0; j<param.NLABELS; j++)
  				REAL(retval)[i+c->num_docs*j] =  c->docs[i].scores[j];
         	}
         	delete c;
         	removeDirectory(dir);
         	UNPROTECT(1);
			return retval;	
  	
}
}



    
	
