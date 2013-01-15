\name{medSTC}
\alias{medSTC}
\Rdversion{1.0}

\title{
  A max-margin Sparse Topical Coding model (Med-STC) 
  for multi-class classification
}
\description{
MedSTC is a novel classification algorithm by Prof. Jun Zhu (http://www.ml-thu.net/~jun/).
}
\usage{
medSTC(documents, mlabels, ntopics, initial_c=0.5, lambda=1, rho=0.01, delta_ell=3600, supervised=TRUE, 
primal_svm=1, var_max_iter=20, convergence=1e-4, em_max_iter=100, em_convergence=1e-4, 
svm_alg_type=2, output_dir=".") 

}

\arguments{
  \item{documents}{
    A list whose length is equal to the number of documents, D.   Each
  element of \var{documents} is an integer matrix with two rows.  Each
  column of \var{documents[[i]]} (i.e., document \eqn{i}) represents a
  word occurring in the document.

  \var{documents[[i]][1, j]} is a
  0-indexed word identifier for the jth word in document i. 
  \var{documents[[i]][2,j]} is an integer specifying the number of times that word appears in
  the document.
  }
   \item{mlabels}{
   		The training labels for the \var{documents}. 
   }
   
    \item{ntopics}{
   		Number of topics to be used in modeling the corpus.
   }
   
    \item{initial_c, lambda, rho}{
   		These are positive-valued regularization constants.
   		Default values are initial_c=0.5, lambda=0.1, rho=0.01
   }
    \item{delta_ell}{
   		The parameter for the svm cost function, i.e., 0/(delta ell) loss. 
   		Only positive values are allowed. Default value is 3600.
   }
   
   \item{supervised}{ 
     If the value is TRUE, the model is a supervised MedSTC; if FALSE, the model 
     is the unsupervised STC.
	}
     
    \item{primal_svm}{
     Only works when "supervised" is set at 1. If the value is 1, uses the
     loss-augmented prediction (i.e., sub-gradient) to update document codes;
     otherwise it uses the gradient with Lagrangian multipliers to update document codes.
     }
     \item{var_max_iter}{
     The maximum number of iterations of coordinate descent for a single document.
	}
     \item{convergence}{
     The convergence criteria for coordinate descent.  Stop if
     (objective_old - objective) / abs(objective_old) is less than this value (or
     after the maximum number of iterations). Note that "objective" is the objective 
     value for a single document.
	}
    \item{em_max_iter}{
     The maximum number of iterations of hierarchical sparse coding, dictionary learning, 
     and svm training (for supervised MedSTC).
	}
     \item{em_convergence}{
     The convergence criteria for coordinate descent. Stop if (objective_old -
     objective) / abs(objective_old) is less than this value (or after the
     maximum number of iterations).  Note that "objective" is the objective value 
     for the whole corpus.
	} 
	\item{svm_alg_type}{
		If set to 0 then the n-slack multi-class SVM is used. If set to 2,
     	then the 1-slack multi-class SVM is used. In our testing, the 1-slack
     	SVM is faster.
	}
	\item{output_dir}{
		A directory for writing intermediate results. 
		Directory is removed after the calculation is done, but is needed during the run.
	}
}
\value{
	\item{model}{
		A model object of the medSTC class, which has a state list with five elements:
		The first two list elements are for storing the model parameter state after the model completed training.
		The third list element is the LogProbabilityOfWordsForTopics, which can be used for 
		word assignments to topics.
		The fourth and fifth model state list elements are Eta and Mu. (refer to paper)
    	The model also stores the original paramater values.
	}

}
\references{
  \cite{Jun Zhu, and Eric P. Xing. 
  		Sparse Topical Coding, 
  		In Proc. of 27th Conference on Uncertainty in 
  		Artificial Intelligence (UAI), Barcelona, Spain, 2011.}
}
\author{
   Jun Zhu (\email{junzhu@cs.cmu.edu}),Aykut Firat (\email{aykutfirat@gmail.com})  
}

\examples{

\dontrun{demo(medSTC)}

}
\keyword{ sparse topical coding models }

