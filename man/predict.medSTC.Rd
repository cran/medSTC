\name{predict.medSTC}
\alias{predict.medSTC}
\Rdversion{1.0}

\title{
  Prediction for max-margin Sparse Topical Coding model (Med-STC) for multi-class classification
}
\description{
MedSTC is a novel classification algorithm by Prof. Jun Zhu (http://www.ml-thu.net/~jun/).
This function predicts test class labels using a medSTC model.
}
\usage{

\method{predict}{medSTC}(object,documents,\dots) 

}

\arguments{ 

  \item{object}{
   		A model object of the medSTC class, which is obtained after running \code{medSTC}.
   }
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
  \item{...}{
  	Currently not used.
  }
}
\value{
	\item{scores}{
		Score values for each label for each document.
	}
	\item{assignments}{
		Predicted class labels for each document.
	}
}
\references{
  \cite{Jun Zhu, and Eric P. Xing. 
  		Sparse Topical Coding, 
  		In Proc. of 27th Conference on Uncertainty in 
  		Artificial Intelligence (UAI), Barcelona, Spain, 2011.}
}
\author{
   Jun Zhu (\email{junzhu@cs.cmu.edu}), Aykut Firat (\email{aykut@crimsonhexagon.com})  
}

\examples{

\dontrun{demo(medSTC)}

}
\keyword{ sparse topical coding models }

