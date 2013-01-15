\name{medSTC-package}
\Rdversion{1.0}
\alias{medSTC-package}
\docType{package}
\title{
  Max-margin supervised Sparse Topical Coding model (Med-STC) 
}
\description{
  This package employs sparse topical coding models for multi-class classification 
  developed by Jun Zhu and Eric P. Xing. The package uses a fast C implementation of SVM, 
  SVMlight (http://svmlight.joachims.org/), developed by Thorsten Joachims <thorsten@joachims.org>.  
}
\details{
\tabular{ll}{
Package: \tab medSTC\cr
Type: \tab Package\cr
Version: \tab 1.0.0\cr
Date: \tab 2013-1-15\cr
License: \tab GPL \cr
LazyLoad: \tab yes\cr
}
}
\author{
  Jun Zhu (\email{junzhu@cs.cmu.edu}),
  Aykut Firat (\email{aykut@crimsonhexagon.com})
}
\references{
  \cite{Jun Zhu, and Eric P. Xing. 
  Sparse Topical Coding, 
  In Proc. of 27th Conference on Uncertainty in 
  Artificial Intelligence (UAI), Barcelona, Spain, 2011.}
  \cite{ T. Joachims, Making large-Scale SVM Learning
       Practical. Advances in Kernel Methods - Support Vector
       Learning, B. Scholkopf and C. Burges and A. Smola (ed.),
       MIT-Press, 1999. 
       http://www-ai.cs.uni-dortmund.de/DOKUMENTE/joachims_99a.pdf} 
}
\keyword{ package }
\seealso{
Functions to fit models:
\code{\link{medSTC}}

Functions for prediction:
\code{\link{predict.medSTC}}

Included data sets:
\code{\link{newsgroups}}
}
\examples{
## See the demo for the newsgroup example:
\dontrun{demo(medSTC)}
}