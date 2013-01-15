\name{newsgroups}
\Rdversion{1.1}
\alias{newsgroups}
\alias{newsgroup.train.documents}
\alias{newsgroup.test.documents}
\alias{newsgroup.train.labels}
\alias{newsgroup.test.labels}
\alias{newsgroup.vocab}
\alias{stopwords}
\docType{data}
\title{
  A shortened collection of newsgroup messages with the first 3 classes.
}
\description{
  The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, 
  partitioned (nearly) evenly across 20 different newsgroups. We use in this package only 
  its first 3 classes for demonstration purposes.
}
\usage{
data(newsgroup.train.documents)
data(newsgroup.test.documents)
data(newsgroup.train.labels)
data(newsgroup.test.labels)
data(newsgroup.vocab)
}
\format{
  \code{newsgroup.train.documents} and \code{newsgroup.test.documents}
  comprise a corpus of 2731 newsgroup documents partitioned into 1633 training 
  and 1098 test cases evenly distributed across 3 classes. 

  \code{newsgroup.train.labels} is a numeric vector of length 1633 which gives
  a class label from 1 to 3 for each training document in the corpus.
  
  \code{newsgroup.test.labels} is a numeric vector of length 1098 which gives
  a class label from 1 to 3 for each test document in the corpus.
  
  \code{newsgroup.vocab} is the vocabulary of the corpus.
  
  \code{stopwords} English stopwords extracted from the tm package.
  
}
\source{
  \cite{http://qwone.com/~jason/20Newsgroups/}
}

\examples{
data(newsgroup.train.documents)
data(newsgroup.test.documents)
data(newsgroup.train.labels)
data(newsgroup.test.labels)
data(newsgroup.vocab)
data(stopwords)
}
\keyword{datasets}
