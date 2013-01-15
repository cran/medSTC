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


#define OFFSET 0;                  // offset for reading data

typedef struct
{
	int gndlabel;	  // the ground truth response variable value
	int predlabel;	  // the predicted response variable value
	int lossAugLabel; // the loss-augmented prediction
	double lhood;
	double* scores;
    int* words;
    int* counts;
    int length;
    int total;
} Document;


class Corpus
{
public:
	Corpus(void);
public:
	~Corpus(void);

	void read_data(char* data_filename, int nLabels);
	void read_flickr_data(char* data_filename, int nLabels);
	Corpus* get_traindata(const int&nfold, const int &foldix);
	Corpus* get_traindata2(const int&nfold, const int &foldix);
	Corpus* get_testdata(const int&nfold, const int &foldix);
	Corpus* get_testdata2(const int&nfold, const int &foldix);
	void reorder(char *filename);

	int max_corpus_length( );

	void shuffle();

public:
    Document* docs;
    int num_terms;
    int num_docs;
};
