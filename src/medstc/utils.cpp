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


#include "utils.h"

double safe_entropy(double *dist, const int &K)
{
	double dNorm = 0;
	for ( int k=0; k<K; k++ ) {
		dNorm += dist[k];
	}

	double dEnt = 0;
	for ( int k=0; k<K; k++ ) {
		double dval = dist[k] / dNorm;
		if ( dist[k] > 1e-30 ) dEnt -= dval * log( dval );
	}

	return dEnt;
}

double log_sum(double log_a, double log_b)
{
	double dval = 0;

	if (log_a < log_b) {
		dval = log_b + log(1 + exp(log_a - log_b));
	} else {
		dval = log_a + log(1 + exp(log_b - log_a));
	}
	return dval;
}

double trigamma(double x)
{
	double p;
	int i;

	x=x+6;
	p=1/(x*x);
	p=(((((0.075757575757576*p-0.033333333333333)*p+0.0238095238095238)
		*p-0.033333333333333)*p+0.166666666666667)*p+1)/x+0.5*p;
	for (i=0; i<6 ;i++)
	{
		x=x-1;
		p=1/(x*x)+p;
	}
	return(p);
}

double digamma(double x)
{
	double p;
	x=x+6;
	p=1/(x*x);
	p=(((0.004166666666667*p-0.003968253986254)*p+
		0.008333333333333)*p-0.083333333333333)*p;
	p=p+log(x)-0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6);
	return p;
}




int argmax(double* x, const int &n)
{
	double max = x[0];
	int argmax = 0;
	for (int i=1; i<n; i++) {
		if (x[i] > max) {
			max = x[i];
			argmax = i;
		}
	}

	return argmax;
}

double dotprod(double *a, double *b, const int&n)
{
	double dres = 0;
	for ( int i=0; i<n; i++ ) {
		dres += a[i] * b[i];
	}
	return dres;
}
/* a vector times a (n x n) square matrix  */
void matrixprod(double *a, double **A, double *res, const int &n)
{
	for ( int i=0; i<n; i++ ) {
		res[i] = 0;
		for ( int j=0; j<n; j++ ) {
			res[i] += a[j] * A[j][i];
		}
	}
}
/* a (n x n) square matrix times a vector. */
void matrixprod(double **A, double *a, double *res, const int &n)
{
	for ( int i=0; i<n; i++ ) {
		res[i] = 0;
		for ( int j=0; j<n; j++ ) {
			res[i] += a[j] * A[i][j];
		}
	}
}

/* A + ab^\top*/
void addmatrix(double **A, double *a, double *b, const int &n, double factor)
{
	for (int i=0; i<n; i++ ) {
		for ( int j=0; j<n; j++ ) {
			A[i][j] += a[i] * b[j] * factor;
		}
	}
}

/* A + ab^\top + ba^\top*/
void addmatrix2(double **A, double *a, double *b, const int &n, double factor)
{
	for (int i=0; i<n; i++ ) {
		for ( int j=0; j<n; j++ ) {
			A[i][j] += (a[i] * b[j] + b[i] * a[j]) * factor;
		}
	}
}

double L2Norm( double *x, const int &K )
{
	double dVal = 0;
	for ( int k=0; k<K; k++ ) {
		dVal += x[k] * x[k];
	}
	return dVal;
}

double L1Norm( double *x, const int &K )
{
	double dVal = 0;
	for ( int k=0; k<K; k++ ) {
		dVal += max(0.0, x[k]);
	}
	return dVal;
}

double L2Dist( double *x, double *m, const int &K )
{
	double dVal = 0;
	for ( int k=0; k<K; k++ ) {
		dVal += (x[k] - m[k]) * (x[k] - m[k]);
	}
	return dVal;
}
double log_poisson( const double &lambda, const int &x )
{
	return (x * log(lambda) - lambda);
}

void quickSort(double *arr, int left, int right) 
{
	int i = left, j = right;
	double tmp;
	double pivot = arr[(left + right) / 2];

	/* partition */
	while (i <= j) 
	{
		while( arr[i] < pivot )
			i ++;

		while( arr[j] > pivot )
			j --;

		if (i <= j) {
			tmp = arr[i];
			arr[i] = arr[j];
			arr[j] = tmp;

			i ++;
			j --;
		}
	};

	/* recursion */
	if (left < j)
		quickSort(arr, left, j);

	if (i < right)
		quickSort(arr, i, right);
}

void printf_mat(char *filename, double **mat, const int &n1, const int &n2)
{
	FILE *fptr = fopen(filename, "w");
	for ( int i=0; i<n1; i++ ) {
		double *dPtr = mat[i];
		for ( int j=0; j<n2; j++ ) {
			fprintf(fptr, "%.10f ", dPtr[j]);
		}
		fprintf(fptr, "\n");
	}
	fclose( fptr );
}
