#pragma once

#include <stdlib.h>
#include <iostream>
#include <exception>
using namespace std;

class ExceptionWithIflag : public exception
{
public:
	int iflag;
	wstring content;
	ExceptionWithIflag();
	ExceptionWithIflag( int i, wstring s );
	~ExceptionWithIflag() throw();
	wstring toString();
};

class CLBFGSCPP;

class CMcsrch
{
public:
	CMcsrch(void);
	~CMcsrch(void);

private:
	int* infoc;// = new int[1], 
	int j;//  = 0;
	double dg, dgm, dginit, dgtes, dgtest;
	double* dgx, *dgxm, *dgy, *dgym;
	double finit, ftest1, fm;
	double *fx, *fxm, *fy, *fym;
	double p5, p66;
	double *stx, *sty;
	double stmin, stmax;
	double width, width1, xtrapf;
	bool *brackt;
	bool stage1;

	double sqr( double x ) ;
	double max3( double x, double y, double z );

public:
	void mcsrch ( int n , double x[] , double f , double g[] , double s[] , int is0 , 
		double stp[] , double ftol , 
		double xtol , int maxfev , int info[] , int nfev[] , double wa[] );

	void mcstep ( double stx[] , double fx[] , double dx[] , double sty[] , 
		double fy[] , double dy[] , double stp[] , double fp , double dp , 
		bool brackt[] , double stpmin , double stpmax , int info[] );
};
class CLBFGSCPP
{
public:
	CLBFGSCPP(void);
	~CLBFGSCPP(void);
	/** Specialized exception class for LBFGS; contains the
	  * <code>iflag</code> value returned by <code>lbfgs</code>.
	  */

	/** Controls the accuracy of the line search <code>mcsrch</code>. If the
	  * function and gradient evaluations are inexpensive with respect
	  * to the cost of the iteration (which is sometimes the case when
	  * solving very large problems) it may be advantageous to set <code>gtol</code>
	  * to a small value. A typical small value is 0.1.  Restriction:
	  * <code>gtol</code> should be greater than 1e-4.
	  */
private:
	CMcsrch m_mcsrch;
public:
	static double gtol;

	/** Specify lower bound for the step in the line search.
	  * The default value is 1e-20. This value need not be modified unless
	  * the exponent is too large for the machine being used, or unless
	  * the problem is extremely badly scaled (in which case the exponent
	  * should be increased).
	  */

	static double stpmin ;

	/** Specify upper bound for the step in the line search.
	  * The default value is 1e20. This value need not be modified unless
	  * the exponent is too large for the machine being used, or unless
	  * the problem is extremely badly scaled (in which case the exponent
	  * should be increased).
	  */

	static double stpmax;

	/** The solution vector as it was at the end of the most recently
	  * completed line search. This will usually be different from the
	  * return value of the parameter <tt>x</tt> of <tt>lbfgs</tt>, which
	  * is modified by line-search steps. A caller which wants to stop the
	  * optimization iterations before <tt>LBFGS.lbfgs</tt> automatically stops
	  * (by reaching a very small gradient) should copy this vector instead
	  * of using <tt>x</tt>. When <tt>LBFGS.lbfgs</tt> automatically stops,
	  * then <tt>x</tt> and <tt>solution_cache</tt> are the same.
	  */
	double* solution_cache;

private:
	double gnorm, stp1, ftol, *stp, ys, yy, sq, yr, beta, xnorm;
	int iter, nfun, point, ispt, iypt, maxfev, *info, bound, npt, cp, i, *nfev, inmc, iycn, iscn;
	bool finish;
	
	double* w;

	/** This method returns the total number of evaluations of the objective
	  * function since the last time LBFGS was restarted. The total number of function
	  * evaluations increases by the number of evaluations required for the
	  * line search; the total is only increased after a successful line search.
	  */
public:
	int nfevaluations() { return nfun; }
	
	void lbfgs ( int n , int m , double x[] , double f , double g[] , 
						  bool diagco , double diag[] , 
					int iprint[] , double eps , double xtol , int iflag[]);//throw (ExceptionWithIflag* e);

	void lb1 ( int iprint[] , int iter , int nfun , double gnorm , int n , 
				  int m , double x[] , double f , double g[] , double stp[] , bool finish );

	void daxpy ( int n , double da , double dx[] , int ix0, int incx , 
						  double dy[] , int iy0, int incy );

	double ddot ( int n, double dx[], int ix0, int incx, 
						   double dy[], int iy0, int incy );
};
