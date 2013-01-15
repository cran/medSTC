/***********************************************************************/
/*                                                                     */
/*   svm_struct_common.h                                               */
/*                                                                     */
/*   Functions and types used by multiple components of SVM-struct.    */
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 03.07.04                                                    */
/*                                                                     */
/*   Copyright (c) 2004  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <R.h>

#include "svm_struct_common.h"

long struct_verbosity=0;                   /* verbosity level (0-4) */

void printIntArray(int* x, int n)
{
  int i;
  for(i=0;i<n;i++)
    Rprintf("%i:",x[i]);
}

void printDoubleArray(double* x, int n)
{
  int i;
  for(i=0;i<n;i++)
    Rprintf("%f:",x[i]);
}

void printWordArray(TOKEN* x)
{
  int i=0;
  for(;x[i].wnum!=0;i++)
    if(x[i].weight != 0)
      Rprintf(" %i:%.2f ",(int)x[i].wnum,x[i].weight);
}

void printW(double *w, long sizePhi, long n,double C)
{
  int i;
  Rprintf("---- w ----\n");
  for(i=0;i<sizePhi;i++)
    {
      Rprintf("%f  ",w[i]);
    }
  Rprintf("\n----- xi ----\n");
  for(;i<sizePhi+2*n;i++)
    {
      Rprintf("%f ",1/sqrt(2*C)*w[i]);
    }
  Rprintf("\n");

}
/**** end print methods ****/

