#include<stdio.h>
#include<stdlib.h>

#define DE
#define SCALAR

typedef struct 
{
  double *d;
  double *mx;
  double *my;
  double *mz;			
  double *E;
  #ifdef DE
  double *gE;
  #endif
  #ifdef SCALAR
  double *c;
  #endif

} Conserved;


void Read_Header(char *filename, int *nx, int *ny, int *nz, double *xlen, double *ylen, double *zlen);

void Read_Grid(char *filename, Conserved C, int nx, int ny, int nz, int x_off, int y_off, int z_off);

