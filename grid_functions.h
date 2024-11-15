#include<stdio.h>
#include<stdlib.h>

#define DE
#define DUST

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
  #ifdef DUST
  double *d_dust_0;
  double *d_dust_1;
  double *d_dust_2;
  double *d_dust_3;
  #endif
} Conserved;


void Read_Header(char *filename, int *nx, int *ny, int *nz, int *x_off, int *y_off, int *z_off, int *nx_local, int *ny_local, int *nz_local);

void Read_Grid(char *filename, Conserved C, int nx_local, int ny_local, int nz_local);

