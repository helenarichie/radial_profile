#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<mpi.h>
#include<hdf5.h>
#include<gsl_statistics_double.h>
#include"grid_functions.h"

#define COLD
int compare(const void *a, const void *b) {
  if (*(double*)a > *(double*)b) return 1;
  else if (*(double*)a < *(double*)b) return -1;
  else return 0; 
}


int main(int argc, char *argv[]) {
  int np_x, np_y, np_z;
  setbuf(stdout, NULL);
  if (argc < 2){
    printf("Add filename as arg (exit) \n");
    exit(-1);
  }
  if (argc < 3){
    printf("Setting np_x = 1\n");
    np_x = 1;
  } else {
    np_x = atoi(argv[2]);
  }
  np_y = np_x;
  if (argc < 4){
    printf("Setting np_z = 1\n");
    np_z = 1;
    printf("Usage: profiles filename np_x np_z \n");
  } else {
    np_z = atoi(argv[3]);
  }
  printf("Filename: %s np_x: %i np_z: %i\n",argv[1],np_x,np_z);
  
  // mpi stuff
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Conserved C;
  int nx;
  int ny;
  int nz;
  int x_off = 0;
  int y_off = 0;
  int z_off = 0;
  double d, vx, vy, vz, n, T, P, S, c, cs, M;
  double x_pos, y_pos, z_pos, r, vr, phi;
  double dx;
  double x_len = 5.;
  double y_len = 5.;
  double z_len = 5.;
  double cone = 30.;
  double r_av;
  double n_av, n_med, n_lo, n_hi;
  double v_av, v_med, v_lo, v_hi;
  double T_av, T_med, T_lo, T_hi;
  double P_av, P_med, P_lo, P_hi;
  double S_av, S_med, S_lo, S_hi;
  double c_av, c_med, c_lo, c_hi;
  double cs_av, cs_med, cs_lo, cs_hi;
  double M_av, M_med, M_lo, M_hi;

  // some constants
  double l_s = 3.086e21; // length scale, centimeters in a kiloparsec
  double m_s = 1.99e33; // mass scale, g in a solar mass
  double t_s = 3.154e10; // time scale, seconds in a kyr
  double d_s = m_s / (l_s*l_s*l_s); // density scale, M_sun / kpc^3
  double v_s = l_s / t_s; // velocity scale, kpc / kyr
  double p_s = d_s*v_s*v_s; // pressure scale, M_sun / kpc kyr^2
  double mp = 1.67e-24; // proton mass in grams
  double KB = 1.3806e-16; // boltzmann constant in cm^2 g / s^2 K
  double v_to_kmps = l_s/t_s/100000.;
  double kmps_to_kpcpkyr = 1.0220122e-6;
  double m_c = d_s*l_s*l_s*l_s / m_s; // characteristic mass in solar masses
  double mu = 0.6;
  double gamma = 5./3.;

  double Tcold = 2e4;
  double Thot = 5e5;
  // Random number generator
  //Ran quickran(0);
  //double prob;

  // Read in some header info
  char filename[200];
  //strcpy(filename, "./2048_new/hdf5/70.h5");
  strcpy(filename, argv[1]);
  Read_Header(filename, &nx, &ny, &nz, &x_len, &y_len, &z_len);
  //printf("xyzlen %d %d %d",x_len,y_len,z_len);
  //printf("Read_Header\n");
  dx = x_len / nx;

  // set number of processes in each direction (for splitting)
  nx = nx/np_x;
  ny = ny/np_y;
  nz = nz/np_z;
  int *ix, *iy, *iz;
  ix = (int *)malloc(size*sizeof(int));
  iy = (int *)malloc(size*sizeof(int));
  iz = (int *)malloc(size*sizeof(int));
  // printf("malloc ix iy iz\n");
  int np=0;
  for(int i=0;i<np_x;i++) {
    for(int j=0;j<np_y;j++) {
      for(int k=0;k<np_z;k++) {
        ix[np] = i;
        iy[np] = j;
        iz[np] = k;
        np++;
      }
    }
  }
  for(int i=0; i<size; i++) {
    x_off = ix[rank]*nx;
    y_off = iy[rank]*ny;
    z_off = iz[rank]*nz;
  }
  #ifdef HOT
  printf("HOT \n");
  #else
  printf("COLD \n");
  #endif

  free(ix);
  free(iy);
  free(iz);

  // Allocate memory for the grid
  C.d  = (double *) malloc(nz*ny*nx*sizeof(double));
  C.mx = (double *) malloc(nz*ny*nx*sizeof(double));
  C.my = (double *) malloc(nz*ny*nx*sizeof(double));
  C.mz = (double *) malloc(nz*ny*nx*sizeof(double));
  C.E  = (double *) malloc(nz*ny*nx*sizeof(double));
  #ifdef DE
  C.gE = (double *) malloc(nz*ny*nx*sizeof(double));
  #endif
  #ifdef SCALAR
  C.c = (double *) malloc(nz*ny*nx*sizeof(double));
  #endif

  // Read in the grid data
  Read_Grid(filename, C, nx, ny, nz, x_off, y_off, z_off);

  // Create arrays to hold cell values as a function of radius
  int N_bins = 80;
  long int r_bins[N_bins];
  int bin;
  for (int i=0; i<N_bins; i++) {
    r_bins[i] = 0;
  }

  // Loop over cells and count cells in each radial bin 
  for (int i=0; i<nx; i++) {
    for (int j=0; j<ny; j++) {
      for (int k=0; k<nz; k++) {
        int id = k + j*nz + i*nz*ny;
        x_pos = (0.5+i+x_off)*dx - x_len/2.;
        y_pos = (0.5+j+y_off)*dx - y_len/2.;
        z_pos = (0.5+k+z_off)*dx - z_len/2.;
        r = sqrt(x_pos*x_pos + y_pos*y_pos + z_pos*z_pos);
        double rbin = 8*r;
        bin = int(rbin);
        if (bin < N_bins) {
          phi = acos(fabs(z_pos)/r);
          if (phi < cone*3.1416/180.) {
            n  = C.d[id]*d_s / (mu*mp);
            T  = C.gE[id]*(gamma-1.0)*p_s/(n*KB);
	    #ifdef HOT
            if (T > Thot) {
	    #else
            if (T < Tcold) {	      
	    #endif
              r_bins[bin]++; // add one to this radial bin count
            }
          }
        }
      }
    }
  }


  double **r_array = (double **)malloc(N_bins*sizeof(double));
  double **n_array = (double **)malloc(N_bins*sizeof(double));
  double **v_array = (double **)malloc(N_bins*sizeof(double));
  double **T_array = (double **)malloc(N_bins*sizeof(double));
  double **P_array = (double **)malloc(N_bins*sizeof(double));
  double **S_array = (double **)malloc(N_bins*sizeof(double));
  double **c_array = (double **)malloc(N_bins*sizeof(double));
  double **cs_array = (double **)malloc(N_bins*sizeof(double));
  double **M_array = (double **)malloc(N_bins*sizeof(double));
  int cell_count[N_bins];
  // allocate data for each radial bin
  for (int bb=0; bb<N_bins; bb++) {
    r_array[bb] = (double*)malloc(r_bins[bb]*sizeof(double));
    n_array[bb] = (double*)malloc(r_bins[bb]*sizeof(double));
    v_array[bb] = (double*)malloc(r_bins[bb]*sizeof(double));
    T_array[bb] = (double*)malloc(r_bins[bb]*sizeof(double));
    P_array[bb] = (double*)malloc(r_bins[bb]*sizeof(double));
    S_array[bb] = (double*)malloc(r_bins[bb]*sizeof(double));
    c_array[bb] = (double*)malloc(r_bins[bb]*sizeof(double));
    cs_array[bb] = (double*)malloc(r_bins[bb]*sizeof(double));
    M_array[bb] = (double*)malloc(r_bins[bb]*sizeof(double));
    cell_count[bb] = 0;
  }

  // Loop over cells and assign values to radial bins
  for (int i=0; i<nx; i++) {
    for (int j=0; j<ny; j++) {
      for (int k=0; k<nz; k++) {
        int id = k + j*nz + i*nz*ny;
        x_pos = (0.5+i+x_off)*dx - x_len/2.;
        y_pos = (0.5+j+y_off)*dx - y_len/2.;
        z_pos = (0.5+k+z_off)*dx - z_len/2.;
        r = sqrt(x_pos*x_pos + y_pos*y_pos + z_pos*z_pos);
        double rbin = 8*r;
        bin = int(rbin);
        if (bin < N_bins) {
          phi = acos(fabs(z_pos)/r);
          if (phi < cone*3.1416/180.) {
            d  = C.d[id];
            n  = d*d_s / (mu*mp);
            T  = C.gE[id]*(gamma-1.0)*p_s/(n*KB);
	    #ifdef SCALAR
            c  = C.c[id]/d;
	    #endif
            vx = C.mx[id]/d;
            vy = C.my[id]/d;
            vz = C.mz[id]/d;
            vr = (vx*x_pos + vy*y_pos + vz*z_pos) / r;
            vr *=v_to_kmps;
            P  = (C.E[id] - 0.5*d*(vx*vx + vy*vy + vz*vz))*(gamma-1.0);
            cs = sqrt(gamma*P/d);
            cs *=v_to_kmps;
            M  = vr / cs;
            P  = n*T;
            d  = d*d_s;
            S  = P * KB * pow(n, -gamma);
	    #ifdef HOT
            if (T > Thot) {
	    #else
            if (T < Tcold) {	      
	    #endif	    
              r_array[bin][cell_count[bin]] = r*n;
              n_array[bin][cell_count[bin]] = n;
              v_array[bin][cell_count[bin]] = vr*n;
              T_array[bin][cell_count[bin]] = T*n;
              P_array[bin][cell_count[bin]] = P*n;
              S_array[bin][cell_count[bin]] = S*n;
	      #ifdef SCALAR
              c_array[bin][cell_count[bin]] = c*n;
	      #endif
              cs_array[bin][cell_count[bin]] = cs*n;
              M_array[bin][cell_count[bin]] = M*n;
              cell_count[bin]++;
            }
          }
        }
      }
    }
  } 

  // free the grid arrays (now just have info in radial bins)
  free(C.d);
  free(C.mx);
  free(C.my);
  free(C.mz);
  free(C.E);
  #ifdef DE
  free(C.gE);
  #endif
  #ifdef SCALAR
  free(C.c);
  #endif

  MPI_Barrier(MPI_COMM_WORLD);


  // do analysis for each bin
  for (int bb=0; bb<N_bins; bb++) {

    int bin_count;
    MPI_Reduce(&cell_count[bb], &bin_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) recvcounts = (int*)malloc(size*sizeof(int));
    MPI_Gather(&cell_count[bb], 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      displs = (int*)malloc(size*sizeof(int));
      displs[0] = 0;
      for (int i=1; i<size; i++) {
        displs[i] = displs[i-1] + recvcounts[i-1];
      }
    }
    double *big_array = NULL;
    if (rank == 0) {
      big_array = (double*)malloc(bin_count*sizeof(double));
      if (big_array == NULL) {
        printf("Error allocating big array.\n");
      }
    }
    // radius 
    MPI_Gatherv(r_array[bb], cell_count[bb], MPI_DOUBLE, big_array, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
    if (rank == 0) {
      qsort(big_array, bin_count, sizeof(double), compare);
      r_av = gsl_stats_mean(big_array, 1, bin_count);
    }
    // density
    MPI_Gatherv(n_array[bb], cell_count[bb], MPI_DOUBLE, big_array, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
    if (rank == 0) {
      qsort(big_array, bin_count, sizeof(double), compare);
      n_av = gsl_stats_mean(big_array, 1, bin_count);
      n_med = gsl_stats_median_from_sorted_data(big_array, 1, bin_count);
      n_lo = gsl_stats_quantile_from_sorted_data(big_array, 1, bin_count, 0.25);
      n_hi = gsl_stats_quantile_from_sorted_data(big_array, 1, bin_count, 0.75);
    }
    // velocity
    MPI_Gatherv(v_array[bb], cell_count[bb], MPI_DOUBLE, big_array, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
    if (rank == 0) {
      qsort(big_array, bin_count, sizeof(double), compare);
      v_av = gsl_stats_mean(big_array, 1, bin_count);
      v_med = gsl_stats_median_from_sorted_data(big_array, 1, bin_count);
      v_lo = gsl_stats_quantile_from_sorted_data(big_array, 1, bin_count, 0.25);
      v_hi = gsl_stats_quantile_from_sorted_data(big_array, 1, bin_count, 0.75);
    }
    // temperature 
    MPI_Gatherv(T_array[bb], cell_count[bb], MPI_DOUBLE, big_array, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
    if (rank == 0) {
      qsort(big_array, bin_count, sizeof(double), compare);
      T_av = gsl_stats_mean(big_array, 1, bin_count);
      T_med = gsl_stats_median_from_sorted_data(big_array, 1, bin_count);
      T_lo = gsl_stats_quantile_from_sorted_data(big_array, 1, bin_count, 0.25);
      T_hi = gsl_stats_quantile_from_sorted_data(big_array, 1, bin_count, 0.75);
    }
    // pressure
    MPI_Gatherv(P_array[bb], cell_count[bb], MPI_DOUBLE, big_array, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
    if (rank == 0) {
      qsort(big_array, bin_count, sizeof(double), compare);
      P_av = gsl_stats_mean(big_array, 1, bin_count);
      P_med = gsl_stats_median_from_sorted_data(big_array, 1, bin_count);
      P_lo = gsl_stats_quantile_from_sorted_data(big_array, 1, bin_count, 0.25);
      P_hi = gsl_stats_quantile_from_sorted_data(big_array, 1, bin_count, 0.75);
    }
    // entropy
    MPI_Gatherv(S_array[bb], cell_count[bb], MPI_DOUBLE, big_array, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
    if (rank == 0) {
      qsort(big_array, bin_count, sizeof(double), compare);
      S_av = gsl_stats_mean(big_array, 1, bin_count);
      S_med = gsl_stats_median_from_sorted_data(big_array, 1, bin_count);
      S_lo = gsl_stats_quantile_from_sorted_data(big_array, 1, bin_count, 0.25);
      S_hi = gsl_stats_quantile_from_sorted_data(big_array, 1, bin_count, 0.75);
    }
    // color
    MPI_Gatherv(c_array[bb], cell_count[bb], MPI_DOUBLE, big_array, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
    if (rank == 0) {
      qsort(big_array, bin_count, sizeof(double), compare);
      c_av = gsl_stats_mean(big_array, 1, bin_count);
      c_med = gsl_stats_median_from_sorted_data(big_array, 1, bin_count);
      c_lo = gsl_stats_quantile_from_sorted_data(big_array, 1, bin_count, 0.25);
      c_hi = gsl_stats_quantile_from_sorted_data(big_array, 1, bin_count, 0.75);
    }
    // sound speed
    MPI_Gatherv(cs_array[bb], cell_count[bb], MPI_DOUBLE, big_array, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
    if (rank == 0) {
      qsort(big_array, bin_count, sizeof(double), compare);
      cs_av = gsl_stats_mean(big_array, 1, bin_count);
      cs_med = gsl_stats_median_from_sorted_data(big_array, 1, bin_count);
      cs_lo = gsl_stats_quantile_from_sorted_data(big_array, 1, bin_count, 0.25);
      cs_hi = gsl_stats_quantile_from_sorted_data(big_array, 1, bin_count, 0.75);
    }
    // Mach number
    MPI_Gatherv(M_array[bb], cell_count[bb], MPI_DOUBLE, big_array, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
    if (rank == 0) {
      qsort(big_array, bin_count, sizeof(double), compare);
      M_av = gsl_stats_mean(big_array, 1, bin_count);
      M_med = gsl_stats_median_from_sorted_data(big_array, 1, bin_count);
      M_lo = gsl_stats_quantile_from_sorted_data(big_array, 1, bin_count, 0.25);
      M_hi = gsl_stats_quantile_from_sorted_data(big_array, 1, bin_count, 0.75);
    }

    if (rank == 0) {
      //printf("%ld %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n", bin_count, r_av, n_av, n_med, n_lo, n_hi, v_av, v_med, v_lo, v_hi, T_av, T_med, T_lo, T_hi, P_av, P_med, P_lo, P_hi, c_av, c_med, c_lo, c_hi, cs_av, cs_med, cs_lo, cs_hi, S_av, S_med, S_lo, S_hi, M_av, M_med, M_lo, M_hi);
      printf("%ld %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n", bin_count, r_av/n_av, n_av, n_med, n_lo, n_hi, v_av/n_av, v_med/n_av, v_lo/n_av, v_hi/n_av, T_av/n_av, T_med/n_av, T_lo/n_av, T_hi/n_av, P_av/n_av, P_med/n_av, P_lo/n_av, P_hi/n_av, c_av/n_av, c_med/n_av, c_lo/n_av, c_hi/n_av, cs_av/n_av, cs_med/n_av, cs_lo/n_av, cs_hi/n_av, S_av/n_av, S_med/n_av, S_lo/n_av, S_hi/n_av, M_av/n_av, M_med/n_av, M_lo/n_av, M_hi/n_av);
    }
    if (rank == 0) {
      free(big_array);
      free(recvcounts);
      free(displs);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }


  for (int bb=0; bb<N_bins; bb++) {
    free(r_array[bb]);
    free(n_array[bb]);
    free(v_array[bb]);
    free(T_array[bb]);
    free(P_array[bb]);
    free(S_array[bb]);
    free(c_array[bb]);
    free(cs_array[bb]);
    free(M_array[bb]);
  }
  //free(r_bins);
  //free(n_av);
  MPI_Finalize();

  return 0;

}


