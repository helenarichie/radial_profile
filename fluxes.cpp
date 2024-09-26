#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<mpi.h>
#include<hdf5.h>
#include<gsl_statistics_double.h>
#include"grid_functions.h"


int compare(const void *a, const void *b) {
  if (*(double*)a > *(double*)b) return 1;
  else if (*(double*)a < *(double*)b) return -1;
  else return 0; 
}


int main(int argc, char *argv[]) {
  
  if (argc < 2) {
    printf("Need filename as arg");
    exit(-1);
  }
  // mpi stuff
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double x_len = 10.;
  double y_len = 10.;
  double z_len = 20.;
  double cone = 85.;

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

  // Create arrays to hold cell values as a function of radius
  int N_bins = 80;
  int bin;
  double bin_width = 0.125;

  double *d_array_hot = (double *)malloc(N_bins*sizeof(double));
  double *d_array_ion = (double *)malloc(N_bins*sizeof(double));
  double *d_array_cold = (double *)malloc(N_bins*sizeof(double));
  double *p_array_hot = (double *)malloc(N_bins*sizeof(double));
  double *p_array_ion = (double *)malloc(N_bins*sizeof(double));
  double *p_array_cold = (double *)malloc(N_bins*sizeof(double));
  double *E_array_hot = (double *)malloc(N_bins*sizeof(double));
  double *E_array_ion = (double *)malloc(N_bins*sizeof(double));
  double *E_array_cold = (double *)malloc(N_bins*sizeof(double));
  double *c_array_hot = (double *)malloc(N_bins*sizeof(double));
  double *c_array_ion = (double *)malloc(N_bins*sizeof(double));
  double *c_array_cold = (double *)malloc(N_bins*sizeof(double));
  double *E_array_th = (double *)malloc(N_bins*sizeof(double));
  int cell_count_hot[N_bins];
  int cell_count_ion[N_bins];
  int cell_count_cold[N_bins];
  // initialize data for each radial bin
  for (int bb=0; bb<N_bins; bb++) {
    d_array_hot[bb] = 0;
    p_array_hot[bb] = 0;
    E_array_hot[bb] = 0;
    c_array_hot[bb] = 0;
    d_array_ion[bb] = 0;
    p_array_ion[bb] = 0;
    E_array_ion[bb] = 0;
    c_array_ion[bb] = 0;
    d_array_cold[bb] = 0;
    p_array_cold[bb] = 0;
    E_array_cold[bb] = 0;
    c_array_cold[bb] = 0;
    E_array_th[bb] = 0;
    cell_count_hot[bb] = 0;
    cell_count_ion[bb] = 0;
    cell_count_cold[bb] = 0;
  }

  int nprocs = atoi(argv[2]);  // the number of GPUs the simulation was run on
  int files_per_rank = nprocs / size;  // the number of files each MPI rank is resposible for

  // each rank reads the files it's responsible for and sums their data into the radial bins
  for (int f_i = 0; f_i < files_per_rank; f_i++) {
    Conserved C;
    double d, vx, vy, vz, n, T, P, E, gE, c; 
    double x_pos, y_pos, z_pos, r, vr, phi;
    double m_flux, p_flux, E_flux, E_flux_kin, E_flux_th, c_flux; 
    double d_sum_hot, p_sum_hot, E_sum_hot, c_sum_hot, d_sum_ion, p_sum_ion, E_sum_ion, c_sum_ion, d_sum_cold, p_sum_cold, E_sum_cold, c_sum_cold, E_sum_th;
    double dx, dy, dz;
    int nx, ny, nz, nx_local, ny_local, nz_local;

    // Read in some header info
    char filename[200];
    strcpy(filename, argv[1]);
    char fnum = rank * files_per_rank + f_i;  // fnum is the GPU ID used to open the raw data file, e.g. x.h5.fnum
    strncat(filename, &fnum, 4);
    printf("%s\n", filename)
    Read_Header(filename, &nx, &ny, &nz, &x_off, &y_off, &z_off, &nx_local, &ny_local, &nz_local);
    dx = x_len / nx_local;
    dy = y_len / ny_local;
    dz = z_len / nz_local;

    // Allocate memory for the grid
    C.d  = (double *) malloc(nz_local*ny_local*nx_local*sizeof(double));
    C.mx = (double *) malloc(nz_local*ny_local*nx_local*sizeof(double));
    C.my = (double *) malloc(nz_local*ny_local*nx_local*sizeof(double));
    C.mz = (double *) malloc(nz_local*ny_local*nx_local*sizeof(double));
    C.E  = (double *) malloc(nz_local*ny_local*nx_local*sizeof(double));
    #ifdef DE
    C.gE = (double *) malloc(nz_local*ny_local*nx_local*sizeof(double));
    #endif
    #ifdef SCALAR
    C.c = (double *) malloc(nz_local*ny_local*nx_local*sizeof(double));
    #endif

    // Read in the grid data
    Read_Grid(filename, C, nx_local, ny_local, nz_local);

    // Loop over cells and assign values to radial bins
    for (int i=0; i<nx_local; i++) {
      for (int j=0; j<ny_local; j++) {
        for (int k=0; k<nz_local; k++) {
          int id = k + j*nz_local + i*nz_local*ny_local;
            x_pos = (0.5+i+x_off)*dx - x_len/2.;
            y_pos = (0.5+j+y_off)*dy - y_len/2.;
            z_pos = (0.5+k+z_off)*dz - z_len/2.;
            r = sqrt(x_pos*x_pos + y_pos*y_pos + z_pos*z_pos);
            phi = acos(fabs(z_pos)/r);
          if (phi < cone*3.1416/180.) {
            double rbin = 8*r;
            bin = int(rbin);
            if (bin < N_bins) {
              d  = C.d[id];
              n  = d*d_s / (mu*mp);
              E  = C.E[id];
              gE = C.gE[id];
              c  = C.c[id] / d;
              P  = gE*(gamma-1.0);
              T  = P*p_s/(n*KB);
              vx = C.mx[id]/d;
              vy = C.my[id]/d;
              vz = C.mz[id]/d;
              vr = (vx*x_pos + vy*y_pos + vz*z_pos) / r;
              m_flux = d*vr*dx*dy*dz;
              c_flux = m_flux*c;
              p_flux = m_flux*vr;
              E_flux_kin = m_flux*0.5*vr*vr;
              E_flux_th = m_flux*(5./2.)*P/d;
              E_flux = E_flux_kin + E_flux_th;
              //if (vr > 0.0) {
              if (T < 2e4) {
                d_array_cold[bin] += m_flux;
                c_array_cold[bin] += c_flux;
                p_array_cold[bin] += p_flux;
                E_array_cold[bin] += E_flux;
                cell_count_cold[bin]++;
              }
              else if (T >= 2e4 && T < 5e5) {
                d_array_ion[bin] += m_flux;
                c_array_ion[bin] += c_flux;
                p_array_ion[bin] += p_flux;
                E_array_ion[bin] += E_flux;
                cell_count_ion[bin]++;
              }
              else {
                d_array_hot[bin] += m_flux;
                c_array_hot[bin] += c_flux;
                p_array_hot[bin] += p_flux;
                E_array_hot[bin] += E_flux;
                E_array_th[bin]  += E_flux_th;
                cell_count_hot[bin]++;
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
  }

  MPI_Barrier(MPI_COMM_WORLD);


  // do analysis for each bin
  for (int bb=0; bb<N_bins; bb++) {

    int bin_count_hot = 0;
    int bin_count_ion = 0;
    int bin_count_cold = 0;
    // count the number of cells in this r-bin
    MPI_Reduce(&cell_count_hot[bb], &bin_count_hot, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD) ;
    MPI_Reduce(&cell_count_ion[bb], &bin_count_ion, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD) ;
    MPI_Reduce(&cell_count_cold[bb], &bin_count_cold, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD) ;
    MPI_Reduce(&d_array_hot[bb], &d_sum_hot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&p_array_hot[bb], &p_sum_hot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&c_array_hot[bb], &c_sum_hot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&E_array_hot[bb], &E_sum_hot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&d_array_ion[bb], &d_sum_ion, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&p_array_ion[bb], &p_sum_ion, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&E_array_ion[bb], &E_sum_ion, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&c_array_ion[bb], &c_sum_ion, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&d_array_cold[bb], &d_sum_cold, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&p_array_cold[bb], &p_sum_cold, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&E_array_cold[bb], &E_sum_cold, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&c_array_cold[bb], &c_sum_cold, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&E_array_th[bb], &E_sum_th, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      printf("%ld %e %e %e %e %ld %e %e %e %e %ld %e %e %e %e %e\n", bin_count_hot, d_sum_hot/bin_width, p_sum_hot/bin_width, E_sum_hot/bin_width, c_sum_hot/bin_width, bin_count_ion, d_sum_ion/bin_width, p_sum_ion/bin_width, E_sum_ion/bin_width, c_sum_ion/bin_width, bin_count_cold, d_sum_cold/bin_width, p_sum_cold/bin_width, E_sum_cold/bin_width, c_sum_cold/bin_width, E_sum_th/bin_width);
      //printf("%ld %e %e %e %e %ld %e %e %e %e %ld %e %e %e %e %e\n", bin_count_hot, d_sum_hot, p_sum_hot, E_sum_hot, c_sum_hot, bin_count_ion, d_sum_ion, p_sum_ion, E_sum_ion, c_sum_ion, bin_count_cold, d_sum_cold, p_sum_cold, E_sum_cold, c_sum_cold, E_sum_th);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  free(d_array_hot);
  free(p_array_hot);
  free(E_array_hot);
  free(c_array_hot);
  free(d_array_ion);
  free(p_array_ion);
  free(E_array_ion);
  free(c_array_ion);
  free(d_array_cold);
  free(p_array_cold);
  free(E_array_cold);
  free(c_array_cold);
  free(E_array_th);
  MPI_Finalize();

  return 0;

}

