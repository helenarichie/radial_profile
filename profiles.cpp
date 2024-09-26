#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<mpi.h>
#include<hdf5.h>
#include<gsl_statistics_double.h>
#include"grid_functions.h"


class OneField
{
public:
  double av;
  double med;
  double lo;
  double hi;
  bool dweight;
  void Print(){
    if (dweight) {
      printf(" %e %e %e %e",av, med, lo, hi);
    } else {
      printf(" %e %e %e %e",av, med, lo, hi);
    }
  }

};



int compare(const void *a, const void *b) {
  if (*(double*)a > *(double*)b) return 1;
  else if (*(double*)a < *(double*)b) return -1;
  else return 0;
}


void Reduction(int cell_count_bb, double * big_array, int * recvcounts, int * displs,
	       int bin_count, int rank, bool dweighted, double n_av, double * field_array) {
    // velocity
    MPI_Gatherv(field_array, cell_count_bb, MPI_DOUBLE, big_array, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
      return;
    }

    qsort(big_array, bin_count, sizeof(double), compare);
    double av = gsl_stats_mean(big_array, 1, bin_count);
    double med = gsl_stats_median_from_sorted_data(big_array, 1, bin_count);
    double lo = gsl_stats_quantile_from_sorted_data(big_array, 1, bin_count, 0.25);
    double hi = gsl_stats_quantile_from_sorted_data(big_array, 1, bin_count, 0.75);
    if (dweighted) {
      av /= n_av;
      med /= n_av;
      lo /= n_av;
      hi /= n_av;
    }
    printf(" %e %e %e %e",av,med,lo,hi);
    return;
}


int main(int argc, char *argv[]) {
  bool dweighted = true;
  bool mask_hot = true;
  double x_len = 10.;
  double y_len = 10.;
  double z_len = 20.;

  setbuf(stdout, NULL);

  if (argc < 2){
    printf("Options: np_x np_y np_z dweight vweight hot cold\n");
    printf("Add filename as arg (exit) \n");
    exit(-1);
  }

  for (int i=2; i<argc; i++) {
    if (strcmp(argv[i],"dweight") == 0) {
      dweighted = true;
    }
    if (strcmp(argv[i],"vweight") == 0) {
      dweighted = false;
    }
    if (strcmp(argv[i],"hot") == 0) {
      mask_hot = true;
    }
    if (strcmp(argv[i],"cold") == 0) {
      mask_hot = false;
    }
  }

  char const * hot_or_cold;
  char const * den_or_vol;
  if (mask_hot) {
    hot_or_cold = "hot";
  } else {
    hot_or_cold = "cold";
  }
  if (dweighted) {
    den_or_vol = "den";
  } else {
    den_or_vol = "vol";
  }

  // The # is so that Numpy automatically ignores it if reading it in
  printf("# Filename: %i Mask: %s Weighting: %s x_len: %e y_len: %e z_len: %e \n",argv[1], hot_or_cold, den_or_vol, x_len, y_len, z_len);

  // mpi stuff
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

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

  // Create arrays to hold cell values as a function of radius
  int N_bins = 80;
  long int r_bins[N_bins];
  int bin;
  for (int i=0; i<N_bins; i++) {
    r_bins[i] = 0;
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

  int nprocs = atoi(argv[2]);  // the number of GPUs the simulation was run on
  int files_per_rank = nprocs / size;  // the number of files each MPI rank is resposible for

  // each rank reads the files it's responsible for and sums their data into the radial bins
  for (int f_i = 0; f_i < files_per_rank; f_i++) {
    Conserved C;
    int nx, ny, nz, nx_local, ny_local, nz_local;
    double d, vx, vy, vz, n, T, P, S, c, cs, M;
    double x_pos, y_pos, z_pos, r, vr, phi;
    double dx, dy, dz;
    double cone = 30.;
    double r_av;
    double n_av, n_med, n_lo, n_hi;
    int x_off, y_off, z_off;

    // Read in some header info
    char filename[200];
    strcpy(filename, argv[1]);
    char fnum = rank * files_per_rank + f_i;  // fnum is the GPU ID used to open the raw data file, e.g. x.h5.fnum
    strncat(filename, &fnum, 4);
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

    // Loop over cells and count cells in each radial bin
    for (int i=0; i<nx_local; i++) {
      for (int j=0; j<ny_local; j++) {
        for (int k=0; k<nz_local; k++) {
        int id = k + j*nz_local + i*nz_local*ny_local;
        x_pos = (0.5+i+x_off)*dx - x_len/2.;
        y_pos = (0.5+j+y_off)*dy - y_len/2.;
        z_pos = (0.5+k+z_off)*dz - z_len/2.;
        r = sqrt(x_pos*x_pos + y_pos*y_pos + z_pos*z_pos);
        double rbin = 8*r;
        bin = int(rbin);
        if (bin < N_bins) {
          phi = acos(fabs(z_pos)/r);
          if (phi < cone*3.1416/180.) {
            n  = C.d[id]*d_s / (mu*mp);
            T  = C.gE[id]*(gamma-1.0)*p_s/(n*KB);

	    if ((mask_hot && (T > Thot)) || (!mask_hot && (T < Tcold))) {
              r_bins[bin]++; // add one to this radial bin count
            }
          }
        }
      }
    }
  }

  // Loop over cells and assign values to radial bins
  for (int i=0; i<nx_local; i++) {
    for (int j=0; j<ny_local; j++) {
      for (int k=0; k<nz_local; k++) {
        int id = k + j*nz_local + i*nz_local*ny_local;
        x_pos = (0.5+i+x_off)*dx - x_len/2.;
        y_pos = (0.5+j+y_off)*dy - y_len/2.;
        z_pos = (0.5+k+z_off)*dz - z_len/2.;
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
	    if ((mask_hot && (T > Thot)) || (!mask_hot && (T < Tcold))) {
	      if (dweighted) {
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
	      } else {
		r_array[bin][cell_count[bin]] = r;
		n_array[bin][cell_count[bin]] = n;
		v_array[bin][cell_count[bin]] = vr;
		T_array[bin][cell_count[bin]] = T;
		P_array[bin][cell_count[bin]] = P;
		S_array[bin][cell_count[bin]] = S;
                #ifdef SCALAR
		c_array[bin][cell_count[bin]] = c;
                #endif
		cs_array[bin][cell_count[bin]] = cs;
		M_array[bin][cell_count[bin]] = M;
	      }
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
  }


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

    if (rank == 0) {
      printf("%ld %e %e %e %e %e",
	     bin_count, r_av/n_av, n_av, n_med, n_lo, n_hi);
    }

    // velocity
    Reduction(cell_count[bb], big_array, recvcounts, displs, bin_count, rank, dweighted, n_av, v_array[bb]);
    // temperature
    Reduction(cell_count[bb], big_array, recvcounts, displs, bin_count, rank, dweighted, n_av, T_array[bb]);
    // pressure
    Reduction(cell_count[bb], big_array, recvcounts, displs, bin_count, rank, dweighted, n_av, P_array[bb]);
    // entropy
    Reduction(cell_count[bb], big_array, recvcounts, displs, bin_count, rank, dweighted, n_av, S_array[bb]);
    // color
    Reduction(cell_count[bb], big_array, recvcounts, displs, bin_count, rank, dweighted, n_av, c_array[bb]);
    // sound speed
    Reduction(cell_count[bb], big_array, recvcounts, displs, bin_count, rank, dweighted, n_av, cs_array[bb]);
    // Mach number
    Reduction(cell_count[bb], big_array, recvcounts, displs, bin_count, rank, dweighted, n_av, M_array[bb]);
    if (rank == 0) {
      printf("\n");
      //printf("%ld %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n", bin_count, r_av, n_av, n_med, n_lo, n_hi, v_av, v_med, v_lo, v_hi, T_av, T_med, T_lo, T_hi, P_av, P_med, P_lo, P_hi, c_av, c_med, c_lo, c_hi, cs_av, cs_med, cs_lo, cs_hi, S_av, S_med, S_lo, S_hi, M_av, M_med, M_lo, M_hi);
      //printf("%ld %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n", bin_count, r_av/n_av, n_av, n_med, n_lo, n_hi, v_av/n_av, v_med/n_av, v_lo/n_av, v_hi/n_av, T_av/n_av, T_med/n_av, T_lo/n_av, T_hi/n_av, P_av/n_av, P_med/n_av, P_lo/n_av, P_hi/n_av, c_av/n_av, c_med/n_av, c_lo/n_av, c_hi/n_av, cs_av/n_av, cs_med/n_av, cs_lo/n_av, cs_hi/n_av, S_av/n_av, S_med/n_av, S_lo/n_av, S_hi/n_av, M_av/n_av, M_med/n_av, M_lo/n_av, M_hi/n_av);
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
