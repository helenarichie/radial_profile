#include<stdio.h>
#include<stdlib.h>
#include<vector>
#include<string.h>
#include<math.h>
#include<algorithm>
#include<mpi.h>
#include<hdf5.h>
#include<gsl_statistics_double.h>
#include"grid_functions.h"


void Reduction(int cell_count_bb, std::vector<double> big_array, int * recvcounts, int * displs,
	       int bin_count, int rank, bool dweighted, double n_av, std::vector<double> field_array) {

    MPI_Gatherv(field_array.data(), cell_count_bb, MPI_DOUBLE, big_array.data(), recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
      return;
    }

    std::sort(big_array.begin(), big_array.end());
    double av = gsl_stats_mean(big_array.data(), 1, bin_count);
    double med = gsl_stats_median_from_sorted_data(big_array.data(), 1, bin_count);
    double lo = gsl_stats_quantile_from_sorted_data(big_array.data(), 1, bin_count, 0.25);
    double hi = gsl_stats_quantile_from_sorted_data(big_array.data(), 1, bin_count, 0.75);
    if (dweighted) {
      if (n_av != 0) {
        av /= n_av;
        med /= n_av;
        lo /= n_av;
        hi /= n_av;
      } else {
        av = 0;
        med = 0;
	lo = 0;
	hi = 0;
      }
    }
    printf(", %e %e %e %e", av, med, lo, hi);
    return;
}


int main(int argc, char *argv[]) {
  // some constants
  double l_s = 3.086e21; // length scale, centimeters in a kiloparsec
  double m_s = 1.99e33; // mass scale, g in a solar mass
  double t_s = 3.154e10; // time scale, seconds in a kyr
  double d_s = m_s / (l_s * l_s * l_s); // density scale, M_sun / kpc^3
  double v_s = l_s / t_s; // velocity scale, kpc / kyr
  double p_s = d_s * v_s * v_s; // pressure scale, M_sun / kpc kyr^2
  double mp = 1.67e-24; // proton mass in grams
  double KB = 1.3806e-16; // boltzmann constant in cm^2 g / s^2 K
  double v_to_kmps = l_s / t_s / 100000.;
  double kmps_to_kpcpkyr = 1.0220122e-6;
  double m_c = d_s * l_s * l_s * l_s / m_s; // characteristic mass in solar masses
  double mu = 0.6;
  double gamma = 5. / 3.;

  double Tcold = 2e4;
  double Thot = 5e5;
	
  bool dweighted = true;
  bool mask_hot = true;
  double x_len = 10.;  // x-dimension of the global simulation volume
  double y_len = 10.;  // y-dimension
  double z_len = 20.;  // z-dimension
  double cone = 30.;   // cone opening angle

  setbuf(stdout, NULL);

  if (argc < 2){
    printf("Options: np_x np_y np_z dweight vweight hot cold\n");
    printf("Add filename as arg (exit) \n");
    exit(-1);
  }

  for (int i = 2; i < argc; i++) {
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

  int nx, ny, nz;  // the size of the entire simulation grid
  char filename[200];  // the filename substring
  { // don't save any variables except nx, ny, nz, and filename in the global function scope
    int x_off, y_off, z_off, nx_local, ny_local, nz_local;
    strcpy(filename, argv[1]);
    int fnum = 0;  // fnum is the GPU ID used to open the raw data file, e.g. x.h5.fnum
    char fnum_str[20];
    sprintf(fnum_str, "%d", fnum);  // convert fnum to a string
    char filename_i[220];
    sprintf(filename_i, "%s%s", filename, fnum_str);  // append fnum_str to the filename (i.e. x.h5. + fnum)
    Read_Header(filename_i, &nx, &ny, &nz, &x_off, &y_off, &z_off, &nx_local, &ny_local, &nz_local);
  }

  // mpi stuff
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    // The # is so that Numpy automatically ignores it if reading it in
    printf("# Filename: %s Cone: %f Mask: %s Weighting: %s x_len: %e y_len: %e z_len: %e \n\n", filename, cone, hot_or_cold, den_or_vol, x_len, y_len, z_len);
  }

  int nprocs = atoi(argv[2]);  // the number of GPUs the simulation was run on
  int files_per_rank = nprocs / size;  // the number of files each MPI rank is resposible for
  int chunk_volume = nx * ny * nz / size;  // the grid chunk size that each rank is responsible for
  
  int N_bins = 8 * 10;  // the number of radial bins, 8 bins for each kpc
  int N_fields = 8;  // the number of physics variables we want to store in stats_vec
  #ifdef SCALAR
  N_fields += 1;
  #endif
  #ifdef DUST
  N_fields += 4;
  #endif

  // 3D vector of size N_bins x N_fields x 1 initialized with value of 0.0
  std::vector<std::vector<std::vector<double>>> stats_vec(N_bins, std::vector<std::vector<double>>(N_fields, std::vector<double>(1, 0.0)));
  
  // Create arrays to hold cell values as a function of radius
  long int r_bins[N_bins];
  int cell_count[N_bins];
  double m_gas_tot[N_bins];
  #ifdef DUST
  double m_dust_0_tot[N_bins];
  double m_dust_1_tot[N_bins];
  double m_dust_2_tot[N_bins];
  double m_dust_3_tot[N_bins];
  #endif

  // initialize arrays
  for (int bb = 0; bb < N_bins; bb++) {
    r_bins[bb] = 0;
    cell_count[bb] = 0;
    m_gas_tot[bb] = 0.0;
    #ifdef DUST
    m_dust_0_tot[bb] = 0.0;
    m_dust_1_tot[bb] = 0.0;
    m_dust_2_tot[bb] = 0.0;
    m_dust_3_tot[bb] = 0.0;
    #endif
  }

  // each rank reads the files it's responsible for and sums their data into the radial bins
  for (int f_i = 0; f_i < files_per_rank; f_i++) {
    Conserved C;
    int bin;
    int x_off, y_off, z_off, nx_local, ny_local, nz_local;
    double d, vx, vy, vz, n, T, P, S, c, cs, M, d_dust_0, d_dust_1, d_dust_2, d_dust_3;
    double x_pos, y_pos, z_pos, r, vr, phi;
    double dx, dy, dz;

    // Read in some header info
    strcpy(filename, argv[1]);
    int fnum = rank * files_per_rank + f_i;  // fnum is the GPU ID used to open the raw data file, e.g. x.h5.fnum
    char fnum_str[20];
    sprintf(fnum_str, "%d", fnum);
    char filename_i[220];
    sprintf(filename_i, "%s%s", filename, fnum_str);
    printf("Rank %d processing file %s\n", rank, filename_i);
    Read_Header(filename_i, &nx, &ny, &nz, &x_off, &y_off, &z_off, &nx_local, &ny_local, &nz_local);
    
    dx = x_len / nx;  // cell x-dimension in kpc
    dy = y_len / ny;  // cell y-dimension in kpc
    dz = z_len / nz;  // cell z-dimension in kpc
    
    // Allocate memory for the local grid
    C.d  = (double *) malloc(nz_local * ny_local * nx_local * sizeof(double));
    C.mx = (double *) malloc(nz_local * ny_local * nx_local * sizeof(double));
    C.my = (double *) malloc(nz_local * ny_local * nx_local * sizeof(double));
    C.mz = (double *) malloc(nz_local * ny_local * nx_local * sizeof(double));
    C.E  = (double *) malloc(nz_local * ny_local * nx_local * sizeof(double));
    #ifdef DE
    C.gE = (double *) malloc(nz_local * ny_local * nx_local * sizeof(double));
    #endif
    #ifdef SCALAR
    C.c = (double *) malloc(nz_local * ny_local * nx_local * sizeof(double));
    #endif
    #ifdef DUST
    C.d_dust_0 = (double *) malloc(nz_local * ny_local * nx_local * sizeof(double));
    C.d_dust_1 = (double *) malloc(nz_local * ny_local * nx_local * sizeof(double));
    C.d_dust_2 = (double *) malloc(nz_local * ny_local * nx_local * sizeof(double));
    C.d_dust_3 = (double *) malloc(nz_local * ny_local * nx_local * sizeof(double));
    #endif

    // Read in the local grid data
    Read_Grid(filename_i, C, nx_local, ny_local, nz_local);

    // Loop over the local grid and count how many hot/cool cells are in each radial bin
    for (int i = 0; i < nx_local; i++) {
      for (int j = 0; j < ny_local; j++) {
        for (int k = 0; k < nz_local; k++) {
        int id = k + j * nz_local + i * nz_local * ny_local;
        x_pos = (0.5 + i + x_off) * dx - x_len / 2.;
        y_pos = (0.5 + j + y_off) * dy - y_len / 2.;
        z_pos = (0.5 + k + z_off) * dz - z_len / 2.;
        r = sqrt(x_pos * x_pos + y_pos * y_pos + z_pos * z_pos);
        double rbin = 8 * r;
        bin = int(rbin);
        if (bin < N_bins) {
          phi = acos(fabs(z_pos)/r);
          if (phi < cone * 3.1416 / 180.) {
            n  = C.d[id] * d_s / (mu * mp);
            T  = C.gE[id] * (gamma - 1.0) * p_s / (n * KB);
	          if ((mask_hot && (T > Thot)) || (!mask_hot && (T < Tcold))) {
              r_bins[bin]++;  // if cell is within the radius, within the cone, and hot/cool, add a cell
            }
          }
        }
      }
    }
  }

  for (int bb = 0; bb < N_bins; bb++) {
    for (int bf = 0; bf < N_fields; bf++) {
      stats_vec[bb][bf].resize(r_bins[bb]);
      // printf("r_bins: %d\n", r_bins[bb]);
    }
  }

  // Loop over cells and assign values to radial bins
  for (int i = 0; i < nx_local; i++) {
    for (int j = 0; j < ny_local; j++) {
      for (int k = 0; k < nz_local; k++) {
        int id = k + j * nz_local + i * nz_local * ny_local;
        x_pos = (0.5 + i + x_off) * dx - x_len / 2.;
        y_pos = (0.5 + j + y_off) * dy - y_len / 2.;
        z_pos = (0.5 + k + z_off) * dz - z_len / 2.;
        r = sqrt(x_pos * x_pos + y_pos * y_pos + z_pos * z_pos);
	      double rbin = 8 * r;
        bin = int(rbin);
        if (bin < N_bins) {
          phi = acos(fabs(z_pos) / r);
          if (phi < cone * 3.1416 / 180.) {
            d  = C.d[id];
            n  = d * d_s / (mu * mp);
            T  = C.gE[id] * (gamma - 1.0) * p_s / (n * KB);
	          #ifdef SCALAR
            c  = C.c[id] / d;
	          #endif
            #ifdef DUST
	          d_dust_0 = C.d_dust_0[id];
            d_dust_1 = C.d_dust_1[id];
	          d_dust_2 = C.d_dust_2[id];
	          d_dust_3 = C.d_dust_3[id];
            #endif
            vx = C.mx[id] / d;
            vy = C.my[id] / d;
            vz = C.mz[id] / d;
            vr = (vx * x_pos + vy * y_pos + vz * z_pos) / r;
            vr *= v_to_kmps;
            P  = (C.E[id] - 0.5 * d * (vx * vx + vy * vy + vz * vz)) * (gamma - 1.0);
            cs = sqrt(gamma * P / d);
            cs *= v_to_kmps;
            M  = vr / cs;
            P  = n * T;
            d  = d * d_s;
            S  = P * KB * pow(n, -gamma);
	    if ((mask_hot && (T > Thot)) || (!mask_hot && (T < Tcold))) {
        // sum values for total mass arrays
	      m_gas_tot[bin] += d / d_s * dx * dy * dz;
        #ifdef DUST
        m_dust_0_tot[bin] += d_dust_0 * dx * dy * dz;
	      m_dust_1_tot[bin] += d_dust_1 * dx * dy * dz;
	      m_dust_2_tot[bin] += d_dust_2 * dx * dy * dz;
	      m_dust_3_tot[bin] += d_dust_3 * dx * dy * dz;
        #endif
	      int cell_index = cell_count[bin];
	      if (dweighted) {
          // save cell values to sub-grid arrays
          int field_i = 0;
          stats_vec[bin][field_i][cell_index] = r * n;
          field_i++;
          stats_vec[bin][field_i][cell_count[bin]] = n;
          field_i++;
          stats_vec[bin][field_i][cell_count[bin]] = vr * n;
          field_i++;
          stats_vec[bin][field_i][cell_count[bin]] = T * n;
          field_i++;
          stats_vec[bin][field_i][cell_count[bin]] = P * n;
          field_i++;
          stats_vec[bin][field_i][cell_count[bin]] = S * n;
          field_i++;
          stats_vec[bin][field_i][cell_count[bin]] = cs * n;
          field_i++;
          stats_vec[bin][field_i][cell_count[bin]] = M * n;
          field_i++;
          #ifdef SCALAR
          stats_vec[bin][field_i][cell_count[bin]] = c * n;
          field_i++;
          #endif
          #ifdef DUST
          stats_vec[bin][field_i][cell_count[bin]] = d_dust_0;
          field_i++;
          stats_vec[bin][field_i][cell_count[bin]] = d_dust_1;
          field_i++;
          stats_vec[bin][field_i][cell_count[bin]] = d_dust_2;
          field_i++;
          stats_vec[bin][field_i][cell_count[bin]] = d_dust_3;
          field_i++;
          #endif
	      } else {
          // save cell values to sub-grid arrays
          int field_i = 0;
          stats_vec[bin][field_i][cell_count[bin]] = r;
          field_i++;
          stats_vec[bin][field_i][cell_count[bin]] = n;
          field_i++;
          stats_vec[bin][field_i][cell_count[bin]] = vr;
          field_i++;
          stats_vec[bin][field_i][cell_count[bin]] = T;
          field_i++;
          stats_vec[bin][field_i][cell_count[bin]] = P;
          field_i++;
          stats_vec[bin][field_i][cell_count[bin]] = S;
          field_i++;
          stats_vec[bin][field_i][cell_count[bin]] = cs;
          field_i++;
          stats_vec[bin][field_i][cell_count[bin]] = M;
          field_i++;
          #ifdef SCALAR
          stats_vec[bin][field_i][cell_count[bin]] = c;
          field_i++;
          #endif
          #ifdef DUST
          stats_vec[bin][field_i][cell_count[bin]] = d_dust_0;
          field_i++;
          stats_vec[bin][field_i][cell_count[bin]] = d_dust_1;
          field_i++;
          stats_vec[bin][field_i][cell_count[bin]] = d_dust_2;
          field_i++;
          stats_vec[bin][field_i][cell_count[bin]] = d_dust_3;
          field_i++;
          #endif
	      }
          cell_count[bin]++;
	    }
	  }
        }
      }
    }
  }
  // free the local grid arrays (now just have info in radial bins)
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
  #ifdef DUST
  free(C.d_dust_0);
  free(C.d_dust_1);
  free(C.d_dust_2);
  free(C.d_dust_3);
  #endif
  
  MPI_Barrier(MPI_COMM_WORLD);
}

  double m_gas_tot_recv[N_bins];
  MPI_Reduce(&m_gas_tot, &m_gas_tot_recv, N_bins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
 
  #ifdef DUST
  double m_dust_0_tot_recv[N_bins], m_dust_1_tot_recv[N_bins], m_dust_2_tot_recv[N_bins], m_dust_3_tot_recv[N_bins];
  MPI_Reduce(&m_dust_0_tot, &m_dust_0_tot_recv, N_bins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&m_dust_1_tot, &m_dust_1_tot_recv, N_bins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&m_dust_2_tot, &m_dust_2_tot_recv, N_bins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&m_dust_3_tot, &m_dust_3_tot_recv, N_bins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  #endif

  if (rank == 0) { 
    printf("\n# Totals: Radial bin [kpc], m_gas [M_sun]");
    #ifdef DUST
    printf(", m_dust [M_sun]");
    #endif
    printf("\n");
    for (int bb=0; bb<N_bins; bb++) {
      printf("%f, %e", bb / 8., m_gas_tot_recv[bb]);
      #ifdef DUST
      printf(", %e", m_dust_0_tot_recv[bb]);
      printf(", %e", m_dust_1_tot_recv[bb]);
      printf(", %e", m_dust_2_tot_recv[bb]);
      printf(", %e", m_dust_3_tot_recv[bb]);
      #endif
      printf("\n");
    }
  }

  if (rank == 0) {
    printf("\n# Statistics: Radial bin [kpc], Bin cell count, Density [cm^-3], Velocity, Temperature, Pressure, Entropy, Sound speed, Mach number (av med lo hi)\n");
  }
  
  // do analysis for each bin
  for (int bb = 0; bb < N_bins; bb++) {
    double r_av, n_av, n_med, n_lo, n_hi;
    int bin_count;
    MPI_Reduce(&cell_count[bb], &bin_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    std::vector<int> recvcounts(1, 0);
    std::vector<int> displs(1, 0);
    if (rank == 0) {
      recvcounts.resize(size);
      displs.resize(size);
    }
    MPI_Gather(&cell_count[bb], 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      displs[0] = 0;
      for (int i = 1; i < size; i++) {
        displs[i] = displs[i-1] + recvcounts[i-1];
      }
    }
    
    // Allocate a single buffer to hold the cell values from each MPI process
    // The buffer gets re-used for each physical variable 
    std::vector<double> big_array(1, 0.0);
    if (rank == 0) {
      if (bin_count > 0) {
        big_array.resize(bin_count);
      }
    }    
    
    // Use MPI_Gatherv to load all cell values of field_i from each MPI process into big_array and do reduction/statstics
    int field_i = 0;
    
    // radius
    MPI_Gatherv(stats_vec[bb][field_i].data(), cell_count[bb], MPI_DOUBLE, big_array.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      std::sort(big_array.begin(), big_array.end());
      r_av = gsl_stats_mean(big_array.data(), 1, bin_count);
    }
    field_i++;

    // density
    MPI_Gatherv(stats_vec[bb][field_i].data(), cell_count[bb], MPI_DOUBLE, big_array.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      std::sort(big_array.begin(), big_array.end());
      n_av = gsl_stats_mean(big_array.data(), 1, bin_count);
      n_med = gsl_stats_median_from_sorted_data(big_array.data(), 1, bin_count);
      n_lo = gsl_stats_quantile_from_sorted_data(big_array.data(), 1, bin_count, 0.25);
      n_hi = gsl_stats_quantile_from_sorted_data(big_array.data(), 1, bin_count, 0.75);

      if (n_av != 0) { 
        printf("%f, %d, %e %e %e %e %e", bb/8., bin_count, r_av/n_av, n_av, n_med, n_lo, n_hi);
      } else {
        printf("%f, %d, %e %e %e %e %e", bb/8., bin_count, 0.0, n_av, n_med, n_lo, n_hi);
      }
    }
    field_i++;
    
    // velocity
    Reduction(cell_count[bb], big_array, recvcounts.data(), displs.data(), bin_count, rank, dweighted, n_av, stats_vec[bb][field_i]);
    field_i++;
    
    // temperature
    Reduction(cell_count[bb], big_array, recvcounts.data(), displs.data(), bin_count, rank, dweighted, n_av, stats_vec[bb][field_i]);
    field_i++;
    
    // pressure
    Reduction(cell_count[bb], big_array, recvcounts.data(), displs.data(), bin_count, rank, dweighted, n_av, stats_vec[bb][field_i]);
    field_i++;
    
    // entropy
    Reduction(cell_count[bb], big_array, recvcounts.data(), displs.data(), bin_count, rank, dweighted, n_av, stats_vec[bb][field_i]);
    field_i++;
    
    // sound speed
    Reduction(cell_count[bb], big_array, recvcounts.data(), displs.data(), bin_count, rank, dweighted, n_av, stats_vec[bb][field_i]);
    field_i++;
    
    // Mach number
    Reduction(cell_count[bb], big_array, recvcounts.data(), displs.data(), bin_count, rank, dweighted, n_av, stats_vec[bb][field_i]);
    field_i++;
    
    #ifdef SCALAR
    // color
    Reduction(cell_count[bb], big_array, recvcounts.data(), displs.data(), bin_count, rank, dweighted, n_av, stats_vec[bb][field_i]);
    field_i++;
    #endif

    #ifdef DUST
    // dust
    Reduction(cell_count[bb], big_array, recvcounts.data(), displs.data(), bin_count, rank, dweighted, n_av, stats_vec[bb][field_i]);
    field_i++;
    Reduction(cell_count[bb], big_array, recvcounts.data(), displs.data(), bin_count, rank, dweighted, n_av, stats_vec[bb][field_i]);
    field_i++;
    Reduction(cell_count[bb], big_array, recvcounts.data(), displs.data(), bin_count, rank, dweighted, n_av, stats_vec[bb][field_i]);
    field_i++;
    Reduction(cell_count[bb], big_array, recvcounts.data(), displs.data(), bin_count, rank, dweighted, n_av, stats_vec[bb][field_i]);
    field_i++;
    #endif

    if (rank == 0) {
      printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  if (rank == 0) {
    printf("\nProfiles complete.\n");
  }

  MPI_Finalize();

  return 0;

}
