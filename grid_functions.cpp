#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<mpi.h>
#include<hdf5.h>
#include"grid_functions.h"


void Read_Header(char *filename, int *nx, int *ny, int *nz, int *x_off, int *y_off, int *z_off, int *nx_local, int *ny_local, int *nz_local) {

  hid_t file_id, attribute_id;
  herr_t status;
  int dims[3];
  int dims_local[3];
  double offsets[3];
  
  //open the file
  file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    printf("Unable to open input file.\n");
    exit(0);
  }

  // Read in header values
  attribute_id = H5Aopen(file_id, "dims", H5P_DEFAULT); 
  status = H5Aread(attribute_id, H5T_NATIVE_INT, &dims);
  status = H5Aclose(attribute_id);

  *nx = dims[0];
  *ny = dims[1];
  *nz = dims[2];

  attribute_id = H5Aopen(file_id, "dims_local", H5P_DEFAULT); 
  status = H5Aread(attribute_id, H5T_NATIVE_INT, &dims_local);
  status = H5Aclose(attribute_id);

  *nx_local = dims_local[0];
  *ny_local = dims_local[1];
  *nz_local = dims_local[2];

  attribute_id = H5Aopen(file_id, "offset", H5P_DEFAULT); 
  status = H5Aread(attribute_id, H5T_NATIVE_INT, &offsets);
  status = H5Aclose(attribute_id);

  *x_off = offsets[0];
  *y_off = offsets[1];
  *z_off = offsets[2];

  /*
  attribute_id = H5Aopen(file_id, "domain", H5P_DEFAULT); 
  status = H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &domain);

  *xlen = domain[0];
  *ylen = domain[1];
  *zlen = domain[2]; 
  */

  status = H5Aclose(attribute_id); 
}

void Read_Grid(char *filename, Conserved C, int nx_local, int ny_local, int nz_local) {

  hid_t file_id, memspace_id, dataset_id, dataspace_id;
  herr_t status;
  hsize_t dimsm[3];
  // create memory space with size of local grid
  dimsm[0] = nx_local;
  dimsm[1] = ny_local;
  dimsm[2] = nz_local;
  memspace_id = H5Screate_simple(3, dimsm, NULL);


  //open the file
  file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    printf("Unable to open input file.\n");
    exit(0);
  }



  // Open the density dataset
  dataset_id = H5Dopen(file_id, "/density", H5P_DEFAULT);
  // Select the requested subset of data
  dataspace_id = H5Dget_space(dataset_id);
  // status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, stride, count, block);
  // Read in the data subset
  status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, H5P_DEFAULT, C.d);
  // Free the dataset id
  status = H5Sclose(dataspace_id);
  status = H5Dclose(dataset_id);

  dataset_id = H5Dopen(file_id, "/momentum_x", H5P_DEFAULT);
  dataspace_id = H5Dget_space(dataset_id);
  // status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, stride, count, block);
  status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, H5P_DEFAULT, C.mx);
  status = H5Sclose(dataspace_id);
  status = H5Dclose(dataset_id);

  // Open the y momentum dataset
  dataset_id = H5Dopen(file_id, "/momentum_y", H5P_DEFAULT);
  dataspace_id = H5Dget_space(dataset_id);
  // status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, stride, count, block);
  status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, H5P_DEFAULT, C.my);
  status = H5Sclose(dataspace_id);
  status = H5Dclose(dataset_id);

  // Open the z momentum dataset
  dataset_id = H5Dopen(file_id, "/momentum_z", H5P_DEFAULT);
  dataspace_id = H5Dget_space(dataset_id);
  // status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, stride, count, block);
  status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, H5P_DEFAULT, C.mz);
  status = H5Sclose(dataspace_id);
  status = H5Dclose(dataset_id);

  // Open the Energy dataset
  dataset_id = H5Dopen(file_id, "/Energy", H5P_DEFAULT);
  dataspace_id = H5Dget_space(dataset_id);
  // status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, stride, count, block);
  status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, H5P_DEFAULT, C.E);
  status = H5Sclose(dataspace_id);
  status = H5Dclose(dataset_id);

  #ifdef DE
  // Open the internal Energy dataset
  dataset_id = H5Dopen(file_id, "/GasEnergy", H5P_DEFAULT);
  dataspace_id = H5Dget_space(dataset_id);
  // status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, stride, count, block);
  status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, H5P_DEFAULT, C.gE);
  status = H5Sclose(dataspace_id);
  status = H5Dclose(dataset_id);
  #endif

  #ifdef SCALAR
  // Open the Color dataset
  dataset_id = H5Dopen(file_id, "/scalar0", H5P_DEFAULT);
  dataspace_id = H5Dget_space(dataset_id);
  // status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, stride, count, block);
  status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, H5P_DEFAULT, C.c);
  status = H5Sclose(dataspace_id);
  status = H5Dclose(dataset_id);
  #endif


  // free the memory space id
  status = H5Sclose(memspace_id);
  // close the file
  status = H5Fclose(file_id);


}



