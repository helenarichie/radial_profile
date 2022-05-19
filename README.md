# radial_profile

Calculate radial profiles for [Cholla](https://github.com/cholla-hydro/cholla) simulation output.

# Usage

Compile to get the executable 'profiles'

    make

Example commands

    ./profiles /path/to/filename.h5 dweight # run with density weighting
    ./profiles /path/to/filename.h5 vweight # run with volume weighting
    ./profiles /path/to/filename.h5 hot     # run on hot material 
    ./profiles /path/to/filename.h5 cold    # run on cold material

    mpirun -np 2 ./profiles /path/to/filename.h5 np_x 2  # 2 tasks in x direction
    mpirun -np 4 ./profiles /path/to/filename.h5 np_y 4  # 4 tasks in y direction
    mpirun -np 8 ./profiles /path/to/filename.h5 np_z 8  # 8 tasks in z direction	

    # combine arguments 2 x 2 x 4 = 16 total tasks
    mpirun -np 16 ./profiles /path/to/filename.h5 np_x 2 np_y 2 np_z 4 vweight hot

Alwin apologizes for a lazy implementation of command line arguments. 
The code checks for the presence of 'hot', 'cold', 'vweight', and 'dweight' in the command line arguments.
Precedence is determined in the code if you foolishly combine opposites.
Whenever you use the np_x keyword, the next argument should be an int. 