
EXEC    = profiles

CC      = mpiCC              # sets the C-compiler

OPTIONS = $(OPTIMIZE) $(OPT)

OBJS   = profiles.o grid_functions.o

INCL   = -I./ -I$(OLCF_GSL_ROOT)/include/gsl/

CFLAGS = $(OPTIONS) -g

LIBS   = -lhdf5 -lgsl -lopenblas 

%.o:	%.cpp
	$(CC) $(CFLAGS) $(INCL) -c $< -o $@

$(EXEC): $(OBJS) 
	$(CC) $(OBJS) $(LIBS) -o $(EXEC) $(INCL)


clean:
	rm -f $(OBJS) $(EXEC)


