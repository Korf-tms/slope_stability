MATLAB_ROOT :=~/.local/matlab2023b
MEX := $(MATLAB_ROOT)/bin/mex

TARGETS := mex_test.mexa64 mex_test_parallel.mexa64

PETSC_CFLAGS := $(shell pkg-config --cflags petsc | sed 's/-D_FORTIFY_SOURCE=[0-9]//g; s/-Wdate-time//g')
PETSC_LIBS := $(shell pkg-config --libs petsc)
MPI_INCLUDE := -I/usr/lib/x86_64-linux-gnu/openmpi/include/

MEX_CFLAGS := CFLAGS='$$CFLAGS -std=c99 -fPIC -w -DPETSC_SKIP_ATTRIBUTE_NODISCARD'
INCLUDES := $(PETSC_CFLAGS) $(MPI_INCLUDE)
LIBS := $(PETSC_LIBS) -lmpi

all: $(TARGETS)

$(TARGETS): %.mexa64: %.c
	$(MEX) -v $< $(MEX_CFLAGS) $(INCLUDES) $(LIBS) -output $*


.PHONY: clean

clean:
	$(RM) $(TARGETS)

run_matlab:
	LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/lib/x86_64-linux-gnu/libmpi.so.40:/usr/lib/x86_64-linux-gnu/libgfortran.so.5 $(MATLAB_ROOT)/bin/matlab
