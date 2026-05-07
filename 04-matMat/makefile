CC = nvcc
CFLAGS = -arch=sm_89 
# Adjust the architecture flag as needed, e.g. -arch=sm_75 on Google Colaboratory 

all: main.exe

main.exe: main.obj mat_utils.obj mat_kernels.obj
	$(CC) main.obj mat_utils.obj mat_kernels.obj -o main.exe $(CFLAGS)

main.obj: main.cu mat_utils.hpp mat_kernels.hpp
	$(CC) -c main.cu -o main.obj $(CFLAGS)

mat_kernels.obj: mat_kernels.cu mat_kernels.hpp
	$(CC) -c mat_kernels.cu -o mat_kernels.obj $(CFLAGS)

mat_utils.obj: mat_utils.cpp mat_utils.hpp
	$(CC) -c mat_utils.cpp -o mat_utils.obj $(CFLAGS)

clean:
	rm -f *.exe *.obj *.pdb