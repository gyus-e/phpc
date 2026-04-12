CC = nvcc
CFLAGS = -arch=sm_89
XFLAGS = -Xcompiler=/openmp
# Adjust the architecture flag as needed, e.g. -arch=sm_75 on Google Colaboratory 

all: main.exe

main.exe: integrals.obj main.obj
	$(CC) integrals.obj main.obj -o main.exe $(CFLAGS) $(XFLAGS)

main.obj: main.cu integrals.hpp
	$(CC) -c main.cu -o main.obj $(CFLAGS) $(XFLAGS)

integrals.obj: integrals.cu integrals.hpp
	$(CC) -c integrals.cu -o integrals.obj $(CFLAGS) $(XFLAGS)

clean:
	rm -f *.exe *.obj *.pdb