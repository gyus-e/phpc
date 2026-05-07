CC = nvcc
CFLAGS = -arch=sm_89 
# Adjust the architecture flag as needed, e.g. -arch=sm_75 on Google Colaboratory 

all: main.exe

main.exe: main.obj utils.obj kernels.obj
	$(CC) main.obj utils.obj kernels.obj -o main.exe $(CFLAGS)

main.obj: main.cu utils.hpp kernels.hpp
	$(CC) -c main.cu -o main.obj $(CFLAGS)

kernels.obj: kernels.cu kernels.hpp
	$(CC) -c kernels.cu -o kernels.obj $(CFLAGS)

utils.obj: utils.cpp utils.hpp
	$(CC) -c utils.cpp -o utils.obj $(CFLAGS)

clean:
	rm -f *.exe *.obj *.pdb