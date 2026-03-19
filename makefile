CC = nvcc

all: main.exe

main.exe: main.cu
	$(CC) -o main.exe main.cu -arch=sm_89 -Xcompiler=/EHsc -L'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\lib\x64' -lcudart


clean:
	rm -f main.exe