CC = nvcc
CFLAGS = -arch=sm_89
XFLAGS = -Xcompiler=/openmp
# Adjust the architecture flag as needed, e.g. -arch=sm_75 on Google Colaboratory 

all: trapezio.exe managed_memory.exe shared_memory_sum.exe

shared_memory_sum.exe: shared_memory_sum.obj
	$(CC) shared_memory_sum.obj -o shared_memory_sum.exe $(CFLAGS)

shared_memory_sum.obj: shared_memory_sum.cu
	$(CC) -c shared_memory_sum.cu -o shared_memory_sum.obj $(CFLAGS)

managed_memory.exe: managed_memory.obj
	$(CC) managed_memory.obj -o managed_memory.exe $(CFLAGS)

managed_memory.obj: managed_memory.cu
	$(CC) -c managed_memory.cu -o managed_memory.obj $(CFLAGS)

trapezio.exe: trapezio.obj
	$(CC) trapezio.obj -o trapezio.exe $(CFLAGS) $(XFLAGS)

trapezio.obj: trapezio.cu
	$(CC) -c trapezio.cu -o trapezio.obj $(CFLAGS) $(XFLAGS)

clean:
	rm -f *.exe *.obj *.pdb