# Links
- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [CUDA Downloads](https://developer.nvidia.com/cuda-downloads) 

# Concepts

![GPU-CPU Architecture](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/gpu-cpu-system-diagram.png)

the GPU can be considered to be a collection of **Streaming Multiprocessors (SMs)** which are organized into groups called **Graphics Processing Clusters (GPCs)**.

Each SM contains a local register file, a unified data cache, and a number of functional units that perform computations. The unified data cache provides the physical resources for shared memory and L1 cache. The allocation of the unified data cache to L1 and shared memory can be configured at runtime. 