# DDK_CUDA_practice_repo
* This repo is for practice of CUDA.
* CUDA is a parallel computing platform and application programming interface (API) model created by Nvidia.

# Figure 1. Thread Batching
![image](https://user-images.githubusercontent.com/67073582/121480862-1ff5a300-c9fe-11eb-833a-a304e2efbcff.png)

# Platforms
* Windows 10
* PyCharm

# API
* PyCUDA

# Template
```python
import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")

multiply_them = mod.get_function("multiply_them")

a = numpy.random.randn(400).astype(numpy.float32)
b = numpy.random.randn(400).astype(numpy.float32)

dest = numpy.zeros_like(a)
multiply_them(
        drv.Out(dest), drv.In(a), drv.In(b),
        block=(400,1,1), grid=(1,1))

print(dest-a*b)
```
# Steps
1. pip install pycuda


# References
* https://docs.nvidia.com/cuda/index.html
* https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)
* https://documen.tician.de/pycuda/
