# Scalar Matrix Multiplication Convolution in SYCL

## Overview

This project explores high-performance convolution operations in Convolutional Neural Networks (CNNs) using a custom **Scalar Matrix Multiplication (SMM)** layer implemented in **SYCL**. 
The implementation is integrated into the AI3 framework to benchmark and compare its performance against PyTorch’s native `Conv2D` layer under varying parallelism parameters.

---

## Objectives

- Implement a custom convolution layer using Scalar Matrix Multiplication in SYCL.
- Integrate the implementation with the AI3 framework.
- Evaluate performance across different thread counts.
- Compare inference times against PyTorch's optimized Conv2D layer.

---

## Implementation Details

### Scalar Matrix Multiplication in SYCL
- Input matrices are split into patches corresponding to convolution kernel receptive fields.
- Each scalar multiplication involves one input and one kernel element.
- Parallelism is controlled via a thread-count parameter to test GPU resource utilization.
- Shared local memory is used to reduce global memory accesses.

### Integration with AI3
- The SMM SYCL kernel is compiled and incorporated into the AI3 framework for compatibility with its data structures.

---

## Benchmarking

- A minimal CNN with a single convolutional layer is used to isolate performance.
- Inference is tested using 8 images.
- Comparisons are made between the SYCL-SMM and PyTorch Conv2D implementations for various thread counts (e.g., 8, 64, 128, ..., 1024).

---

## Key Results

- SYCL-SMM performance improves with increased parallelism.
- At 1024 threads, SYCL-SMM slightly outperformed PyTorch Conv2D.
- PyTorch shows stable performance due to mature kernel optimization.
- Overheads were more noticeable at low thread counts due to launch and memory transfer costs.

---

## Challenges

- Initial compilation issues due to tensor data type mismatches.
- Memory-bound problems and indexing bugs in SYCL kernel.
- Debugging in SYCL was difficult due to limited tooling.

---

## Conclusion

This project demonstrates that a SYCL-based convolution implementation using Scalar Matrix Multiplication can achieve performance comparable to PyTorch’s optimized Conv2D layer at high thread counts. Future work may involve optimizing memory tiling and implementing task-based parallelism for further speedup.

