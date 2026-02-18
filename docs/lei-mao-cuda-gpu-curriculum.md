# Lei Mao's CUDA & GPU articles: a structured curriculum

Lei Mao (leimao.github.io) has written 45+ articles on CUDA and GPU programming. They cover everything from first principles to cutting-edge Tensor Core optimization, but they're scattered across his blog without a learning path. This document arranges them into a structured curriculum you can follow from start to finish.

---

## Module 1: CUDA fundamentals

Start here. These articles cover the toolchain, execution model, and basic programming concepts you need before touching any optimization work.

| # | Title | Link |
|---|-------|------|
| 1.1 | CUDA Driver vs CUDA Runtime | [link](https://leimao.github.io/blog/CUDA-Driver-VS-CUDA-Runtime/) |
| 1.2 | CUDA Compilation (PTX, CUBIN, FATBIN) | [link](https://leimao.github.io/blog/CUDA-Compilation/) |
| 1.3 | CUDA Compatibility (forward/backward) | [link](https://leimao.github.io/blog/CUDA-Compatibility/) |
| 1.4 | CUDA Block and Grid | [link](https://leimao.github.io/blog/CUDA-Concept-Block-Grid/) |
| 1.5 | Proper CUDA Error Checking | [link](https://leimao.github.io/blog/Proper-CUDA-Error-Checking/) |
| 1.6 | Pass Function Pointers to Kernels in CUDA | [link](https://leimao.github.io/blog/Pass-Function-Pointers-to-Kernels-CUDA/) |
| 1.7 | Load CUDA Kernel at Runtime Using CUDA Driver APIs | [link](https://leimao.github.io/blog/CUDA-Driver-Runtime-Load-Run-Kernel/) |

---

## Module 2: Memory hierarchy

CUDA performance lives and dies by how you use memory. Read these in order; each builds on the previous one.

| # | Title | Link |
|---|-------|------|
| 2.1 | CUDA Coalesced Memory Access | [link](https://leimao.github.io/blog/CUDA-Coalesced-Memory-Access/) |
| 2.2 | CUDA Data Alignment | [link](https://leimao.github.io/blog/CUDA-Data-Alignment/) |
| 2.3 | CUDA Vectorized Memory Access | [link](https://leimao.github.io/blog/CUDA-Vectorized-Memory-Access/) |
| 2.4 | Page-Locked Host Memory for Data Transfer | [link](https://leimao.github.io/blog/Page-Locked-Host-Memory-Data-Transfer/) |
| 2.5 | CUDA Zero Copy Mapped Memory | [link](https://leimao.github.io/blog/CUDA-Zero-Copy-Mapped-Memory/) |
| 2.6 | CUDA Local Memory | [link](https://leimao.github.io/blog/CUDA-Local-Memory/) |
| 2.7 | CUDA Shared Memory Bank | [link](https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/) |
| 2.8 | CUDA Shared Memory Swizzling | [link](https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/) |
| 2.9 | CUDA Shared Memory Bank Conflict-Free Vectorized Access | [link](https://leimao.github.io/blog/CUDA-Shared-Memory-Bank-Conflict-Free-Vectorized-Access/) |

---

## Module 3: Streams, concurrency, and synchronization

Once you can write correct kernels and manage memory, the next step is overlapping work. These articles cover the concurrency model.

| # | Title | Link |
|---|-------|------|
| 3.1 | CUDA Stream | [link](https://leimao.github.io/blog/CUDA-Stream/) |
| 3.2 | CUDA Default Stream (legacy vs per-thread) | [link](https://leimao.github.io/blog/CUDA-Default-Stream/) |
| 3.3 | CUDA Kernel Execution Overlap | [link](https://leimao.github.io/blog/CUDA-Kernel-Execution-Overlap/) |
| 3.4 | CUDA Rendezvous Stream | [link](https://leimao.github.io/blog/CUDA-Rendezvous-Stream/) |
| 3.5 | PyTorch CUDA Graph Capture | [link](https://leimao.github.io/blog/PyTorch-CUDA-Graph-Capture/) |

---

## Module 4: Parallel primitives and cooperative groups

Reduction and cooperative groups are the building blocks for most parallel algorithms. Read reduction first, then cooperative groups for the cleaner API.

| # | Title | Link |
|---|-------|------|
| 4.1 | CUDA Reduction | [link](https://leimao.github.io/blog/CUDA-Reduction/) |
| 4.2 | CUDA Cooperative Groups | [link](https://leimao.github.io/blog/CUDA-Cooperative-Groups/) |

---

## Module 5: Performance measurement and profiling

Before optimizing, you need to measure correctly.

| # | Title | Link |
|---|-------|------|
| 5.1 | CUDA Performance Hot vs Cold Measurement | [link](https://leimao.github.io/blog/CUDA-Performance-Hot-Cold-Measurement/) |
| 5.2 | CUDA Occupancy Calculation | [link](https://leimao.github.io/blog/CUDA-Occupancy-Calculation/) |
| 5.3 | Nsight Compute in Docker | [link](https://leimao.github.io/blog/Docker-Nsight-Compute/) |
| 5.4 | NVIDIA NVML GPU Statistics (mimicking nvidia-smi) | [link](https://leimao.github.io/blog/NVIDIA-NVML-GPU-Statistics/) |

---

## Module 6: Matrix multiplication; from naive to optimized

This is the canonical CUDA learning project. Start with the basic version, then read the deep-dive optimization article.

| # | Title | Link |
|---|-------|------|
| 6.1 | CUDA Matrix Multiplication (naive + batched) | [link](https://leimao.github.io/blog/CUDA-Matrix-Multiplication/) |
| 6.2 | CUDA Matrix Multiplication Optimization (comprehensive) | [link](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/) |

---

## Module 7: Convolution on GPUs

Convolution is the second most important compute pattern after GEMM. These articles connect convolution to matrix multiplication and cover GPU-friendly tensor layouts.

| # | Title | Link |
|---|-------|------|
| 7.1 | Convolution and Transposed Convolution as Matrix Multiplication | [link](https://leimao.github.io/blog/Convolution-Transposed-Convolution-As-Matrix-Multiplication/) |
| 7.2 | Transposed Convolution as Convolution | [link](https://leimao.github.io/blog/Transposed-Convolution-As-Convolution/) |
| 7.3 | CUDA Tensor Layouts for Convolution (NCHW vs NHWC vs NC/xHWx) | [link](https://leimao.github.io/blog/CUDA-Convolution-Tensor-Layouts/) |

---

## Module 8: Tensor Cores and NVIDIA MMA instructions

Tensor Cores are specialized hardware for matrix math. These articles go from high-level concepts down to individual MMA instructions and benchmarking.

| # | Title | Link |
|---|-------|------|
| 8.1 | NVIDIA Tensor Core Programming (wmma API) | [link](https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/) |
| 8.2 | NVIDIA Tensor Core TN Layout MMA Instruction | [link](https://leimao.github.io/blog/NVIDIA-Tensor-Core-MMA-Instruction-TN-Layout/) |
| 8.3 | Benchmarking NVIDIA Tensor Core MMA Instruction Peak Performances | [link](https://leimao.github.io/blog/Benchmarking-NVIDIA-Tensor-Core-MMA-Peak-Performances/) |

---

## Module 9: CUTLASS and CuTe

CUTLASS is NVIDIA's open-source GEMM library. CuTe is the layout/tensor abstraction it's built on. These articles are dense but rewarding. Follow the order below: math foundations first, then data structures, then operations.

### 9A: CuTe foundations

| # | Title | Link |
|---|-------|------|
| 9.1 | CuTe Layout Algebra (math foundations) | [link](https://leimao.github.io/article/CuTe-Layout-Algebra/) |
| 9.2 | CuTe Arithmetic Tuple Tensor | [link](https://leimao.github.io/blog/CuTe-Arithmetic-Tuple-Tensor/) |
| 9.3 | CuTe Index to Coordinate | [link](https://leimao.github.io/blog/CuTe-Index-To-Coordinate/) |
| 9.4 | CuTe Inverse Layout | [link](https://leimao.github.io/blog/CuTe-Inverse-Layout/) |
| 9.5 | CuTe Blocked and Raked Products | [link](https://leimao.github.io/blog/CuTe-Blocked-Raked-Products/) |
| 9.6 | CuTe Swizzle | [link](https://leimao.github.io/blog/CuTe-Swizzle/) |

### 9B: CuTe operations

| # | Title | Link |
|---|-------|------|
| 9.7 | CuTe Thread-Value Layout | [link](https://leimao.github.io/blog/CuTe-Thread-Value-Layout/) |
| 9.8 | CuTe Tiled Copy | [link](https://leimao.github.io/blog/CuTe-Tiled-Copy/) |
| 9.9 | CuTe ldmatrix (shared memory to register loads) | [link](https://leimao.github.io/blog/CuTe-ldmatrix/) |
| 9.10 | CuTe Tiled MMA | [link](https://leimao.github.io/blog/CuTe-Tiled-MMA/) |
| 9.11 | CuTe Matrix Transpose | [link](https://leimao.github.io/article/CuTe-Matrix-Transpose/) |

### 9C: Building with CUTLASS

| # | Title | Link |
|---|-------|------|
| 9.12 | Build and Develop CUTLASS CUDA Kernels | [link](https://leimao.github.io/blog/Build-Develop-CUTLASS-CUDA-Kernels/) |

---

## Module 10: TensorRT (inference optimization)

These are useful once you move from writing kernels to deploying models. They cover TensorRT's API, quantization, and plugin system.

| # | Title | Link |
|---|-------|------|
| 10.1 | TensorRT Documentation and API References | [link](https://leimao.github.io/blog/TensorRT-Documentations-API-References/) |
| 10.2 | TensorRT Custom Plugin Example | [link](https://leimao.github.io/blog/TensorRT-Custom-Plugin-Example/) |
| 10.3 | TensorRT Implicit Weight Quantization | [link](https://leimao.github.io/blog/TensorRT-Implicit-Weight-Quantization/) |
| 10.4 | PyTorch Eager Mode Quantization TensorRT Acceleration | [link](https://leimao.github.io/blog/PyTorch-Eager-Mode-Quantization-TensorRT-Acceleration/) |
| 10.5 | PyTorch Custom ONNX Operator Export | [link](https://leimao.github.io/blog/PyTorch-Custom-ONNX-Operator-Export/) |

---

## Module 11: System-level topics

Big-picture articles that tie together multiple concepts.

| # | Title | Link |
|---|-------|------|
| 11.1 | System Performance Optimizations | [link](https://leimao.github.io/article/System-Performance-Optimizations/) |
| 11.2 | Transformer Autoregressive Inference Optimization | [link](https://leimao.github.io/article/Transformer-Autoregressive-Inference-Optimization/) |
| 11.3 | How to Debug Deep Learning Inference Applications | [link](https://leimao.github.io/article/How-To-Debug-Deep-Learning-Inference-Applications/) |
| 11.4 | Install NVIDIA RTX 5080 (hardware setup reference) | [link](https://leimao.github.io/blog/Install-NVIDIA-RTX-5080/) |

---

## Suggested reading order

If you're new to CUDA, go module by module: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11.

If you already know the basics (modules 1-3), jump to Module 6 (matrix multiplication) and work through 6 → 8 → 9. That path takes you from naive GEMM to Tensor Core programming to CUTLASS, which is where the most value is.

If you're focused on inference deployment, read modules 1-3 lightly, then skip to Module 10 (TensorRT) and Module 11 (system optimization).

---

*Source: leimao.github.io, compiled February 2026.*
