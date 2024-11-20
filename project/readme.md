# Solving Matrix Chain Product Problem on GPUs

The implementation described in this repository is based on the methods presented in the paper 'Accelerating the Dynamic Programming for the Matrix Chain Product on the GPU' by Kazufumi Nishida, Yasuaki Ito, and Koji Nakano, published at the IEEE International Conference on Networking and Computing, 2011.

## Overview

This project explores a parallel implementation of the dynamic programming algorithm for the **Matrix Chain Product Problem (MCPP)** on GPUs using CUDA. The algorithm addresses the challenge of minimizing the number of scalar multiplications required to compute the product of a sequence of matrices by optimizing the parenthesization of operations.

Besides, the implementation achieves significant improvements in execution efficiency by exploiting coalesced memory accesses to reduce the number of global memory transactions and utilizing per-block shared memory for faster, localized data access.

## Problem Statement

The **Matrix Chain Product Problem** involves determining the optimal order of multiplications for a chain of matrices to minimize computational cost. The problem, solvable using dynamic programming, has a time complexity of **O(n³)** and a space complexity of **O(n²)**. This research aims to exploit GPU parallelism to significantly accelerate the solution.

## Contributions

1. **Parallel Algorithm Implementation:**  
   Developed three CUDA kernels for GPU execution:
   - `OneThreadPerOneEntry`: Allocates one thread per matrix table entry.
   - `OneBlockPerOneEntry`: Allocates one block per entry for parallel execution by threads.
   - `BlocksPerOneEntry`: Allocates multiple blocks per entry for complex computations.

2. **Dynamic Kernel Selection:**  
   Proposed a strategy to dynamically select the most efficient kernel during execution based on workload characteristics to balance computation across GPU cores.

3. **Performance Optimization:**  
   - Addressed GPU-specific challenges like memory coalescing and thread-block allocation.
   - Explored varying parallelization strategies for maximum efficiency.

4. **Experimental Validation:**  
   Evaluated the implementation on an NVIDIA GeForce GTX 480 GPU, achieving a **40x speedup** compared to a sequential CPU implementation for chains of 16,384 matrices.

## Results

- **Performance:** The GPU implementation shows significant acceleration, reducing computation time from 29,282 seconds on a CPU to 701.6 seconds on a GPU for the largest test case.
- **Optimal Kernel Selection:** Dynamic selection of kernels ensures efficient use of GPU resources across varying computational workloads.

## Implementation Details

The algorithm computes the optimal parenthesization by:
1. Using dynamic programming to build tables (`m` for costs, `s` for split points) incrementally.
2. Parallelizing independent tasks, such as the computation of diagonal entries in the cost table.
3. Adapting thread and block configurations to match workload requirements for each computation step.

## Usage

The implementation is tailored for CUDA-enabled GPUs and requires the NVIDIA CUDA toolkit. It is suitable for researchers and engineers seeking high-performance solutions for matrix chain multiplication problems.

## Conclusion

This project demonstrates how GPU parallelism can effectively accelerate dynamic programming algorithms. It introduces a flexible kernel selection mechanism and establishes a benchmark for GPU-based solutions to combinatorial optimization problems.

---

**Keywords:** Dynamic Programming, Matrix Chain Product, GPGPU, CUDA, Parallel Processing
