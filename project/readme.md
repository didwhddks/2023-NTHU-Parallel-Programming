# Solving Matrix Chain Product Problem on GPUs

The implementation described in this repository is based on the concept presented in the paper "**Accelerating the Dynamic Programming for the Matrix Chain Product on the GPU**" by Kazufumi Nishida, Yasuaki Ito, and Koji Nakano, published at the IEEE International Conference on Networking and Computing, 2011.

## Overview

This project explores a parallel implementation of the dynamic programming algorithm for the **Matrix Chain Product Problem (MCPP)** on GPUs using CUDA. The algorithm addresses the challenge of minimizing the number of scalar multiplications required to compute the product of a sequence of matrices by optimizing the parenthesization of operations.

The presented GPU implementation achieves significant improvements in execution efficiency by leveraging the data parallelism of GPUs, optimizing coalesced memory accesses to minimize global memory transactions, and utilizing per-block shared memory for faster and localized data access.

## Matrix Chain Product Problem

**Definition:** The **Matrix Chain Product Problem** is an optimization problem for finding parentheses of the matrix chain that gives the minimum total number of multiplications necessary to compute the product of the matrix chain.

**Solution:** Using the concept of dynamic programming, the optimal solution to the original problem is derived by considering the optimal solutions of all possible sub-problems. The standard dynamic programming solution for the Matrix Chain Multiplication Problem has a time complexity of $\mathbb{O}(n^3)$ and a space complexity of $\mathbb{O}(n^2)$ due to the use of 2D DP tables.

## GPU Implementations

In a standard dynamic programming solution, the 2-dimensional DP table is computed in a bottom-up order, starting with shorter chains of matrices and progressing to longer ones. When offloading this computation to GPUs, a straightforward approach is to assign each entry of the 2D DP table to a separate GPU thread, enabling parallel computation of independent entries.

However, chains of matrices with different lengths generally exhibit data dependencies, preventing their computation within the same kernel. To ensure these dependencies are respected, computations for chains of varying lengths must be offloaded to the GPU sequentially, starting with the shortest chains and proceeding to the longest ones.

### Baseline

In the baseline GPU implementation, only data dependencies are considered. It groups the computation tasks for entries of the same chain length into a single kernel, assigning the computation of each entry to an individual GPU thread.

### Coalesced Memory Accesses

### Shared Memory Usage

### Loop Unrolling

### Warp Divergence

## Results

## Conclusion

This project demonstrates how GPU parallelism can effectively accelerate dynamic programming algorithms. It introduces a flexible kernel selection mechanism and establishes a benchmark for GPU-based solutions to combinatorial optimization problems.

## References

[1] **Kazufumi Nishida, Yasuaki Ito, and Koji Nakano,** "Accelerating the Dynamic Programming for the Matrix Chain Product on the GPU," *2011 Second International Conference on Networking and Computing (ICNC)*, Higashihiroshima, Japan, 2011, pp. 320-326, doi: [10.1109/ICNC.2011.62](https://doi.org/10.1109/ICNC.2011.62).
