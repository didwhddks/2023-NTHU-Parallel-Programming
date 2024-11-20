# Solving Matrix Chain Product Problem on GPUs

The implementation described in this repository is based on the methods presented in the paper 'Accelerating the Dynamic Programming for the Matrix Chain Product on the GPU' by Kazufumi Nishida, Yasuaki Ito, and Koji Nakano, published at the IEEE International Conference on Networking and Computing, 2011.

## Overview

This project explores a parallel implementation of the dynamic programming algorithm for the **Matrix Chain Product Problem (MCPP)** on GPUs using CUDA. The algorithm addresses the challenge of minimizing the number of scalar multiplications required to compute the product of a sequence of matrices by optimizing the parenthesization of operations.

Besides, the implementation achieves significant improvements in execution efficiency by exploiting coalesced memory accesses to reduce the number of global memory transactions and utilizing per-block shared memory for faster, localized data access.

## Problem Statement

The **Matrix Chain Product Problem** involves determining the optimal order of multiplications for a chain of matrices to minimize computational cost. The problem, solvable using dynamic programming, has a time complexity of **O(n³)** and a space complexity of **O(n²)**. This referenced research aims to exploit GPU parallelism to significantly accelerate the solution.

## Implementation Details

## Results

## Conclusion

This project demonstrates how GPU parallelism can effectively accelerate dynamic programming algorithms. It introduces a flexible kernel selection mechanism and establishes a benchmark for GPU-based solutions to combinatorial optimization problems.

## References

[1] **Kazufumi Nishida, Yasuaki Ito, and Koji Nakano,** "Accelerating the Dynamic Programming for the Matrix Chain Product on the GPU," *2011 Second International Conference on Networking and Computing (ICNC)*, Higashihiroshima, Japan, 2011, pp. 320-326, doi: [10.1109/ICNC.2011.62](https://doi.org/10.1109/ICNC.2011.62).