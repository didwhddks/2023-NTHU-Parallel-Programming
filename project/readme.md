# Solving Matrix Chain Product Problem on GPUs

This project implements a GPU-accelerated dynamic programming algorithm for the Matrix Chain Product Problem (MCP) using CUDA. The implementation is based on the paper: "**Accelerating the Dynamic Programming for the Matrix Chain Product on the GPU**" by Kazufumi Nishida, Yasuaki Ito, and Koji Nakano (IEEE ICNC 2011).

## Overview

The Matrix Chain Product Problem seeks the optimal parenthesization of a matrix sequence to minimize scalar multiplications. The standard dynamic programming solution runs in $\mathbb{O}(n^3)$ time and $\mathbb{O}(n^2)$ space.

This project accelerates the solution by leveraging **GPU parallelism**, optimizing:

- Coalesced memory accesses for efficient global memory transactions
- Shared memory usage to reduce latency
- Loop unrolling and warp-aware execution for performance improvements

## Matrix Chain Product Problem

**Definition:** The **Matrix Chain Product Problem** is an optimization problem for finding parentheses of the matrix chain that gives the minimum total number of multiplications necessary to compute the product of the matrix chain. Suppose that a chain of three or more matrices to be multiplied is given. The total number of multiplications may vary depending on the order of multiplication.

## GPU Implementations

A standard DP table is computed bottom-up. GPU parallelization assigns table entries to threads while respecting data dependencies:

- Baseline: Each DP entry is computed in parallel per chain length
- Optimized versions: Improve memory access patterns and computation efficiency

## Results

I evaluated GPU implementations on an NVIDIA T4 GPU with 2560 CUDA cores (64 cores per SM) and 16GB memory. For comparison, I also tested a traditional CPU implementation on an Apple Silicon M3 chip, which features up to 8 performance cores, 4 efficiency cores, a 16-core Neural Engine, and a unified memory architecture.

The experimental configuration follows the original paper, using **N = 16,384** and the **oneThreadPerEntry** approach.

||  CPU   | GPU_baseline | GPU_coalesced | GPU_optimized |
|:-:| :-:  | :-:  | :-: | :-: |
| Execution time (ms) | 3,118,010  | 194,250 | 70,783.3 | 50,736.4 |

The maximum speedup is roughly 61x.

## Conclusion

This project demonstrates how GPU parallelism can effectively accelerate dynamic programming algorithms. It introduces a flexible kernel selection mechanism and establishes a benchmark for GPU-based solutions to combinatorial optimization problems.

## References

[1] **Kazufumi Nishida, Yasuaki Ito, and Koji Nakano,** "Accelerating the Dynamic Programming for the Matrix Chain Product on the GPU," *2011 Second International Conference on Networking and Computing (ICNC)*, Higashihiroshima, Japan, 2011, pp. 320-326, doi: [10.1109/ICNC.2011.62](https://doi.org/10.1109/ICNC.2011.62).
