# "Fixed-Support Wasserstein Barycenters: Computational Hardness and Fast Algorithm." in NeurIPS'20 

These codes provide implementations of solvers for computing fixed-support Wasserstein barycenters using the fast iterative Bregman projection (FastIBP) algorithm. 

# About
We study the fixed-support Wasserstein barycenter problem (FS-WBP), which consists in computing the Wasserstein barycenter of m discrete probability measures supported on a finite metric space of size n. We show that (i) the FS-WBP in standard linear programming (LP) form is not a minimum-cost flow problem when m and n are larger than 3. We also develop a provably fast \textit{deterministic} variant of the celebrated iterative Bregman projection (IBP) algorithm, named \textsc{FastIBP}, with a complexity bound which is better than the best known complexity bound of the IBP algorithm in terms of desired tolerance, and that of other accelerated algorithms in terms of n.

# Code
Implementations in MATLAB are provided, including the experiments conducted on synthetic data and real MNIST images. 

# References
Fixed-Support Wasserstein Barycenters: Computational Hardness and Fast Algorithm. T. Lin, N. Ho, X. Chen, M. Cuturi and M. I. Jordan. In NeurIPS'2020.  
