"""EXPLORATORY: time brute-force neighbour counting against the kd-tree (synthetic data of the real size)."""
import time

import numpy as np
from scipy.spatial import cKDTree

rng = np.random.default_rng(0)
for dim in (13, 17):
    X = rng.standard_normal((42500, dim))
    Q = rng.standard_normal((1000, dim)) * 1.2
    thr = 3.0
    tree = cKDTree(X)
    t0 = time.perf_counter()
    d = tree.query(Q, k=10)[0][:, -1]
    t1 = time.perf_counter()
    d_ub = tree.query(Q, k=10, distance_upper_bound=thr)[0][:, -1]
    t2 = time.perf_counter()
    x2 = np.einsum("ij,ij->i", X, X)
    XT = np.ascontiguousarray(X.T)
    t3 = time.perf_counter()
    D2 = (np.einsum("ij,ij->i", Q, Q)[:, None] + x2[None, :]) - 2.0 * (Q @ XT)
    cnt = (D2 < thr * thr).sum(1)
    t4 = time.perf_counter()
    kth = np.sqrt(np.maximum(np.partition(D2, 9, axis=1)[:, 9], 0))
    t5 = time.perf_counter()
    print(dim, "kdtree", round(t1 - t0, 3), "kdtree ub", round(t2 - t1, 3), "brute count", round(t4 - t3, 3),
          "partition", round(t5 - t4, 3))
    print("agree", np.mean((d > thr) == (cnt < 10)), np.mean(np.isinf(d_ub) == (d > thr)), np.abs(kth - d).max())
    t0 = time.perf_counter()
    dd = tree.query(Q, k=10, eps=1.0)[0][:, -1]
    print("eps=1 kdtree", round(time.perf_counter() - t0, 3), "max rel err", np.max(dd / d - 1))
