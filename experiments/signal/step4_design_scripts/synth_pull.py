# cos(BC pull, true Q^pi ascent) at the S-common start, common states (K0 rollouts) vs each level's own states.
import sys, math, numpy as np
sys.path.insert(0, ".")
from lq_common import A, B, Gr, qq, J, Sigma, Ks, kgrad, cos, D
K0 = 0.5 * Ks + 0.5 * D
H = qq(K0, Gr)
S0 = Sigma(K0)
asc = lambda S: -kgrad(H, K0, S)            # ascent direction of E Q^pi(s, -K s) in K
levels = dict(expert=Ks, mixed=0.5 * Ks, medium=0.1338 * Ks, poor=0 * Ks)
for name, Kbc in levels.items():
    pull = (Kbc - K0)                         # BC pull direction is (K_bc - K) Sigma
    print("%-7s common states: cos(pull, ascent) = %+.3f   J(K_bc) = %.4f" % (name, cos(pull @ S0, asc(S0)), J(Kbc)))
for name, Kb in dict(expert=Ks, medium=0.1338 * Ks, poor=0 * Ks).items():
    S = Sigma(Kb)
    print("%-7s own states:    cos(pull, ascent) = %+.3f   tr Sigma %.3f" % (name, cos((Kb - K0) @ S, asc(S)), np.trace(S)))
