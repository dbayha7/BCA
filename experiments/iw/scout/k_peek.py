import numpy as np
import experiments.shift  # noqa
from experiments.shift import recorder as RC, thruster_chain as TC
np.set_printoptions(precision=4, suppress=True, linewidth=160)
K = RC.textbook_gain()
print("K floats\n", K[:, :10])
print("K water (current, wave1, wave2)\n", K[:, 10:])
print("hold pattern at 80%:", -0.8 * K[:, 10] / K[0, 10])
print("v1 full-throttle pattern:", RC.full_throttle_pattern(K))
print("KNEE", repr(TC.KNEE), "float32(-0.8)", repr(float(np.float32(-0.8))))
