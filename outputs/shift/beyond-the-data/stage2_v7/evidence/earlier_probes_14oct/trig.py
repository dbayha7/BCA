import sys, ctypes, platform, numpy as np
ph = np.random.default_rng(3).uniform(0, 2 * np.pi, 1_000_000)
x = np.random.default_rng(4).standard_normal(1_000_000) * 3
np.savez(sys.argv[1], sin=np.sin(ph), cos=np.cos(ph), hypot=np.hypot(x, ph), tanh=np.tanh(x), atan=np.arctan(x))
print(platform.libc_ver(), ctypes.CDLL("libc.so.6").gnu_get_libc_version and ctypes.cast(ctypes.CDLL("libc.so.6").gnu_get_libc_version(), ctypes.c_char_p).value if False else "")
