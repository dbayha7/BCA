"""Optional, information only: does JAX see the graphics card, and how fast is one TD3+BC actor step there compared
with one processor? Runs in its own environment (/root/bcagpu, with jax[cuda12]==0.6.2), never in the Stage 2
environment, and imports nothing of the study: the networks are restated here with the gate's sizes (state 13,
command 4, two hidden layers of 256, batch 256). The pre-registration requires the processor for every phase, so this
number only shows what a later version that allowed the graphics card could gain. Prints one JSON line.
"""

import json
import sys
import time

sys.dont_write_bytecode = True

DX, DA, BATCH, STEPS = 13, 4, 256, 300


def main():
    try:
        import jax
        import jax.numpy as jnp
        import flax.linen as nn
        import optax
    except ImportError as error:
        print(json.dumps(dict(error=f"jax, flax or optax missing in this environment: {error}")))
        return

    class Actor(nn.Module):
        @nn.compact
        def __call__(self, s):
            x = nn.relu(nn.Dense(256)(s))
            x = nn.relu(nn.Dense(256)(x))
            return jnp.tanh(nn.Dense(DA)(x))

    class Critic(nn.Module):
        @nn.compact
        def __call__(self, s, a):
            x = jnp.concatenate([s, a], -1)
            q = []
            for _ in range(2):
                h = nn.relu(nn.Dense(256)(x))
                h = nn.relu(nn.Dense(256)(h))
                q.append(nn.Dense(1)(h)[..., 0])
            return q[0], q[1]

    actor, critic = Actor(), Critic()
    opt = optax.adam(3e-4)

    def setup(device):
        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
            s = jax.random.normal(k1, (BATCH, DX))
            a = jnp.tanh(jax.random.normal(k2, (BATCH, DA)))
            pa = actor.init(k3, s)
            pc = critic.init(k3, s, a)
            st = opt.init(pa)
        # committed to the device, so the compiled step runs there
        return jax.device_put((s, a, pa, pc, st), device)

    def loss(pa, pc, s, a):
        pi = actor.apply(pa, s)
        q, _ = critic.apply(pc, s, pi)
        lam = 2.5 / jax.lax.stop_gradient(jnp.abs(q).mean())
        return -lam * q.mean() + jnp.mean((pi - a) ** 2)

    @jax.jit
    def step(pa, pc, st, s, a):
        g = jax.grad(loss)(pa, pc, s, a)
        up, st = opt.update(g, st, pa)
        return optax.apply_updates(pa, up), st

    def rate(device):
        s, a, pa, pc, st = setup(device)
        pa, st = step(pa, pc, st, s, a)
        jax.block_until_ready(pa)
        t0 = time.perf_counter()
        for _ in range(STEPS):
            pa, st = step(pa, pc, st, s, a)
        jax.block_until_ready(pa)
        return STEPS / (time.perf_counter() - t0)

    devices = [str(d) for d in jax.devices()]
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    cpu_rate = rate(jax.devices("cpu")[0])
    if not gpus:
        print(json.dumps(dict(error=f"JAX sees no graphics card (devices {devices})", devices=devices,
                              cpu_steps_per_s=round(cpu_rate, 1))))
        return
    gpu_rate = rate(gpus[0])
    print(json.dumps(dict(devices=devices, jax=jax.__version__, gpu_steps_per_s=round(gpu_rate, 1),
                          cpu_steps_per_s=round(cpu_rate, 1), ratio=round(gpu_rate / cpu_rate, 2))))


if __name__ == "__main__":
    main()
