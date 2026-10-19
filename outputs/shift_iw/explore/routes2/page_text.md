# The capped pull does not raise return, and what might

*Exploratory. Seed 0 of the cloud rerun unless marked "5 seeds". Nothing in this section is a pre-registered result.*

The capped pull is BCA's main channel. While TD3+BC trains, BCA's band adds extra pull toward the recorded command, more where the band is wide, up to a cap. BCA IW weights the band toward strong current.

## What we see

**It scores below plain TD3+BC on weak data in every situation.** Points, capped pull minus TD3+BC (BCA IW in brackets), 1,000 fixed starts per situation:

| Quality | Recorded | Big kicks | Big waves | Strong current |
|---|---|---|---|---|
| random | -5.7 (-6.5) | -7.9 (-7.4) | -10.9 (-11.4) | -11.7 (-12.6) |
| medium-replay | -3.1 (-2.2) | -5.0 (-4.0) | -5.3 (-2.5) | -5.9 (-6.1) |
| medium | -3.0 (-2.3) | -4.1 (-3.6) | -6.7 (-3.0) | -3.6 (-4.8) |
| medium-expert | -0.7 (-0.8) | -2.1 (-2.3) | -0.3 (-0.6) | +0.2 (-0.5) |
| expert | -0.2 (-0.2) | -0.6 (-0.6) | +0.0 (-0.0) | +0.2 (+0.2) |

On random data, TD3+BC scores 70.3 in recorded conditions, against 64.6 for the capped pull and 63.8 for BCA IW. In strong current: 64.1, 52.4 and 51.4.

## Why

- **It is one even pull.** The extra pull is nearly the same on every row (99% even), about 1.33 times TD3+BC's. That is just TD3+BC with a stronger pull.
- **On weak data it pulls toward zero.** Most recorded commands there are random, and random commands average zero. The actor's commands shrink by 10-21%, and the floats drift. Energy use barely changes.
- **The band switching on is the cause.** Over 5 seeds the learning curves are identical until the band first switches on, then they part. By the end, random, medium-replay and medium trail by 5.3-8.8 points on average, with every seed behind.
- **A better band would not help.** Over 5 seeds, caution placed by the critic's exact error still lost return on weak data. The band also cannot see the error that matters: it is almost the same at stalling full throttle and at the 80% hold.

## What the toy worlds say

In the toy worlds, every expected return is exact. There, **a width inside training did not help**: BCA's width in the pull did worse than simply refusing thin data (regret 0.376 against 0.122). **A gate on each deviation did help**: reading held-out outcomes of a risky action, and letting it through only if they beat the safe action, closed 38-87% of the room. ThrusterChain matches: the capped pull is that width, and the main test's gains so far (restructured BCA and the route search) came only from reading held-out outcomes.

## Potential fixes (untested ideas)

*None of these has run. Each needs the owner's go and pass marks frozen before any run, and each must beat a control without BCA before BCA gets credit.*

1. **A certified throttle ceiling.** TD3+BC trains as usual. When acting, each thruster may go one throttle step higher only if BCA's certificate, on held-out tries of that step, says it beat the step below. *Follows directly from the toys' gate.*
2. **Less pull, guarded.** Halve the pull toward the recorded commands, the opposite of the capped pull, and use the same ceiling as a guard against stalling. *Follows from the toys: drop the width, keep the gate.*
3. **A pull that copies only good rows.** Keep TD3+BC's pull strength, but stop copying recorded commands the critic rates clearly worse than the actor's own, with BCA's band setting "clearly". *A new use of the width that the toys never tested; the least backed of the three.*

A planned check would also show the failure with a control: train TD3+BC with a constant extra pull of the same size and no band. If it matches the capped pull, the band adds nothing.
