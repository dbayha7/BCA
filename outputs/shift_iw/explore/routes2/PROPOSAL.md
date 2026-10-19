# The capped pull: failure and fixes

## 1. Why the capped pull does not raise return

1. **It is one even pull.** Dose 1.32-1.34 on every quality, 99% even: TD3+BC with alpha about 1.9, not 2.5. *Measured.*
2. **On weak data it pulls toward zero.** Random commands (69-100% of rows) average zero, so commands shrink 10-21%; floats drift. Seed 0: 3.0-11.7 points below TD3+BC in all four situations (random in strong current: 52.4 vs 64.1); within 2.1 at the top. *Measured.*
3. **The dose causes it.** Curves match until the band switches on, then fall to -8.8, -5.8, -5.3 points, every seed negative. *Measured, 5 seeds.*
4. **No band could rescue it.** Caution placed by the exact critic error lost 37-58 return units on weak data; the band barely tells stalling full throttle from the 80% hold. *Measured, 5 seeds.*
5. **The toys agree:** a width in the pull had regret 0.376, against 0.122 for refusing thin data. *Toy seeds.*

## 2. Potential fixes

Judges' top three that can raise return, flaws repaired. Exploratory screen: weighting test (spec v5), five qualities, seeds 0-4, strong-start points, 5-seed means. Confirming needs fresh seeds 15-19 and its own pre-registration.

### A. Credible-target pull (judges 3/3/3)
- **Does:** TD3+BC's total pull, but no copying of a recorded command the frozen critic rates below the actor's by more than BCA's band width. Actor-only.
- **Why:** Copies good rows only. Seed 0: drops 30-63% of random rows, 0.2-5.2% of the recorder's.
- **Pass (all):** P1 +2.0 over TD3+BC on 2 of medium-replay, medium, medium-expert, 4 of 5 seeds positive; P2 none below -1.0; P3 +1.5 over random release; P4 (band's part) +1.0 over margin 0, the best plain margin (1, 2 or 4 residual spreads) and a matched-share constant. Without P4: "filtering helps; the band adds nothing".
- **Kill:** before training, over 10% of recorder rows or under 15% of random rows released; over 60% released at 4,000 steps; 3 points behind on 3 of 5 seeds at 8,000.
- **Compute:** band fits need approval. Seed 0: 2 h, cloud; seeds 1-4: 23 CPU-h.

### B. Certified throttle ceiling (3/3/3; two designs merged)
- **Does:** TD3+BC unchanged. When acting, each thruster stops at the highest 0.05 step that BCA's certificate (estimated weights) says beat the step below on held-out tries; no tries, no step.
- **Why:** The toys' recommended role. TD3+BC pushes thruster 1 past 80% on 34-46% of top-quality strong-start commands; medium-expert holds 67-86 tries per step there.
- **Pass:** G1 +2.0 over TD3+BC on medium-expert, 4 of 5 seeds; G2 nothing below -1.0, any bank; G3 +1.5 over the range limit and shuffled outcomes, +1.0 over a matched global ceiling. G4 (BCA's part): +2.0 summed against the best plain test; else "outcome reading helped, not BCA".
- **Kill:** no room (knee clip under +1.0); ceiling 0.85+ with 30+ tries past 0.80; ceiling 0.60 or below twice; under 2% clipped.
- **Compute:** no training. Seed 0: under 1 h, cloud; seeds 1-4: 9 CPU-h of shared hosts plus 6.

### C. Guarded release (3/3/2)
- **Does:** Halves the pull (cloning weight 0.5), the capped pull's opposite, with B's ceiling as guard.
- **Why:** Each unit of extra even pull cost 12.5-20.0 points on weak data.
- **Pass:** B1 +3.0 over TD3+BC on 2 of random, medium-replay, medium, 4 of 5 seeds; B2 +1.5 over release alone; B3 +1.5 over range limit and shuffled, +1.0 over a matched ceiling; none below -1.0. B1 alone: "less pull helps; BCA adds nothing".
- **Kill:** guard idle (under 2% of commands past 80%); critic exploitation.
- **Compute:** 8-11 CPU-h on B's hosts; suits the laptop.

## 3. Rejected
- **Survivor-certified pull:** exact weights; any outcome reader passes.
- **Co-trained replacement:** 55-65 CPU-h; TD3+BC passes it.
- **Result F's plain-rule controls:** not a fix; run later.
- **Episode density, support boundary:** a classifier gets the credit.
- **Fallback switch, band-matched comparison:** can only tie step 2.
- **Matched-strength check:** shows the failure only; optional below.

## 4. What I need from you
- **Go or no** for the seed-0 screens of A, B, C, with A's band fits: about 3 h, cloud.
- **Then seeds 1-4** if checks hold: 45 CPU-h, 11-12 h in the cloud or 6 with the laptop.
- **Optional:** that check (a no-band constant dose, co-trained on the desktop, 18 CPU-h).
- **Nothing runs before your go;** marks are frozen (SHA-256) first.
