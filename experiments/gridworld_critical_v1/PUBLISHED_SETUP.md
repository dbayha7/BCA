# The published setup this testbed copies

Prepared November 10 2026 for David. **Status: the published map and rules are transcribed. Nothing is built or run yet.**

Source: Kumar, Hong, Singh and Levine, "When Should We Prefer Offline Reinforcement Learning Over Behavioral Cloning?", ICLR 2022, arXiv 2204.05618. Section 5, Appendix E.1 and Figure 7 (page 33). The paper takes its gridworld from Fu, Kumar, Soh and Levine, "Diagnosing Bottlenecks in Deep Q-learning Algorithms", ICML 2019, whose code is public at `github.com/justinjfu/diagnosing_qlearning` (folder `rlutil/envs/gridcraft`).

Quick navigation: [The maps](#1-the-maps) · [What the paper states](#2-what-the-paper-states) · [What the paper leaves open](#3-what-the-paper-leaves-open) · [What is not in any published setup](#4-what-is-not-in-any-published-setup)

## 1 The maps

Read from Figure 7 by color, cell by cell, with `transcribe_figure.py` on the page rendered at 600 dots per inch. Every one of the 300 cells was classified; none was ambiguous. `S` start, `G` goal, `L` lava, `#` wall, `.` open. Row 0 is the top row.

Single Critical (the one this testbed uses):

```text
      col 0 1 2 3 4 5 6 7 8 9
row 0     S . . . . . . # # #
row 1     . # # # . . . . . .
row 2     . . . . . # . . . #
row 3     # . . # . . . # . .
row 4     # . . . . . # # . .
row 5     # . . . . . . . . #
row 6     # . . # # # L . L #
row 7     . . . L . . . . . .
row 8     # . . . . . . . # #
row 9     . . . L . . . . . G
```

Multiple Critical:

```text
S . . . . . . . # #
# # # # . . # . . #
# . . . . . . . . .
# # L . L # # # # #
# . . . . . . . . #
. . . # # # L . L #
# . . # . . . . . .
. . . L . . # . . .
. # . . . . # L . L
. . . L . . # # . G
```

Cliffwalk:

```text
S . . . L L L L L L
L L L . L L L L L L
L L L . L L L L L L
L L L . L L L L L L
L L L . . . . L L L
L L L L L L . L L L
L L L L L L . L L L
L L . . . . . L L L
L L . L L L L L L L
L L . . . . . . . G
```

Two facts about the Single Critical map that matter for the design:

- **It has two critical cells.** Row 6, column 7 has lava to its left and right. Row 8, column 3 has lava above and below.
- **It has two shortest routes from start to goal, 18 steps each.** One goes through each critical cell. So a learner that refuses one critical cell can still reach the goal by the other route, at no cost from the start and at a large cost from cells near the refused one.

## 2 What the paper states

Quoted from Appendix E.1.

| Item | The paper's words |
|---|---|
| Size and contents | "10 x 10 grids, with a start and goal state, and walls and lava placed in between" |
| Goal reward | "the agent earns a reward of 1 upon reaching the goal state" |
| Lava | "if the agent reaches a lava state, then its reward is 0 for the rest of the trajectory" |
| Moves | "The agent is able to move in either of the four direction (or choose to stay still)" |
| Noise | "there is a 10% chance that the agent travels in a different direction than commanded" |
| Critical states | "holes in walls through which the agent must pass; if the agent chooses a wrong action at those states, it veers off into a lava state" |

Its datasets are trajectories from the optimal policy, optionally from a different start distribution, or mixed with trajectories from a random policy. No move is deliberately left out of any dataset.

## 3 What the paper leaves open

The paper does not state these. Where the cited code settles one, I follow the code.

| Open item | What the cited code does | What this testbed will do |
|---|---|---|
| Exact noise rule | With the noise probability, the move is drawn evenly from all legal moves, which include the commanded move and staying still | Follow the code, with probability 0.10. The chance of any one other legal move is then 2% at a cell with five legal moves |
| Moving into a wall or off the grid | The agent stays where it is | Follow the code |
| Whether the goal ends the episode | It does not; the code pays while the agent stands on the tile | End the episode with a reward of 1, which is what "upon reaching" describes |
| Discount | Not in the code | 0.95, as in our earlier testbed. This is our choice |
| Episode length | Not set | 200 steps for evaluation, as before. This is our choice |

## 4 What is not in any published setup

Both would be our additions, and both are needed for the test David asked for.

1. **A withheld move.** In every published setup I read, moves are rare by chance. None leaves one move out on purpose.
2. **Continuous actions.** Every published gridworld uses discrete moves. BCA needs continuous actions, so the policy would output five preferences that are turned into one of the five moves.
