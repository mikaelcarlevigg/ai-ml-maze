# llm-rungs: a hand-built ladder through AI, one rung at a time

A personal learning project: build practical AI/ML from scratch, one skill per file, in pure
Python, with no numpy and no frameworks. It started from one question:

> **When does learning actually help, and when does it not?**

Everything below is something we found by running the code, not by reading about it. The neural net
in `rung9.py` is written by hand so nothing stays a black box.

---

## Run it

```bash
python3 rung8.py     # tabular Q-learning masters ONE maze (watch it learn)
python3 rung9.py     # a hand-written neural net vs a table on UNSEEN goals
python3 rung10.py    # generalization across unseen grids   (-t train+save, -s new grid)
python3 race.py      # 3 agents side by side: random | hand-coded | learned   (-t, -s)
python3 rung11.py    # the flip: a HIDDEN lethal rule, learning wins   (-t, -s)
```

No install step. Requires Python 3.8+.

Every image below is a real terminal capture of the actual agents. Nothing generated, nothing
staged.

---

## What we found

1. **A learner can master a single maze, perfectly, but only that one.** Tabular Q-learning found
   the BFS-optimal 44-step path (BFS = the shortest-path oracle we grade against). With state = the
   absolute cell, it memorizes, and transfers to nothing.

2. **Change the state representation and the same learner generalizes.** Swap "which cell am I in"
   for local features (walls around me, direction to the goal) and it solves mazes it never trained
   on, but only on sparse grids. On dense mazes it collapses: the same local view means different
   things in different places (state aliasing), which caps it around 80%.

3. **A memory of visited cells helps, but doesn't break that ceiling.** The cap is about what the
   agent can *sense*, not how hard it tries.

4. **A neural net beats a table when the input is rich.** On continuous goal offsets the table
   scored 1% and the hand-written net 97%. The table can't interpolate between states it has seen.
   The net can.

5. **The honest fairness fix: line-of-sight, not an always-on goal compass.** The agent only "sees"
   the goal straight down a clear row or column. It still learned (~88%), and it stopped feeling
   like cheating.

6. **A dumb 5-line hand-coded rule beat the learned agent.** On plain sparse grids: hand-coded 91%,
   learned ~82%. When the good strategy is writable, learning it from scratch just produces a noisier
   copy of something you could type directly.

   ![Three agents race the same grid: random, hand-coded, learned](images/race.png)
   *`race.py`: on this unseen grid random stays lost, hand-coded reaches the goal in 24 steps, the learned agent in 90.*

7. **We were wrong about the hand-coded agent, and measured it.** We claimed it never re-treads. It
   does, about 27% of its steps (vs random's ~72%). Measured, not assumed.

8. **"Learned always wins at 17 steps" turned out to be a bug in how we looked.** The race used a
   fixed seed, so it was deterministic. With a fresh random grid each run, hand-coded wins most on
   reliability, and learned is only sometimes faster. Reliability vs speed is a real trade-off, not
   a winner.

9. **More training didn't help past a point. It plateaued.** The learned agent wobbled
   (82, 66, 76, 79) around the aliasing ceiling. If the input lacks the information, no amount of
   training fixes it.

10. **The flip: hide a rule, and learning wins.** We made some tiles lethal, walkable but deadly,
    and which ones is a secret you only learn by dying. Trap-blind hand-code: 15/100 solved, 85
    deaths. The learner, from the death penalty alone: 45/100, 1 death. Same two approaches, opposite
    winner. The only thing we changed was whether the rule was hidden.

    ![Hand-coded walks into a lethal trap while the learned agent routes around it](images/trap-flip.png)
    *`rung11.py`: the hand-coded agent dies on a trap at step 13. The learned agent, which discovered the hidden rule, reaches the goal in 17.*

**What it all points to:** the deciding levers are whether *you* can write the rule, and
*representation*, meaning what the agent can sense and remember. Learning earns its place only when
the rule can't be written by hand, and its value grows with how much of the world is hidden.

---

*A personal learning project by Mikael Carlevigg. Pure Python, built to understand, not to ship.*
