# race.py - THREE agents on the SAME grid, side by side, stepping together:
#   RANDOM (no memory)  |  HAND-CODED memory rule (no learning)  |  LEARNED
# Watch both memory-users crush the random one - and the hand-coded rule even
# edge out the learned one (see the chat: 91 vs 82 vs 26 out of 100).

import os
import random
import sys
import time
import mazelab as env
import rung10

SEED_FILE = os.path.join(os.path.dirname(__file__), "last_seed.txt")


def save_seed(seed):
    with open(SEED_FILE, "w") as f:
        f.write(str(seed))


def load_seed():
    if os.path.exists(SEED_FILE):
        with open(SEED_FILE) as f:
            return int(f.read().strip())
    return None


def grid_rows(grid, agent, trail, goal):
    """Display rows for one grid (2 chars per cell, so it looks square)."""
    trail = set(trail)
    rows = []
    for r in range(len(grid)):
        s = ""
        for c in range(len(grid[0])):
            if (r, c) == agent:
                s += "@ "
            elif (r, c) == goal:
                s += "G "
            elif grid[r][c] == 1:
                s += "██"
            elif (r, c) in trail:
                s += "* "
            else:
                s += "· "
        rows.append(s)
    return rows


def open_dirs(pos, grid):
    r, c = pos
    out = []
    for d, (dr, dc) in enumerate(env.ACTIONS):
        nr, nc = r + dr, c + dc
        if 0 <= nr < env.SIZE and 0 <= nc < env.SIZE and grid[nr][nc] == 0:
            out.append(d)
    return out


# --- the three policies (all take the same args; some ignore some) ---
def move_random(pos, grid, vis, goal, Q):
    return random.randrange(4)


def move_handcoded(pos, grid, vis, goal, Q):
    los = rung10.line_of_sight(pos, grid, goal)
    if los != -1:
        return los                                  # sees goal -> beeline
    ods = open_dirs(pos, grid)
    unvis = [d for d in ods if (pos[0] + env.ACTIONS[d][0], pos[1] + env.ACTIONS[d][1]) not in vis]
    if unvis:
        return random.choice(unvis)                 # prefer new ground
    if ods:
        return random.choice(ods)                   # dead end -> backtrack
    return random.randrange(4)


def move_learned(pos, grid, vis, goal, Q):
    return rung10.best_action(Q, rung10.observe(pos, grid, vis, goal))


class Runner:
    def __init__(self, name, move):
        self.name = name
        self.move = move
        self.pos = env.START
        self.vis = {env.START}
        self.trail = [env.START]
        self.done = False
        self.steps = 0

    def step(self, grid, goal, Q):
        if self.done:
            return
        a = self.move(self.pos, grid, self.vis, goal, Q)
        self.pos = env.move(self.pos, env.ACTIONS[a], grid)
        self.vis.add(self.pos)
        self.trail.append(self.pos)
        self.steps += 1
        if self.pos == goal:
            self.done = True


def race(Q, seed, delay=0.12):
    grid, goal = env.make_sparse(seed, rung10.OBSTACLES)
    runners = [
        Runner("RANDOM (no memory)", move_random),
        Runner("HAND-CODED (memory rule)", move_handcoded),
        Runner("LEARNED (trained)", move_learned),
    ]
    step = 0
    # run until BOTH memory-users finish (random usually never does)
    while step < 200 and not (runners[1].done and runners[2].done):
        step += 1
        for r in runners:
            r.step(grid, goal, Q)
        print("\033[H\033[J", end="")
        print("  ".join(r.name.ljust(42) for r in runners))
        rowsets = [grid_rows(grid, r.pos, r.trail, goal) for r in runners]
        for i in range(len(grid)):
            print("  ".join(rs[i] for rs in rowsets))
        status = "   ".join(
            "{}: {}".format(r.name.split()[0], "GOAL@{}".format(r.steps) if r.done else "step {}".format(r.steps))
            for r in runners)
        print("\n" + status)
        time.sleep(delay)

    print("\n=== result on this unseen grid ===")
    for r in runners:
        outcome = "reached goal in {} steps".format(r.steps) if r.done else "did NOT reach goal (still lost)"
        print("  {:<26}: {}".format(r.name, outcome))


def main():
    do_train = "-t" in sys.argv or "train" in sys.argv
    Q = rung10.load_agent()
    if do_train:
        print("Fresh training..." if Q is None
              else "Continuing training the saved agent ({} states so far)...".format(len(Q)))
        Q, _ = rung10.train(Q)                       # None -> fresh; existing Q -> warm start
        rung10.save_agent(Q)
        print("Saved agent ({} states) to {}".format(len(Q), rung10.AGENT_FILE))
    elif Q is None:
        print("No saved agent yet - training one (run with  -t  later to train it more)...")
        Q, _ = rung10.train()
        rung10.save_agent(Q)
        print("Saved agent ({} states).".format(len(Q)))
    else:
        print("Loaded saved agent ({} states). Run  python race.py -t  to train it more.".format(len(Q)))
    if "-s" in sys.argv:
        seed = random.randrange(9000, 999999)      # -s : brand new grid
        save_seed(seed)
        note = "NEW grid - run without -s to keep it"
    else:
        seed = load_seed()                         # otherwise: keep the last grid
        if seed is None:
            seed = random.randrange(9000, 999999)
            save_seed(seed)
            note = "first grid - use -s for a new one"
        else:
            note = "same grid as last run - use -s for a new one"
    print("Racing on grid #{} ({}). Any grid now - the LEARNED may fail.".format(seed, note))
    print("(wide window ~135 cols; Ctrl-C to stop)")
    time.sleep(2)
    race(Q, seed)


if __name__ == "__main__":
    main()
