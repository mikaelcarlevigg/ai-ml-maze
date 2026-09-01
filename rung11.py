# rung11.py - WHEN LEARNING FINALLY WINS.
#
# The plain maze (rung 10) was solvable by a hand-written rule, so learning
# LOST to it (91% vs 82%). Here we HIDE a rule a human can't write: some tiles
# are LETHAL, and their meaning is a secret. A hand-coded navigator (blind to
# the trap) walks onto them and dies; a LEARNED agent DISCOVERS "that kind of
# tile = death" purely from the penalty, and wins. Same two approaches,
# OPPOSITE winner - because now the rule is UN-WRITABLE.
#
# grid: 0 open, 1 wall, 2 TRAP (walkable but lethal; its meaning is hidden)

import os
import pickle
import random
import sys
import time
from collections import deque
import mazelab as env

SIZE = env.SIZE
OBSTACLES = 0.10
TRAPS = 0.08          # fraction of open cells that are (secretly) lethal
TRAIN_GRIDS = 300
TRAIN_STEPS = 120000
MAX_STEPS = 300
GAMMA = 0.95
TRAP_PENALTY = -50.0

AGENT_FILE = os.path.join(os.path.dirname(__file__), "agent_trap.pkl")
SEED_FILE = os.path.join(os.path.dirname(__file__), "last_seed_trap.txt")


def save_agent(Q):
    with open(AGENT_FILE, "wb") as f:
        pickle.dump(Q, f)


def load_agent():
    if os.path.exists(AGENT_FILE):
        with open(AGENT_FILE, "rb") as f:
            return pickle.load(f)
    return None


def save_seed(seed):
    with open(SEED_FILE, "w") as f:
        f.write(str(seed))


def load_seed():
    if os.path.exists(SEED_FILE):
        with open(SEED_FILE) as f:
            return int(f.read().strip())
    return None


def safe_path(grid, goal):
    """Is there a route from START to goal over SAFE (non-trap) open cells?"""
    if grid[1][1] != 0 or grid[goal[0]][goal[1]] != 0:
        return False
    q = deque([(1, 1)])
    seen = {(1, 1)}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in env.ACTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < SIZE and 0 <= nc < SIZE and grid[nr][nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return False


def make_trap(seed):
    """Sparse walls + a random goal + scattered LETHAL trap tiles (value 2),
    always keeping a trap-free route to the goal. Returns (grid, goal)."""
    goal = (SIZE - 2, SIZE - 2)
    for attempt in range(200):
        rng = random.Random(seed * 7919 + attempt)
        grid = [[0] * SIZE for _ in range(SIZE)]
        for r in range(SIZE):
            for c in range(SIZE):
                if r == 0 or c == 0 or r == SIZE - 1 or c == SIZE - 1:
                    grid[r][c] = 1
                elif rng.random() < OBSTACLES:
                    grid[r][c] = 1
        grid[1][1] = 0
        goal = (rng.randrange(1, SIZE - 1), rng.randrange(1, SIZE - 1))
        grid[goal[0]][goal[1]] = 0
        for r in range(1, SIZE - 1):
            for c in range(1, SIZE - 1):
                if grid[r][c] == 0 and (r, c) not in ((1, 1), goal) and rng.random() < TRAPS:
                    grid[r][c] = 2
        if goal != (1, 1) and safe_path(grid, goal):
            return grid, goal
    return grid, goal


def tile(grid, r, c):
    if not (0 <= r < SIZE and 0 <= c < SIZE):
        return 1
    return grid[r][c]


def move(pos, action, grid):
    """Step onto an open OR a trap tile; only walls block. (Traps don't stop
    you - they KILL you, which the caller handles.)"""
    nr, nc = pos[0] + env.ACTIONS[action][0], pos[1] + env.ACTIONS[action][1]
    return (nr, nc) if tile(grid, nr, nc) != 1 else pos


def line_of_sight(pos, grid, goal):
    r, c = pos
    gr, gc = goal
    if r == gr:
        step = 1 if gc > c else -1
        cc = c + step
        while cc != gc:
            if grid[r][cc] == 1:
                return -1
            cc += step
        return 3 if gc > c else 2
    if c == gc:
        step = 1 if gr > r else -1
        rr = r + step
        while rr != gr:
            if grid[rr][c] == 1:
                return -1
            rr += step
        return 1 if gr > r else 0
    return -1


def observe(pos, grid, visited, goal):
    """Senses: walls(4) + 'this tile LOOKS different'(4 - the trap perception,
    whose MEANING is hidden) + visited(4) + line of sight. The learner must
    figure out for itself that the different-looking tile means death."""
    r, c = pos
    walls, traps, seen = [], [], []
    for dr, dc in env.ACTIONS:
        t = tile(grid, r + dr, c + dc)
        walls.append(1 if t == 1 else 0)
        traps.append(1 if t == 2 else 0)
        seen.append(1 if (r + dr, c + dc) in visited else 0)
    return tuple(walls) + tuple(traps) + tuple(seen) + (line_of_sight(pos, grid, goal),)


def q_row(Q, state):
    if state not in Q:
        Q[state] = [0.0, 0.0, 0.0, 0.0]
    return Q[state]


def best_action(Q, state):
    values = q_row(Q, state)
    best = max(values)
    return random.choice([a for a in range(4) if values[a] == best])


def walkable(pos, grid):
    r, c = pos
    return [d for d, (dr, dc) in enumerate(env.ACTIONS) if tile(grid, r + dr, c + dc) != 1]


def handcoded(pos, grid, visited, goal):
    """A human's maze-navigation rule - and TRAP-BLIND: it doesn't know tiles
    can kill, so it happily steps onto them."""
    los = line_of_sight(pos, grid, goal)
    if los != -1:
        return los
    wd = walkable(pos, grid)
    unvis = [d for d in wd if (pos[0] + env.ACTIONS[d][0], pos[1] + env.ACTIONS[d][1]) not in visited]
    if unvis:
        return random.choice(unvis)
    return random.choice(wd) if wd else random.randrange(4)


def train(Q=None):
    grids = [make_trap(s) for s in range(TRAIN_GRIDS)]
    if Q is None:
        Q = {}                          # fresh; pass an existing Q to CONTINUE training it
    rng = random.Random(0)
    for episode in range(TRAIN_STEPS):
        frac = episode / TRAIN_STEPS
        alpha = 0.2 * (1 - frac) + 0.02 * frac
        epsilon = 0.35 * (1 - frac) + 0.03 * frac
        grid, goal = rng.choice(grids)
        pos, visited = env.START, {env.START}
        for _ in range(MAX_STEPS):
            state = observe(pos, grid, visited, goal)
            a = rng.randrange(4) if rng.random() < epsilon else best_action(Q, state)
            new_pos = move(pos, a, grid)
            visited.add(new_pos)
            if grid[new_pos[0]][new_pos[1]] == 2:            # DIED on a trap
                row = q_row(Q, state)
                row[a] += alpha * (TRAP_PENALTY - row[a])    # terminal: just the penalty
                break
            reward = 0.0 if new_pos == goal else -1.0
            old = q_row(Q, state)[a]
            best_future = max(q_row(Q, observe(new_pos, grid, visited, goal)))
            q_row(Q, state)[a] = old + alpha * (reward + GAMMA * best_future - old)
            pos = new_pos
            if pos == goal:
                break
    return Q


def run(pick_action, grid, goal):
    """Run one episode; return ('goal'|'dead'|'timeout', path)."""
    pos, visited, path = env.START, {env.START}, [env.START]
    for _ in range(MAX_STEPS):
        pos = move(pos, pick_action(pos, grid, visited, goal), grid)
        visited.add(pos)
        path.append(pos)
        if grid[pos[0]][pos[1]] == 2:
            return "dead", path
        if pos == goal:
            return "goal", path
    return "timeout", path


def evaluate(pick_action, seeds):
    solved = deaths = 0
    for s in seeds:
        grid, goal = make_trap(s)
        outcome, _ = run(pick_action, grid, goal)
        solved += outcome == "goal"
        deaths += outcome == "dead"
    return solved, deaths


def grid_rows(grid, agent, trail, goal):
    trail = set(trail)
    rows = []
    for r in range(SIZE):
        s = ""
        for c in range(SIZE):
            if (r, c) == agent:
                s += "@ "
            elif (r, c) == goal:
                s += "G "
            elif grid[r][c] == 1:
                s += "██"
            elif grid[r][c] == 2:
                s += "x "
            elif (r, c) in trail:
                s += "* "
            else:
                s += "· "
        rows.append(s)
    return rows


def race2(Q, seed, delay=0.15):
    """Side by side on the SAME trap-maze: trap-blind hand-coder vs learned."""
    grid, goal = make_trap(seed)
    learned = lambda pos, g, v, go: best_action(Q, observe(pos, g, v, go))
    agents = [
        {"name": "HAND-CODED (trap-blind)", "pick": handcoded},
        {"name": "LEARNED (avoids traps)", "pick": learned},
    ]
    for a in agents:
        a["pos"] = env.START
        a["vis"] = {env.START}
        a["trail"] = [env.START]
        a["state"] = "moving..."
        a["alive"] = True
    step = 0
    while step < MAX_STEPS and any(a["alive"] for a in agents):
        step += 1
        for a in agents:
            if not a["alive"]:
                continue
            action = a["pick"](a["pos"], grid, a["vis"], goal)
            a["pos"] = move(a["pos"], action, grid)
            a["vis"].add(a["pos"])
            a["trail"].append(a["pos"])
            if grid[a["pos"][0]][a["pos"][1]] == 2:
                a["state"] = "DIED at step {} (stepped on a trap x)".format(step)
                a["alive"] = False
            elif a["pos"] == goal:
                a["state"] = "REACHED GOAL in {} steps".format(step)
                a["alive"] = False
        print("\033[H\033[J", end="")
        print("  ".join(a["name"].ljust(42) for a in agents))
        rowsets = [grid_rows(grid, a["pos"], a["trail"], goal) for a in agents]
        for i in range(SIZE):
            print("  ".join(rs[i] for rs in rowsets))
        print("\nstep {}".format(step))
        for a in agents:
            print("  {:<24}: {}".format(a["name"], a["state"]))
        time.sleep(delay)
    print("\n=== " + "   ".join("{}: {}".format(a["name"].split()[0], a["state"]) for a in agents) + " ===")


def main():
    do_train = "-t" in sys.argv or "train" in sys.argv
    Q = load_agent()
    if do_train or Q is None:
        print("Fresh training on trap-mazes (~1-2 min)..." if Q is None
              else "Continuing training the saved trap-agent ({} states)...".format(len(Q)))
        Q = train(Q)
        save_agent(Q)
        print("Saved agent ({} states) to {}".format(len(Q), AGENT_FILE))
    else:
        print("Loaded saved trap-agent ({} states). Run  python rung11.py -t  to train it more.".format(len(Q)))

    learned = lambda pos, grid, vis, goal: best_action(Q, observe(pos, grid, vis, goal))

    if do_train:                                   # only re-measure when we actually (re)trained
        hs, hd = evaluate(handcoded, range(9000, 9100))
        ls, ld = evaluate(learned, range(9000, 9100))
        print("\nTrap-maze (hidden lethal tiles - a rule you must DISCOVER):")
        print("  HAND-CODED (trap-blind) : {:>3}/100 solved,  {:>3} deaths".format(hs, hd))
        print("  LEARNED   (from reward) : {:>3}/100 solved,  {:>3} deaths".format(ls, ld))
        print("(On a PLAIN maze the hand-coded WON, 91 vs 82. Hide one rule and it flips.)")

    # pick the grid for the side-by-side
    if "-s" in sys.argv or load_seed() is None:
        start = random.randrange(9000, 999000)
        seed = start
        while seed < start + 500:                  # a grid where the hand-coder dies AND the learned solves
            grid, goal = make_trap(seed)
            if run(handcoded, grid, goal)[0] == "dead" and run(learned, grid, goal)[0] == "goal":
                break
            seed += 1
        save_seed(seed)
    else:
        seed = load_seed()

    print("\nSide by side on grid #{}: hand-coder DIES on a trap (x), learned weaves around.".format(seed))
    print("(-s for a new grid, -t to train more; wide window ~90 cols; Ctrl-C)")
    time.sleep(2.5)
    race2(Q, seed)


if __name__ == "__main__":
    main()
