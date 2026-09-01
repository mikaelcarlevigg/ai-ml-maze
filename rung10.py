# rung10.py - GENERALISATION IN ACTION (random obstacles AND a random goal).
#
# A learned policy that solves obstacle-grids it has NEVER seen - now with the
# goal placed anywhere, not just the corner. The KEY (as in rung 9) is the
# STATE: the agent only sees LOCAL, grid-independent features - walls around it,
# where it has been, and roughly which way the goal is - so what it learns on
# one grid transfers to ALL of them, wherever the goal sits.
#
# (This uses a TABLE, not a net: generalisation comes from the FEATURES, not the
# learner. A net is only needed when the features get too rich for a table.)

import os
import pickle
import random
import time
import mazelab as env

OBSTACLES = 0.13      # obstacle density (harder; 0.15+ starts to trap the agent)
TRAIN_GRIDS = 300     # how many different grids to practise on
TRAIN_STEPS = 120000  # sight-only is harder to learn -> more practice needed
MAX_STEPS = 400       # denser grids need more room to explore before giving up
GAMMA = 0.95          # alpha + epsilon DECAY over training, inside train()


def line_of_sight(pos, grid, goal):
    """Can the agent SEE the goal? Only along a clear straight row or column (no
    walls between). Returns the direction it sees it in (0-3), or -1 if out of
    sight. No compass: when it can't see the goal, it must explore to find it."""
    r, c = pos
    gr, gc = goal
    if r == gr:                                  # same row - look sideways
        step = 1 if gc > c else -1
        cc = c + step
        while cc != gc:
            if grid[r][cc] == 1:
                return -1                        # a wall blocks the view
            cc += step
        return 3 if gc > c else 2                # right / left
    if c == gc:                                  # same column - look up/down
        step = 1 if gr > r else -1
        rr = r + step
        while rr != gr:
            if grid[rr][c] == 1:
                return -1
            rr += step
        return 1 if gr > r else 0                # down / up
    return -1                                    # not aligned -> can't see it


def observe(pos, grid, visited, goal):
    """What the agent SENSES: the 4 walls next to it, which neighbours it has
    already visited, and - ONLY when there's a clear line of sight - which way
    the goal is. No sight = no goal info, so it has to explore to find it."""
    r, c = pos
    walls, seen = [], []
    for dr, dc in env.ACTIONS:
        nr, nc = r + dr, c + dc
        open_here = 0 <= nr < env.SIZE and 0 <= nc < env.SIZE and grid[nr][nc] == 0
        walls.append(0 if open_here else 1)
        seen.append(1 if (nr, nc) in visited else 0)
    return tuple(walls) + tuple(seen) + (line_of_sight(pos, grid, goal),)


def q_row(Q, state):
    if state not in Q:
        Q[state] = [0.0, 0.0, 0.0, 0.0]
    return Q[state]


def best_action(Q, state):
    values = q_row(Q, state)
    best = max(values)
    tied = [a for a in range(4) if values[a] == best]
    return random.choice(tied)


def solve_path(Q, grid, goal):
    """Follow the learned policy on one grid; return the path it walks."""
    pos, visited, path = env.START, {env.START}, [env.START]
    for _ in range(MAX_STEPS):
        pos = env.move(pos, env.ACTIONS[best_action(Q, observe(pos, grid, visited, goal))], grid)
        visited.add(pos)
        path.append(pos)
        if pos == goal:
            break
    return path


def solve_rate(Q, seeds):
    """Fraction of these (unseen) grids the greedy policy actually solves."""
    solved = 0
    for s in seeds:
        grid, goal = env.make_sparse(s, OBSTACLES)
        if solve_path(Q, grid, goal)[-1] == goal:
            solved += 1
    return solved / len(seeds)


AGENT_FILE = os.path.join(os.path.dirname(__file__), "agent.pkl")


def save_agent(Q):
    with open(AGENT_FILE, "wb") as f:
        pickle.dump(Q, f)


def load_agent():
    if os.path.exists(AGENT_FILE):
        with open(AGENT_FILE, "rb") as f:
            return pickle.load(f)
    return None


def train(Q=None):
    grids = [env.make_sparse(s, OBSTACLES) for s in range(TRAIN_GRIDS)]   # list of (grid, goal)
    if Q is None:
        Q = {}                          # fresh; pass an existing Q to CONTINUE training it
    rng = random.Random(0)
    checkpoints = {0, 2000, 8000, 20000, 40000, TRAIN_STEPS - 1}
    unseen = range(9000, 9050)     # grids NEVER trained on, for the score
    history = []
    for episode in range(TRAIN_STEPS):
        # decay exploration + learning-rate over time, so the policy SETTLES
        frac = episode / TRAIN_STEPS
        alpha = 0.2 * (1 - frac) + 0.02 * frac
        epsilon = 0.35 * (1 - frac) + 0.03 * frac
        grid, goal = rng.choice(grids)
        pos, visited = env.START, {env.START}
        for _ in range(MAX_STEPS):
            state = observe(pos, grid, visited, goal)
            if rng.random() < epsilon:
                a = rng.randrange(4)
            else:
                a = best_action(Q, state)
            new_pos = env.move(pos, env.ACTIONS[a], grid)
            visited.add(new_pos)
            reward = 0.0 if new_pos == goal else -1.0
            # THE LEARNING (the same Q-update as rung 8) - keyed by FEATURES:
            old = q_row(Q, state)[a]
            best_future = max(q_row(Q, observe(new_pos, grid, visited, goal)))
            q_row(Q, state)[a] = old + alpha * (reward + GAMMA * best_future - old)
            pos = new_pos
            if pos == goal:
                break
        if episode in checkpoints:
            history.append((episode, solve_rate(Q, unseen)))
    return Q, history


def main():
    print("Practising on {} random grids (scattered walls + a random goal)...".format(TRAIN_GRIDS))
    baseline = solve_rate({}, range(9000, 9050))     # empty brain = random walk
    Q, history = train()

    print("\nGrids it has NEVER seen, solved:")
    print("  BEFORE training (random) : {:>4.0%}".format(baseline))
    print("  AFTER training           : {:>4.0%}".format(history[-1][1]))
    print("\nover training (climbs, wobbles - local features alias many states,")
    print("so a table can't settle perfectly; it lands high anyway):")
    for episode, rate in history:
        print("  after {:>6} attempts : {:>4.0%}  {}".format(episode, rate, "#" * int(rate * 40)))

    print("\nNow watch it solve grids it has NEVER seen (Ctrl-C to skip):")
    time.sleep(2)
    shown, seed = 0, 9100
    while shown < 3 and seed < 9400:
        grid, goal = env.make_sparse(seed, OBSTACLES)
        path = solve_path(Q, grid, goal)
        if path[-1] == goal:
            env.animate(grid, path, label="unseen grid #{}".format(seed), goal=goal)
            time.sleep(1)
            shown += 1
        seed += 1


if __name__ == "__main__":
    main()
