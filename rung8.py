# rung8.py - THE MACHINE LEARNING (reinforcement learning: Q-learning).
#
# All the maze + drawing lives in mazelab.py, imported as `env`. Anything
# written `env.something` is plumbing you can ignore for now. EVERYTHING ELSE
# in this file is the learning. That is the whole point of the split.
#
# The idea in one breath: a gubbe learns the shortest way through a maze by
# trial and error. He keeps a NOTEBOOK (Q). Every step he nudges one number
# in it. After thousands of steps the notebook tells him the way.

import random
import time
import mazelab as env

# --- learning hyperparameters (the knobs) ---
EPISODES = 3000    # how many attempts at the maze
MAX_STEPS = 3000   # give up an attempt after this many steps
ALPHA = 0.1        # learning rate  (how far each nudge moves a number)
GAMMA = 0.95       # discount       (how much the future counts vs now)
EPSILON = 0.25     # exploration    (chance of a random move instead of the best)


def best_action(Q, pos):
    """THE POLICY: the best-known direction from this cell.
    Ties (e.g. when all four numbers are still 0) break RANDOMLY, so an
    untrained gubbe explores instead of always going the same way."""
    values = Q[pos[0]][pos[1]]
    best_value = values[0]
    for v in values:
        if v > best_value:
            best_value = v
    tied = []
    for a in range(len(values)):
        if values[a] == best_value:
            tied.append(a)
    return random.choice(tied)


def train(grid):
    """The whole of the learning. Fills in and returns the notebook Q."""
    rows, cols = len(grid), len(grid[0])
    goal = env.goal_of(rows)

    # THE NOTEBOOK: Q[row][col][direction] = how good that move looks. All 0.0.
    Q = [[[0.0 for _ in env.ACTIONS] for _ in range(cols)] for _ in range(rows)]

    rng = random.Random(env.SEED)
    steps_per_episode = []
    snapshots = []
    checkpoints = {0, 60, 120, 200, 280, 360, 440, 600, EPISODES - 1}

    for episode in range(EPISODES):              # one attempt at the maze
        pos = env.START
        steps = MAX_STEPS
        for t in range(MAX_STEPS):               # one step
            # pick a direction: usually the best-known, sometimes random
            if rng.random() < EPSILON:
                a = rng.randrange(len(env.ACTIONS))
            else:
                a = best_action(Q, pos)

            new_pos = env.move(pos, env.ACTIONS[a], grid)
            reward = 0.0 if new_pos == goal else -1.0   # -1 per step = hurry up

            # ==== THE LEARNING (your STEP 4): the Q-update ====
            # Nudge this ONE number toward "the reward now + the best you can
            # do from where you landed". Repeat this millions of times and the
            # goal's value seeps backward through the whole maze.
            old = Q[pos[0]][pos[1]][a]
            best_future = max(Q[new_pos[0]][new_pos[1]])
            Q[pos[0]][pos[1]][a] = old + ALPHA * (reward + GAMMA * best_future - old)
            # ==================================================

            pos = new_pos
            if pos == goal:
                steps = t + 1
                break
        steps_per_episode.append(steps)
        if episode in checkpoints:
            snapshots.append((episode, field_of(Q, grid)))

    return Q, steps_per_episode, snapshots


def rollout(grid, Q, epsilon, max_len=150):
    """Run one attempt following the policy and record the path (to draw it)."""
    rng = random.Random()
    goal = env.goal_of(len(grid))
    pos = env.START
    traj = [pos]
    for _ in range(max_len):
        if rng.random() < epsilon:
            a = rng.randrange(len(env.ACTIONS))
        else:
            a = best_action(Q, pos)
        pos = env.move(pos, env.ACTIONS[a], grid)
        traj.append(pos)
        if pos == goal:
            break
    return traj


def field_of(Q, grid):
    """For drawing only: from which cells does the greedy policy already
    reach the goal? (The 'solved region' that grows back from G to S.)"""
    rows, cols = len(grid), len(grid[0])
    goal = env.goal_of(rows)
    field = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                pos = (r, c)
                seen = set()
                for _ in range(rows * cols):
                    if pos == goal:
                        field[r][c] = True
                        break
                    if pos in seen:
                        break
                    seen.add(pos)
                    pos = env.move(pos, env.ACTIONS[best_action(Q, pos)], grid)
    return field


def main():
    grid = env.make_maze(env.SIZE, env.SEED)
    opt = env.optimal_steps(grid)

    print("The maze (@ = start, G = goal):")
    env.render(grid, agent=env.START)
    print("\nBFS optimal (the perfect verifier says): {} steps".format(opt))
    print("\nTraining...")

    Q, curve, snapshots = train(grid)

    print("\nepisode : steps-to-goal      (optimal = {})".format(opt))
    for e in range(0, EPISODES, max(1, EPISODES // 20)):
        print("{:7d} : {}".format(e, curve[e]))
    print("{:7d} : {}".format(EPISODES - 1, curve[-1]))

    print("\nWatch the SOLVED region grow back from the goal to the start")
    print("(@ = greedy reaches goal from here, · = not yet; START joins LAST):")
    time.sleep(2.5)
    for episode, field in snapshots:
        env.render_field(grid, field, label="after episode {}".format(episode))
        time.sleep(1.3)

    print("\nStart joined the region -> now the gubbe just follows it:")
    time.sleep(1.5)
    env.animate(grid, rollout(grid, Q, epsilon=0.0), label="the learned path")
    print("\noptimal was {} steps.".format(opt))


if __name__ == "__main__":
    main()
