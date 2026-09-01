# mazelab.py
# The "plumbing" for rung 8 - the maze world, the perfect verifier (BFS),
# and the ASCII drawing. NONE of this is machine learning. It only exists so
# that rung8.py can focus purely on the reinforcement learning. Ignore it.

import random
import time
from collections import deque

SIZE = 21          # odd; the maze is SIZE x SIZE (bigger = harder)
SEED = 20          # change for a different maze (seeded = reproducible)
ANIM_DELAY = 0.15  # seconds between animation frames - RAISE to go slower

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right
START = (1, 1)


def goal_of(size):
    return (size - 2, size - 2)


def make_maze(size, seed):
    rng = random.Random(seed)
    grid = [[1] * size for _ in range(size)]

    def carve(r, c):
        grid[r][c] = 0
        dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 1 <= nr < size - 1 and 1 <= nc < size - 1 and grid[nr][nc] == 1:
                grid[r + dr // 2][c + dc // 2] = 0
                carve(nr, nc)

    carve(1, 1)
    grid[size - 2][size - 2] = 0
    return grid


def make_sparse(seed, wall_prob=0.10):
    """A grid with SCATTERED obstacles (not a dense maze) AND a RANDOM goal (not
    always the corner). Returns (grid, goal). Always keeps a path from START to
    that goal (retries until it does)."""
    goal = goal_of(SIZE)
    for attempt in range(120):
        rng = random.Random(seed * 997 + attempt)
        grid = [[0] * SIZE for _ in range(SIZE)]
        for r in range(SIZE):
            for c in range(SIZE):
                if r == 0 or c == 0 or r == SIZE - 1 or c == SIZE - 1:
                    grid[r][c] = 1
                elif rng.random() < wall_prob:
                    grid[r][c] = 1
        grid[START[0]][START[1]] = 0
        goal = (rng.randrange(1, SIZE - 1), rng.randrange(1, SIZE - 1))
        grid[goal[0]][goal[1]] = 0
        if goal != START and optimal_steps(grid, goal) is not None:
            return grid, goal
    return grid, goal


def move(pos, action, grid):
    """Try to move; if it would hit a wall or the edge, STAY put."""
    nr, nc = pos[0] + action[0], pos[1] + action[1]
    if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == 0:
        return (nr, nc)
    return pos


def observe(pos, grid):
    """What the agent SEES from a cell: LOCAL features that mean the SAME thing
    in every maze (unlike an absolute cell position) - this is the key to
    generalisation. Returns (wall_up, wall_down, wall_left, wall_right,
    goal_row_dir, goal_col_dir)."""
    r, c = pos
    goal = goal_of(len(grid))
    walls = []
    for dr, dc in ACTIONS:
        nr, nc = r + dr, c + dc
        open_here = 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == 0
        walls.append(0 if open_here else 1)
    goal_row_dir = (goal[0] > r) - (goal[0] < r)   # +1 goal below, -1 above, 0 same
    goal_col_dir = (goal[1] > c) - (goal[1] < c)   # +1 goal right, -1 left, 0 same
    return (walls[0], walls[1], walls[2], walls[3], goal_row_dir, goal_col_dir)


def optimal_steps(grid, goal=None):
    """BFS = the TRUE shortest number of steps. The oracle you grade against."""
    if goal is None:
        goal = goal_of(len(grid))
    queue = deque([(START, 0)])
    seen = {START}
    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == goal:
            return dist
        for dr, dc in ACTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) \
                    and grid[nr][nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
    return None


def render(grid, agent=None, path=None, goal=None):
    if goal is None:
        goal = goal_of(len(grid))
    path = set(path or [])
    for r in range(len(grid)):
        row = ""
        for c in range(len(grid[0])):
            if (r, c) == agent:
                row += "@ "
            elif (r, c) == goal:
                row += "G "
            elif grid[r][c] == 1:
                row += "██"
            elif grid[r][c] == 2:
                row += "x "                        # a LETHAL trap tile (rung 11)
            elif (r, c) in path:
                row += "* "
            else:
                row += "· "
        print(row)


def animate(grid, path, delay=ANIM_DELAY, label="", goal=None):
    for i in range(len(path)):
        print("\033[H\033[J", end="")   # clear screen
        render(grid, agent=path[i], path=path[: i + 1], goal=goal)
        if label:
            print("\n" + label)
        print("step {} / {}".format(i, len(path) - 1))
        time.sleep(delay)


def render_field(grid, field, label=""):
    """Draw the 'solved region'. @ = greedy reaches the goal from here,
    · = not yet. Watch @ grow from G back to S (start joins LAST)."""
    goal = goal_of(len(grid))
    print("\033[H\033[J", end="")
    for r in range(len(grid)):
        row = ""
        for c in range(len(grid[0])):
            if (r, c) == START:
                row += "S "
            elif (r, c) == goal:
                row += "G "
            elif grid[r][c] == 1:
                row += "##"
            elif field[r][c]:
                row += "@ "
            else:
                row += "· "
        print(row)
    if label:
        print("\n" + label)
