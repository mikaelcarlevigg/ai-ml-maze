# rung9.py - THE LEAP: a NEURAL NET from scratch (pure Python, no libraries).
#
# rung 8's notebook (Q-table) stores ONE number per situation, so it can only
# answer situations it has already SEEN. A neural net instead COMPUTES the
# answer from a handful of weights - so it can answer situations it has NEVER
# seen. That is "function approximation", and it is why nets generalise where
# tables can't. (It's also, scaled up enormously, what an LLM is.)
#
# The task (kept deliberately simple - no maze yet): given the offset to a goal
# (how far up/down and left/right it is), pick which of 4 directions heads
# toward it. The net learns this from examples, then works on offsets it never
# trained on. A table, asked about an unseen offset, has nothing to say.
#
# directions:  up=0  down=1  left=2  right=3

import random
import math

random.seed(1)
HIDDEN = 12      # neurons in the middle layer
LR = 0.05        # learning rate (same role as ALPHA in rung 8)
STEPS = 60000    # how many training examples to learn from


def correct_action(dr, dc):
    """The 'right answer' we train against: head along the BIGGER offset.
    dr = goal_row - my_row (negative = goal is above me).
    dc = goal_col - my_col (negative = goal is to my left)."""
    if abs(dr) >= abs(dc):
        return 0 if dr < 0 else 1        # up if goal is above, else down
    return 2 if dc < 0 else 3            # left if goal is left, else right


# ---------------------------------------------------------------------
# THE NET'S BRAIN = weights. NOT one number per situation (that was the
# table). Just these few numbers - combined by forward() - answer for ANY
# input. Layer 1 turns the 2 inputs into HIDDEN "features"; layer 2 turns
# those into 4 scores (one per direction).
# ---------------------------------------------------------------------
def rnd():
    return random.uniform(-0.5, 0.5)

W1 = [[rnd() for _ in range(HIDDEN)] for _ in range(2)]   # 2 inputs  -> HIDDEN
b1 = [0.0] * HIDDEN
W2 = [[rnd() for _ in range(4)] for _ in range(HIDDEN)]   # HIDDEN    -> 4 outputs
b2 = [0.0] * 4


def forward(x):
    """Run the 2 inputs through the net and get 4 scores out.
    A neuron is just: a weighted sum of its inputs + a bias, then ReLU
    (max(0, .)), which lets the net bend into non-straight-line shapes."""
    z1 = [sum(x[i] * W1[i][j] for i in range(2)) + b1[j] for j in range(HIDDEN)]
    hidden = [max(0.0, z) for z in z1]                                     # ReLU
    out = [sum(hidden[j] * W2[j][k] for j in range(HIDDEN)) + b2[k] for k in range(4)]
    return z1, hidden, out


def train():
    """The learning. For each example: GUESS, measure the ERROR, then push
    every weight a little in the direction that shrinks that error. Carrying
    the error backward through the net to find each weight's share of the
    blame is BACKPROPAGATION - the net's version of rung 8's little nudge."""
    for step in range(STEPS):
        dr = random.uniform(-10, 10)
        dc = random.uniform(-10, 10)
        x = [dr / 10.0, dc / 10.0]           # feed the net the offset (scaled)
        y = correct_action(dr, dc)           # the right answer for this example

        z1, hidden, out = forward(x)

        # softmax: turn the 4 raw scores into probabilities that sum to 1
        m = max(out)
        ex = [math.exp(v - m) for v in out]
        total = sum(ex)
        p = [e / total for e in ex]

        # error at the output = predicted probability - target (1 for the right
        # action, 0 for the others). "How wrong were we, per direction."
        grad_out = [p[k] - (1.0 if k == y else 0.0) for k in range(4)]

        # backprop: carry the blame back to the hidden layer...
        grad_hidden = [sum(grad_out[k] * W2[j][k] for k in range(4)) for j in range(HIDDEN)]

        # ...then nudge every weight DOWN its share of the blame (a gradient step)
        for j in range(HIDDEN):
            grad_z = grad_hidden[j] if z1[j] > 0 else 0.0    # ReLU: blame only where it was active
            for i in range(2):
                W1[i][j] -= LR * x[i] * grad_z
            b1[j] -= LR * grad_z
        for j in range(HIDDEN):
            for k in range(4):
                W2[j][k] -= LR * hidden[j] * grad_out[k]
        for k in range(4):
            b2[k] -= LR * grad_out[k]


def predict(dr, dc):
    _, _, out = forward([dr / 10.0, dc / 10.0])
    return out.index(max(out))


def main():
    print("Training the net on random goal-offsets...")
    train()

    # For contrast: a TABLE that memorises the answer for exact offsets it saw.
    table = {}
    for _ in range(STEPS):
        key = (round(random.uniform(-10, 10), 2), round(random.uniform(-10, 10), 2))
        table[key] = correct_action(*key)

    net_ok = tab_ok = 0
    trials = 3000
    for _ in range(trials):
        dr = round(random.uniform(-10, 10), 2)
        dc = round(random.uniform(-10, 10), 2)
        want = correct_action(dr, dc)
        if predict(dr, dc) == want:
            net_ok += 1
        if table.get((dr, dc)) == want:          # unseen exact key -> None -> wrong
            tab_ok += 1

    print("\nOn offsets NEITHER has trained on:")
    print("  NET   got {:.0%} right  (it COMPUTES the answer -> generalises)".format(net_ok / trials))
    print("  TABLE got {:.0%} right  (no entry for an unseen key -> can't)".format(tab_ok / trials))

    print("\nA few example predictions (0=up 1=down 2=left 3=right):")
    for dr, dc in [(-7, 2), (5, -1), (1, 8), (-3, -9)]:
        print("  goal offset (dr={:+}, dc={:+}) -> net says {}   (right answer {})".format(
            dr, dc, predict(dr, dc), correct_action(dr, dc)))


if __name__ == "__main__":
    main()
