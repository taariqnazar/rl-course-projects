// Project 2: Temporal-Difference Learning and Function Approximation

#set document(title: "Project 2: Temporal-Difference Learning and Function Approximation", author: "Taariq Nazar")
#set page(paper: "a4", margin: 2.5cm, numbering: "1")
#set text(size: 11pt)
#set par(justify: true)
#set heading(numbering: none)
#show link: set text(fill: blue)

#let pts(n) = text(weight: "bold", fill: rgb("#555"))[(#n p)]
#let lead(body) = par(text(style: "italic", fill: rgb("#555"), body))

#align(center)[
  #set par(justify: false)
  #text(size: 18pt, weight: "bold")[Project 2: Temporal-Difference Learning and Function Approximation]
  #v(0.3em)
  Taariq Nazar \
  #link("mailto:taariq.nazar@math.su.se")[taariq.nazar\@math.su.se]
]

= Instructions

Hand in a single typed PDF with your answers, plots and tables, and your code as an appendix. Keep answers short. A plot plus two sentences beats a page of text. See the #link("https://github.com/taariqnazar/rl-course-projects#writing-your-report")[repository README] for how to write the report.

You may use any programming language. Python with NumPy is recommended: the helper code at #link("https://github.com/taariqnazar/rl-course-projects/tree/main/project-2/helpers")[`github.com/taariqnazar/rl-course-projects`] (folder `project-2/helpers`) provides the environments (`environments.py`), a tile coder (`tiles.py`) and plotting functions (`plotting.py`).

The project is worth 100 points and you need at least 50 to pass. The _Grading Criteria_ document on the course page explains how this project counts towards your final grade.

A late submission loses 15 points. A failed submission that is resubmitted loses 7 points, plus the 15 points for being late.

Whenever results are random, use *at least 10 random seeds*. Unless an exercise says otherwise, plot the mean with a shaded band of ±1 standard deviation. See @fig-example for an example.

= The environments

All three are examples from Sutton & Barto; the helpers implement them exactly as described there. All are undiscounted ($gamma = 1$).

- *Random walk* (Examples 6.2, 7.1 and 9.1). States $1, dots, n$ in a row, start in the middle, move left or right at random. Entering the left end gives $-1$ (0 in Example 6.2), the right end $+1$, everything else $0$. `RandomWalk(5, left_reward=0)` and `RandomWalk(19)` move one state per step; `RandomWalk(1000, max_jump=100)` jumps up to 100 states. `env.true_values()` returns the exact values, so you can measure the error of every estimate.
- *Cliff walking* (Example 6.6). A $4 times 12$ grid with deterministic moves. Every step gives $-1$; stepping into the cliff gives $-100$ and sends the agent back to the start.
- *Mountain Car* (Example 10.1). A car in a valley must rock back and forth to reach the top. The state $(x, v)$ is continuous, there are three actions, and every step gives $-1$.

= Part A: Tabular methods (45 p)

#lead[Monte Carlo waits until the end of an episode. TD updates after every step, using its own estimates (Sutton & Barto, Ch. 6).]

== A1. TD(0) vs Monte Carlo prediction #pts(10)

Use the 5-state random walk of Example 6.2 (`RandomWalk(5, left_reward=0)`) and start from $V equiv 0.5$.

+ Implement TD(0) (§6.1) and constant-$alpha$ MC (every-visit, eq. 6.1). Run each for 100 episodes and record the RMSE over the 5 states after every episode. Use $alpha in {0.05, 0.1, 0.15}$ for TD and $alpha in {0.01, 0.02, 0.03, 0.04}$ for MC, and plot all seven curves in one figure.
+ Which method has the lower error after 10 and after 100 episodes? How does the choice of $alpha$ trade early speed against final error?

== A2. n-step TD #pts(10)

#lead[n-step methods sit between one-step TD and Monte Carlo (Ch. 7).]

+ Implement n-step TD prediction (§7.1) on the 19-state random walk from $V equiv 0$. For $n in {1, 2, 4, 8, 16, 512}$ and $alpha in {0, 0.1, dots, 1}$, compute the RMSE over the 19 states, averaged over the first 10 episodes and over 100 runs. Plot error against $alpha$, one curve per $n$ (mean only; limit the $y$-axis to $[0.25, 0.55]$).
+ Which $n$ and $alpha$ give the lowest error? How does the best $alpha$ change with $n$, and why? Which earlier methods are $n = 1$ and (approximately) $n = 512$?

== A3. Sarsa, Q-learning and Expected Sarsa #pts(15)

#lead[The same idea gives control: learn action values and act $epsilon$-greedily.]

Use cliff walking with $epsilon = 0.1$, $alpha = 0.5$ and $Q equiv 0$, for 500 episodes. Break ties between equal action values at random.

+ Implement Sarsa (§6.4), Q-learning (§6.5) and Expected Sarsa (§6.6). Plot the return per episode for all three as learning curves (smooth each run with a moving average over 10 episodes; limit the $y$-axis to $[-100, 0]$).
+ For one seed, show the greedy policy each method has learned after 500 episodes (helper: `plot_cliff_policy`). Note if it does not reach the goal.
+ Which method learns the shortest path? Which collects the most reward while learning? Explain the difference.

== A4. n-step Sarsa #pts(10)

#lead[n-step returns work for control as well.]

+ Implement n-step Sarsa (§7.2) on cliff walking with $epsilon = 0.1$. For $n in {1, 2, 4, 8}$ and $alpha in {0.1, 0.2, 0.4, 0.6, 0.8}$, record the average return over the first 200 episodes. Plot it against $alpha$, one curve per $n$.
+ Which values of $n$ and $alpha$ work well? How does the best $alpha$ depend on $n$?

= Part B: Function approximation (45 p)

#lead[A table needs one value per state. With many or continuous states we learn a parameterised $hat(v)(s, bold(w))$ instead (Ch. 9, 10, 12).]

== B1. Prediction with state aggregation and tile coding #pts(15)

Use the 1000-state random walk. The helper `TileCoder` covers both feature types: `TileCoder([1], [1001], n_tilings=1, tiles_per_dim=10)` is state aggregation with 10 groups of 100 states, and `TileCoder([1], [1001], n_tilings=50, tiles_per_dim=5)` is 50 tilings of width 200, each offset by 4 states (§9.5.4). Measure the error as the unweighted RMSE over all 1000 states.

+ With 10 groups, implement gradient Monte Carlo (§9.3, $alpha = 2 times 10^(-4)$) and semi-gradient TD(0) (§9.3, $alpha = 2 times 10^(-3)$). Plot their RMSE against episodes, up to 5000.
+ For one seed after 5000 episodes, plot both approximations together with the true values against the state. Why is the result a staircase, and why does TD's staircase differ from Monte Carlo's?
+ Compare gradient Monte Carlo with one tiling of width 200 (`TileCoder([1], [1001], n_tilings=1, tiles_per_dim=5)`, $alpha = 5 times 10^(-4)$) and with 50 tilings of width 200 ($alpha = 5 times 10^(-4) slash 50$). Plot the RMSE against episodes, up to 5000. Why does tile coding reach a lower error, even though each tile is just as wide?

== B2. The $lambda$-return and TD($lambda$) #pts(12)

#block(breakable: false)[
#lead[Instead of choosing one $n$, the $lambda$-return averages all n-step returns.]

Use the 19-state random walk with one feature per state, so that $hat(v)(s, bold(w)) = w_s$.

+ Implement the offline $lambda$-return algorithm (§12.1): at the end of each episode, compute every $G_t^lambda$ with the values from the start of the episode, then apply the updates $w_(S_t) <- w_(S_t) + alpha (G_t^lambda - w_(S_t))$ in order.
+ Implement semi-gradient TD($lambda$) with accumulating traces (§12.2).
+ For both, and for $lambda in {0, 0.4, 0.8, 0.9, 0.95, 1}$ and $alpha in {0, 0.1, dots, 1}$, compute the RMSE averaged over the first 10 episodes and over 100 runs. Plot error against $alpha$, one curve per $lambda$ (mean only; limit the $y$-axis to $[0.25, 0.55]$). TD($lambda$) may diverge for large $alpha$ and $lambda$; leave those points out.
+ Which $lambda$ works best? Compare with your n-step results from A2. Where do the offline $lambda$-return and TD($lambda$) differ, and why?
]

== B3. Semi-gradient Sarsa on Mountain Car #pts(18)

#lead[Approximating action values gives control in a continuous state space.]

Use tile coding with 8 tilings of $8 times 8$ tiles over $(x, v)$ (`TileCoder` with its default arguments and the bounds in `MountainCar`), one weight vector per action, and $bold(w) = bold(0)$. Act greedily ($epsilon = 0$), breaking ties at random. Stop an episode after 10000 steps.

+ Implement episodic semi-gradient Sarsa (§10.1). Plot learning curves of the return per episode over 500 episodes for $alpha in {0.1 slash 8, 0.2 slash 8, 0.5 slash 8}$ (limit the $y$-axis to $[-1000, 0]$).
+ For one run with $alpha = 0.5 slash 8$, plot the cost-to-go $-max_a hat(q)(s, a, bold(w))$ after 1, 12, 100 and 500 episodes (helper: `plot_cost_to_go`). Describe how it changes.
+ The agent never takes a random action. Why does it still explore?
+ Implement n-step semi-gradient Sarsa (§10.2). Compare $n = 1$ ($alpha = 0.5 slash 8$) with $n = 8$ ($alpha = 0.3 slash 8$) in one learning-curve plot. Which learns faster?

= Part C: Putting it together (10 p)

Fill in the table below *from your own experiments*, for the methods as you used them, and add a few sentences on what $n$ and $lambda$ control.

#block(breakable: false, align(center, table(
  columns: 5, align: left, stroke: 0.5pt,
  table.header([*Method*], [*Bootstraps?*], [*On- or off-policy*], [*Used with function approx.?*], [*Main weakness*]),
  [On-policy MC (Project 1)], [], [], [], [],
  [TD(0)], [], [], [], [],
  [n-step TD / TD($lambda$)], [], [], [], [],
  [Sarsa], [], [], [], [],
  [Q-learning], [], [], [], [],
  [Expected Sarsa], [], [], [], [],
  [Semi-gradient Sarsa], [], [], [], [],
)))

#figure(
  image("/common/figures/example-plot.svg", width: 62%),
  caption: [Example learning curve: mean over 10 seeds with a ±1 standard deviation band; the dashed line is the optimum. Synthetic data.],
) <fig-example>
