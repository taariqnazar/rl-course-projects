# Reinforcement Learning: Course Projects

Projects for the reinforcement learning course, based on Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.).

| Project | Topic | Instructions | Helper code |
|---|---|---|---|
| [Project 1](project-1) | Dynamic programming and Monte Carlo methods (ch. 1–5) | [PDF](project-1/instructions/project-1.pdf) | [helpers](project-1/helpers) |
| [Project 2](project-2) | Coming soon | | |
| [Project 3](project-3) | Coming soon | | |

Each released project folder contains:

- `instructions/`: the assignment as a Typst source and compiled PDF.
- `helpers/`: code you may use in your solution.

## Writing your report

Hand in one typed PDF. Follow the exercise numbering (A1, A2, …) so each answer is easy to find, and put your code in an appendix.

Keep it short. Answer each question with a figure or table and a few sentences that say what it shows. Describe what *you* observed, not what the book says should happen.

**Figures.** Use the type that fits the result:

| Result | Figure |
|---|---|
| Learning curve (error or value vs. episodes or sweeps) | Mean over seeds as a line, with a shaded ±1 std band |
| Variance between runs is the point | One thin line per seed |
| Value function on a grid | Heat map with the value written in each cell |
| Policy on a grid | Arrows in each cell |
| A handful of numbers | A table, not a plot |

Every figure should:

- have a number and a caption, and be referred to in the text,
- have labelled axes, and a legend when it shows more than one curve,
- use a log axis when values span several orders of magnitude,
- use the same colour for the same method throughout the report,
- show one message; split it into two figures if you need two.

Random results need at least 10 seeds. Say how many you used.

![Example learning curve: mean over 10 seeds with a ±1 std band](common/figures/example-plot.svg)

*Example of a learning-curve figure (synthetic data).* The helper code in each project includes functions for these plot types.

## Building the PDFs

```sh
typst compile --root . project-1/instructions/project-1.typ
```
