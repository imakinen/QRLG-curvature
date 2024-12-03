### Description

The program `curvature.py` generates a symbolic expression representing the quantum-reduced loop gravity operator corresponding to Eq. (3.2) of [2211.04826](https://arxiv.org/abs/2211.04826). The program outputs the expression with each term printed on a separate line.

### Usage

    python curvature.py

### Options

* `-v, --one-vertex`\
Generate an expression representing the action of the operator on a one-vertex state. (See section 4.1 of [2412.01375](https://arxiv.org/abs/2412.01375).) By default the action on a generic reduced spin network state is given.
* `-l, --latex`\
Format the output as LaTeX code.
* `-t, --term`\
Instead of the entire operator, consider only a single term (1-16) in Eq. (3.2).
