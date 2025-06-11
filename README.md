# MILP Model for V @ X @ s(X^T @ W @ X)

This repository contains a Python script (`milp_model.py`) that uses Google's OR-Tools to create a Mixed Integer Linear Programming (MILP) model for the mathematical expression `F = V @ X @ s(X^T @ W @ X)`.

## Mathematical Formulation

The components are:
- `V`: A known `d x d` matrix.
- `W`: A known `d x d` matrix.
- `X`: A `d x l` matrix of decision variables, where each element `X_ij` can take values `+1` or `-1`.
- `s(Q)`: The element-wise sign function applied to matrix `Q`. If `Q_ab > 0`, `s(Q_ab) = 1`. If `Q_ab < 0`, `s(Q_ab) = -1`. If `Q_ab = 0`, `s(Q_ab) = 0`.
- `Q = X^T @ W @ X`: This results in an `l x l` matrix.
- `@`: Denotes standard matrix multiplication.

The script defines variables and constraints to represent this expression `F` (which is a `d x l` matrix) within a CP-SAT model.

## Script: `milp_model.py`

### Function: `create_milp_model(V_np, W_np, l_dim)`

- **Inputs:**
    - `V_np`: A `numpy` array of shape `(d,d)` representing matrix V.
    - `W_np`: A `numpy` array of shape `(d,d)` representing matrix W.
    - `l_dim`: An integer representing the number of columns `l` for matrix X.
- **Outputs:**
    - `model`: The `ortools.sat.python.cp_model.CpModel` object containing all variables and constraints.
    - `X_vars`: A `d x l` Python list of lists, where `X_vars[i][j]` is the CP-SAT integer variable for `X_ij`.
    - `F_vars`: A `d x l` Python list of lists, where `F_vars[i][j]` is the CP-SAT numeric variable representing the element `F_ij` of the final expression.

### Usage Example
The script includes a `if __name__ == "__main__":` block that demonstrates how to:
1. Define sample `V`, `W` matrices and `l`.
2. Call `create_milp_model` to build the model.
3. Solve the model using `CpSolver`.
4. Print the values of `X_vars` and `F_vars` if a solution is found.

## Dependencies
- Python 3.x
- `ortools`
- `numpy`

To install dependencies:
```bash
pip install ortools numpy
```

## Running the Example
To run the example provided in the script:
```bash
python milp_model.py
```
This will build and solve the model for a small sample instance and print the results. The main purpose of this function is to return the model and variables so that further objectives or constraints can be added externally.
