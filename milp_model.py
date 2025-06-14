from ortools.math_opt.python import mathopt
import numpy as np

def create_milp_model(V_np, W_np, l_dim):
    """
    Creates a MILP model for the given V_np, W_np, and l_dim using MathOpt.

    Args:
        V_np: dxd numpy array.
        W_np: dxd numpy array.
        l_dim: integer, number of columns for X.

    Returns:
        model: mathopt.Model
        X_vars_list: List of lists of model variables for X.
        F_vars_list: List of lists of model variables for F.
    """
    model = mathopt.Model(name="milp_v_x_s_xtwx")
    d_dim = V_np.shape[0]
    epsilon = 1e-6

    # 4. Create X_vars (d_dim x l_dim matrix of model variables)
    # B_vars are binary variables (0 or 1)
    B_vars = [[model.add_binary_variable(name=f'B_{i}_{j}') for j in range(l_dim)] for i in range(d_dim)]
    # X_vars are integer variables (-1 or 1)
    X_vars_list = [[model.add_variable(lb=-1, ub=1, is_integer=True, name=f'X_{i}_{j}') for j in range(l_dim)] for i in range(d_dim)]

    # Constraint: X_ij = 2*B_ij - 1  =>  X_ij - 2*B_ij = -1
    for i in range(d_dim):
        for j in range(l_dim):
            model.add_linear_constraint(X_vars_list[i][j] - 2 * B_vars[i][j], lb=-1, ub=-1)

    # 5. Create Q_vars (l_dim x l_dim matrix of model variables for X^T @ W @ X)
    # Calculate M_q for bounds on Q_vars elements.
    # M_q is the maximum possible absolute value of any element in Q.
    # Each element Q_ab = sum_{c,e} X_ca * W_ce * X_eb. Max val is sum_{c,e} |W_ce|.
    # A simpler, generally safe upper bound for |Q_ab|: d_dim * d_dim * max_abs_W_val
    max_abs_W_val = np.max(np.abs(W_np))
    if d_dim == 0: # If d_dim is 0, W_np might be empty or invalid.
        M_q_bound = 0.0
    elif max_abs_W_val == 0: # W_np is all zeros
        M_q_bound = 0.0 # Q will always be 0
    else:
        M_q_bound = float(d_dim * d_dim * max_abs_W_val)

    # Q_vars are continuous variables, bounds [-M_q_bound, M_q_bound]
    Q_vars_list = [[model.add_variable(lb=-M_q_bound, ub=M_q_bound, name=f'Q_{a}_{b}') for b in range(l_dim)] for a in range(l_dim)]

    for a_idx in range(l_dim):
        for b_idx in range(l_dim):
            current_Q_sum_terms = []
            for c_idx in range(d_dim):
                for e_idx in range(d_dim):
                    W_ce_val = W_np[c_idx, e_idx]
                    if W_ce_val == 0:
                        continue

                    B_ca = B_vars[c_idx][a_idx]
                    B_eb = B_vars[e_idx][b_idx]

                    # prod_B_ceab = B_ca * B_eb (linearized)
                    # This auxiliary variable is binary.
                    prod_B_ceab = model.add_binary_variable(name=f'ProdB_{c_idx}_{e_idx}_{a_idx}_{b_idx}')

                    # Linearization constraints for prod_B_ceab = B_ca * B_eb:
                    # prod_B_ceab <= B_ca
                    model.add_linear_constraint(prod_B_ceab - B_ca <= 0)
                    # prod_B_ceab <= B_eb
                    model.add_linear_constraint(prod_B_ceab - B_eb <= 0)
                    # prod_B_ceab >= B_ca + B_eb - 1
                    model.add_linear_constraint(prod_B_ceab - B_ca - B_eb >= -1)

                    # Original term: W_ce_val * X_ca * X_eb
                    # X_ca * X_eb = (2*B_ca - 1) * (2*B_eb - 1)
                    #             = 4*B_ca*B_eb - 2*B_ca - 2*B_eb + 1
                    #             = 4*prod_B_ceab - 2*B_ca - 2*B_eb + 1
                    term_expr = W_ce_val * (4 * prod_B_ceab - 2 * B_ca - 2 * B_eb + 1)
                    current_Q_sum_terms.append(term_expr)

            if not current_Q_sum_terms:
                 model.add_linear_constraint(Q_vars_list[a_idx][b_idx] == 0)
            else:
                # Q_ab = sum of all term_expr
                model.add_linear_constraint(Q_vars_list[a_idx][b_idx] == sum(current_Q_sum_terms))

    # 6. Create S_vars (l_dim x l_dim matrix of model variables for sign(Q) - {0,1} valued)
    # S_ab = 1 if Q_ab > 0, else S_ab = 0 if Q_ab <= 0
    # TODO: Check MathOpt equivalent for NewBoolVar (already done, it's add_binary_variable)
    S_vars_list = [[model.add_binary_variable(name=f'S_{a}_{b}') for b in range(l_dim)] for a in range(l_dim)]

    # Use M_q_bound for S_vars constraints as well. Epsilon is still needed for strict inequality.
    # This M_q_bound was defined before Q_vars.
    for a_idx in range(l_dim):
        for b_idx in range(l_dim):
            Q_ab_val = Q_vars_list[a_idx][b_idx] # This is now a MathOpt variable
            S_ab = S_vars_list[a_idx][b_idx] # This is a MathOpt binary variable

            # Constraints for S_ab based on Q_ab_val using MathOpt indicator constraints:
            # S_ab = 1  => Q_ab_val >= epsilon
            # S_ab = 0  => Q_ab_val <= 0 (this means Q_ab can be 0 or negative)

            # S_ab = 1 => Q_ab_val >= epsilon
            model.add_indicator_constraint(S_ab, True, Q_ab_val >= epsilon)

            # S_ab = 1 => Q_ab_val <= M_q_bound (upper bound for Q when S_ab is 1)
            if M_q_bound > 0 : # Only add if M_q_bound is meaningful (Q_ab_val could be positive)
                 model.add_indicator_constraint(S_ab, True, Q_ab_val <= M_q_bound)

            # S_ab = 0 => Q_ab_val <= 0
            model.add_indicator_constraint(S_ab, False, Q_ab_val <= 0)

            # S_ab = 0 => Q_ab_val >= -M_q_bound (lower bound for Q when S_ab is 0)
            if M_q_bound > 0: # Only add if M_q_bound is meaningful (Q_ab_val could be negative)
                model.add_indicator_constraint(S_ab, False, Q_ab_val >= -M_q_bound)

            # Note on M_q_bound = 0 case:
            # If M_q_bound is 0, then Q_ab_val is fixed to 0 (since its lb=0, ub=0).
            # S_ab = 1 => 0 >= epsilon (Constraint will make S_ab=0 if epsilon > 0)
            # S_ab = 1 => 0 <= 0 (Constraint is fine)
            # S_ab = 0 => 0 <= 0 (Constraint is fine)
            # S_ab = 0 => 0 >= 0 (Constraint is fine)
            # So, if epsilon > 0 and M_q_bound = 0, S_ab will be forced to 0. This is correct.

    # 7. Create VX_prod_vars (d_dim x l_dim matrix for V @ X)
    # VX_ik = sum_j V_ij * X_jk. Max value of X_jk is 1 or -1.
    # So, M_vx_ik = sum_j |V_ij|. m_vx_ik = -sum_j |V_ij|.

    # M_vx_row_abs_sum[i] stores sum(abs(V_np[i,:])) for each row i
    M_vx_row_abs_sum = [np.sum(np.abs(V_np[i, :])) for i in range(d_dim)]

    VX_prod_vars_list = []
    for i_idx in range(d_dim):
        M_val = float(M_vx_row_abs_sum[i_idx]) # Ensure float for bounds
        m_val = -M_val
        # Assuming V_np can be float, VX_prod_vars should be continuous.
        # If V_np elements are integers, these variables will effectively be integer sums.
        # No need to explicitly set is_integer=True unless V_np is strictly integer and integer results are critical.
        row_vars = [model.add_variable(lb=m_val, ub=M_val, name=f'VX_{i_idx}_{k_idx}') for k_idx in range(l_dim)]
        VX_prod_vars_list.append(row_vars)

    # Define constraints for VX_prod_vars_list[i][k]
    for i_idx in range(d_dim):
        for k_idx in range(l_dim):
            current_VX_sum_terms = []
            for j_col_V_idx in range(d_dim): # V is d_dim x d_dim, X is d_dim x l_dim
                V_ij_val = V_np[i_idx, j_col_V_idx]
                if V_ij_val == 0:
                    continue
                # X_vars_list[j_col_V_idx][k_idx] is a MathOpt variable
                current_VX_sum_terms.append(V_ij_val * X_vars_list[j_col_V_idx][k_idx])

            if not current_VX_sum_terms:
                 model.add_linear_constraint(VX_prod_vars_list[i_idx][k_idx] == 0)
            else:
                vx_sum_expr = sum(current_VX_sum_terms)
                model.add_linear_constraint(VX_prod_vars_list[i_idx][k_idx] == vx_sum_expr)

    # 8. Create F_vars (d_dim x l_dim matrix for (V @ X) @ S)
    F_vars_list = []
    for i_idx in range(d_dim):
        # Calculate bounds for F_vars_list[i_idx][j_idx]
        # Each F_ij is sum of l_dim terms: TermF_ijk = VX_ik * S_kj
        # VX_ik (Y_ik) is bounded by [-M_Y_val, M_Y_val] where M_Y_val = M_vx_row_abs_sum[i_idx]
        # S_kj is binary {0,1}.
        # So, TermF_ijk is bounded by [min(0, -M_Y_val), max(0, M_Y_val)]
        M_Y_val_for_row = float(M_vx_row_abs_sum[i_idx])
        m_Y_val_for_row = -M_Y_val_for_row

        term_min_bound = min(0.0, m_Y_val_for_row)
        term_max_bound = max(0.0, M_Y_val_for_row)

        # Summing l_dim such terms:
        row_min_F_bound = l_dim * term_min_bound
        row_max_F_bound = l_dim * term_max_bound

        row_F_vars = [model.add_variable(lb=row_min_F_bound, ub=row_max_F_bound, name=f'F_{i_idx}_{j_idx}') for j_idx in range(l_dim)]
        F_vars_list.append(row_F_vars)

    for i_idx in range(d_dim): # F is d_dim x l_dim
        M_Y_val = float(M_vx_row_abs_sum[i_idx]) # Bound for Y_ik = VX_prod_vars_list[i_idx][k_col_S_idx]
        m_Y_val = -M_Y_val

        for j_idx in range(l_dim):
            current_F_sum_terms = []
            for k_col_S_idx in range(l_dim): # (V@X) is d_dim x l_dim, S is l_dim x l_dim. Sum over k_col_S_idx
                Y_ik = VX_prod_vars_list[i_idx][k_col_S_idx]
                s_bin_kj = S_vars_list[k_col_S_idx][j_idx] # This is a MathOpt BoolVar {0,1}

                # Linearize Y_ik * s_bin_kj using McCormick envelope
                # term_prod_var = Y_ik * s_bin_kj
                # Bounds for term_prod_var: [min(0, m_Y_val), max(0, M_Y_val)]
                min_prod_bound = min(0.0, m_Y_val) # ensure float
                max_prod_bound = max(0.0, M_Y_val) # ensure float

                term_prod_var = model.add_variable(lb=min_prod_bound, ub=max_prod_bound, name=f'TermF_{i_idx}_{j_idx}_{k_col_S_idx}')

                # McCormick envelope constraints for term_prod_var = Y_ik * s_bin_kj:
                # 1. term_prod_var <= M_Y_val * s_bin_kj  => term_prod_var - M_Y_val * s_bin_kj <= 0
                model.add_linear_constraint(term_prod_var - M_Y_val * s_bin_kj <= 0)
                # 2. term_prod_var >= m_Y_val * s_bin_kj  => term_prod_var - m_Y_val * s_bin_kj >= 0
                model.add_linear_constraint(term_prod_var - m_Y_val * s_bin_kj >= 0)
                # 3. term_prod_var <= Y_ik - m_Y_val * (1 - s_bin_kj) => term_prod_var - Y_ik + m_Y_val * (1 - s_bin_kj) <= 0
                #    term_prod_var - Y_ik + m_Y_val - m_Y_val * s_bin_kj <= 0
                model.add_linear_constraint(term_prod_var - Y_ik - m_Y_val * s_bin_kj <= -m_Y_val)
                # 4. term_prod_var >= Y_ik - M_Y_val * (1 - s_bin_kj) => term_prod_var - Y_ik + M_Y_val * (1 - s_bin_kj) >= 0
                #    term_prod_var - Y_ik + M_Y_val - M_Y_val * s_bin_kj >= 0
                model.add_linear_constraint(term_prod_var - Y_ik + M_Y_val * s_bin_kj >= -M_Y_val)

                current_F_sum_terms.append(term_prod_var)

            if not current_F_sum_terms: # Should not happen if l_dim > 0
                 model.add_linear_constraint(F_vars_list[i_idx][j_idx] == 0)
            else:
                f_sum_expr = sum(current_F_sum_terms)
                model.add_linear_constraint(F_vars_list[i_idx][j_idx] == f_sum_expr)

    return model, X_vars_list, F_vars_list # Return only required vars as per original spec for main example


if __name__ == "__main__":
    # 1. Imports are at the top of the file.
    # create_milp_model is defined above.

    # 2. Define dimensions
    d_sample = 2
    l_sample = 2

    # 3. Create sample V_sample
    V_sample = np.array([[1, 2],
                         [3, 4]], dtype=np.int64)

    # 4. Create sample W_sample
    W_sample = np.array([[5, 0],  # Set one W element to 0 to test W_ce_val == 0 condition
                         [7, 8]], dtype=np.int64)

    print(f"Running MILP model with d_dim={d_sample}, l_dim={l_sample}")
    print("V_sample:\n", V_sample)
    print("W_sample:\n", W_sample)

    # 5. Call create_milp_model
    # The function was returning more variables for debugging, adjust if necessary
    # For the main example, we only need model, X_vars, F_vars.
    # Let's modify the return statement of create_milp_model to match original requirement or unpack selectively.
    # For now, assume create_milp_model returns model, X_vars_list, F_vars_list
    model, X_vars, F_vars = create_milp_model(V_sample, W_sample, l_sample)

    # (Optional) Add a simple objective if desired, e.g. maximize sum of F_vars elements
    # objective_terms = []
    # for i in range(d_sample):
    #     for j in range(l_sample):
    #         objective_terms.append(F_vars[i][j])
    # if objective_terms:
    #    model.maximize(sum(objective_terms)) # TODO: Check MathOpt equivalent

    # 6. Create solver
    # solver = cp.CpSolver() # CP-SAT solver
    # TODO: MathOpt uses a different way to solve, e.g. mathopt.Solver(...)
    solver_type = mathopt.SolverType.GSCIP # Example, could be CP_SAT if supported for this model type

    # 7. Set time limit
    # solver.parameters.max_time_in_seconds = 30.0 # CP-SAT specific
    # solver.parameters.log_search_progress = True # CP-SAT specific
    solve_parameters = mathopt.SolveParameters(
        time_limit_sec=30.0,
        # enable_output=True # TODO: Check MathOpt equivalent for log_search_progress
    )
    # For GSCIP or other MIP solvers, you might set parameters like this:
    # import json
    # solver_options = mathopt.SolverOptions(gscip=mathopt.GScipParameters(param_values=json.dumps({"limits/time": 30})))

    # 8. Call Solve
    # status = solver.Solve(model) # CP-SAT solve
    result = mathopt.solve(model, solver_type, params=solve_parameters)


    # 9. Print solver status
    # print(f"\nSolver status: {solver.StatusName(status)}") # CP-SAT status
    print(f"\nSolver termination reason: {result.termination.reason_string()}")
    print(f"Objective value: {result.objective_value()}")


    # 10. If solution found
    # if status == cp.OPTIMAL or status == cp.FEASIBLE: # CP-SAT status
    if result.has_primal_feasible_solution():
        print("Solution found.")
        solution = result.variable_values()

        print("\nX_vars:")
        X_solution = np.zeros((d_sample, l_sample))
        for i in range(d_sample):
            row_str = []
            for j in range(l_sample):
                val = solution[X_vars[i][j]] # solver.Value(X_vars[i][j])
                X_solution[i,j] = val
                row_str.append(f"{val:3.0f}") # Ensure formatting is appropriate
            print(" ".join(row_str))

        # For verification, let's manually compute Q = X^T W X
        # Q_sol = X_solution.T @ W_sample @ X_solution
        # print("\nQ_sol (manual calculation from X_vars solution):") # TODO: Check if Q_vars can be accessed
        # print(Q_sol)
        # S_sol_manual = np.sign(Q_sol) # approx
        # print("\nS_sol_manual (sign of Q_sol):")
        # print(S_sol_manual)


        # VX_sol = V_sample @ X_solution
        # print("\nVX_sol (manual V @ X_solution):")
        # print(VX_sol)

        # F_sol_manual = VX_sol @ S_sol_manual
        # print("\nF_sol_manual (VX_sol @ S_sol_manual):")
        # print(F_sol_manual)


        print("\nF_vars:")
        for i in range(d_sample):
            row_str = []
            for j in range(l_sample):
                val = solution[F_vars[i][j]] # solver.Value(F_vars[i][j])
                row_str.append(f"{val:8.2f}") # F_vars can be large
            print(" ".join(row_str))

        # You can also retrieve other variables if create_milp_model returns them
        # And if they are MathOpt variables.
        # For example, if B_vars, Q_vars_list, S_vars_list are returned:
        # model_full, X_vars, F_vars, B_vars_sol, Q_vars_sol, S_vars_sol, VX_vars_sol = create_milp_model(V_sample, W_sample, l_sample)
        # status = solver.Solve(model_full)
        # ... then print solver.Value for those if needed for debugging.
        # print("\nQ_vars (from model):")
        # for a in range(l_sample):
        #     row_str = []
        #     for b in range(l_sample):
        #         # Need Q_vars_list from create_milp_model if we want to print them
        #         # Assuming it's returned as the 4th element for this example
        #         # _, _, _, _, Q_vars_model, _, _ = create_milp_model(V_sample, W_sample, l_sample) # This would re-create model
        #         # This requires modifying create_milp_model to return Q_vars_list or getting them from the model solution somehow if named
        #         # For now, this part is commented out as Q_vars_list is not directly returned by the simplified call
        #         # val = solver.Value(Q_vars_sol[a][b]) # if Q_vars_sol is available
        #         # row_str.append(f"{val:8.2f}")
        #         pass
        #     # print(" ".join(row_str))

        # print("\nS_vars (from model):")
        # for a in range(l_sample):
        #     row_str = []
        #     for b in range(l_sample):
        #         # val = solver.Value(S_vars_sol[a][b]) # if S_vars_sol is available
        #         # row_str.append(f"{val:3}")
        #         pass
            # print(" ".join(row_str))


    # 11. Else
    else:
        print("No solution found.")
