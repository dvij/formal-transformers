import ortools.sat.python.cp_model as cp
import numpy as np

def create_milp_model(V_np, W_np, l_dim):
    """
    Creates a MILP model for the given V_np, W_np, and l_dim.

    Args:
        V_np: dxd numpy array.
        W_np: dxd numpy array.
        l_dim: integer, number of columns for X.

    Returns:
        model: ortools.sat.python.cp_model.CpModel
        X_vars_list: List of lists of model variables for X.
        F_vars_list: List of lists of model variables for F.
    """
    model = cp.CpModel()
    d_dim = V_np.shape[0]
    epsilon = 1e-6

    # 4. Create X_vars (d_dim x l_dim matrix of model variables)
    B_vars = [[model.NewBoolVar(f'B_{i}_{j}') for j in range(l_dim)] for i in range(d_dim)]
    X_vars_list = [[model.NewIntVar(-1, 1, f'X_{i}_{j}') for j in range(l_dim)] for i in range(d_dim)]
    for i in range(d_dim):
        for j in range(l_dim):
            model.Add(X_vars_list[i][j] == 2 * B_vars[i][j] - 1)

    # 5. Create Q_vars (l_dim x l_dim matrix of model variables for X^T @ W @ X)
    Q_vars_list = [[model.NewIntVar(-cp.INT_MAX, cp.INT_MAX, f'Q_{a}_{b}') for b in range(l_dim)] for a in range(l_dim)]
    for a in range(l_dim):
        for b in range(l_dim):
            current_Q_sum_terms = []
            for c in range(d_dim):
                for e in range(d_dim):
                    W_ce_val = W_np[c, e]
                    if W_ce_val == 0:
                        continue

                    # X_ca = X_vars_list[c][a]
                    # X_eb = X_vars_list[e][b]
                    B_ca = B_vars[c][a]
                    B_eb = B_vars[e][b]

                    prod_B_ceab = model.NewBoolVar(f'ProdB_{c}_{e}_{a}_{b}')
                    # Linearization for B_ca * B_eb
                    model.Add(prod_B_ceab <= B_ca)
                    model.Add(prod_B_ceab <= B_eb)
                    model.Add(prod_B_ceab >= B_ca + B_eb - 1)

                    # term_val = W_ce_val * (X_ca * X_eb)
                    # X_ca * X_eb = (2*B_ca - 1) * (2*B_eb - 1) = 4*B_ca*B_eb - 2*B_ca - 2*B_eb + 1
                    # X_ca * X_eb = 4*prod_B_ceab - 2*B_ca - 2*B_eb + 1
                    term_val_expr = W_ce_val * (4 * prod_B_ceab - 2 * B_ca - 2 * B_eb + 1)
                    current_Q_sum_terms.append(term_val_expr)

            if not current_Q_sum_terms:
                 model.Add(Q_vars_list[a][b] == 0)
            else:
                model.Add(Q_vars_list[a][b] == cp.LinearExpr.Sum(current_Q_sum_terms))

    # 6. Create S_vars (l_dim x l_dim matrix of model variables for sign(Q))
    S_vars_list = [[model.NewIntVar(-1, 1, f'S_{a}_{b}') for b in range(l_dim)] for a in range(l_dim)]
    # Store is_pos, is_neg, is_zero for S to be used in F_vars linearization
    S_is_pos_vars = [[model.NewBoolVar(f'is_S_pos_{a}_{b}') for b in range(l_dim)] for a in range(l_dim)]
    S_is_neg_vars = [[model.NewBoolVar(f'is_S_neg_{a}_{b}') for b in range(l_dim)] for a in range(l_dim)]
    S_is_zero_vars = [[model.NewBoolVar(f'is_S_zero_{a}_{b}') for b in range(l_dim)] for a in range(l_dim)]

    M_q_max_abs_W = np.max(np.abs(W_np))
    if M_q_max_abs_W == 0 : # Handle case where W is all zeros
        M_q = epsilon
    else:
        M_q = d_dim * d_dim * M_q_max_abs_W + epsilon


    for a in range(l_dim):
        for b in range(l_dim):
            Q_ab = Q_vars_list[a][b]
            S_ab = S_vars_list[a][b]

            is_pos_ab = S_is_pos_vars[a][b] # model.NewBoolVar(f'is_pos_Q_{a}_{b}')
            is_neg_ab = S_is_neg_vars[a][b] # model.NewBoolVar(f'is_neg_Q_{a}_{b}')
            is_zero_ab = S_is_zero_vars[a][b] # model.NewBoolVar(f'is_zero_Q_{a}_{b}')

            model.Add(is_pos_ab + is_neg_ab + is_zero_ab == 1)
            model.Add(S_ab == is_pos_ab - is_neg_ab)

            # Constraints for sign definition
            model.Add(Q_ab >= epsilon).OnlyEnforceIf(is_pos_ab)
            model.Add(Q_ab <= M_q).OnlyEnforceIf(is_pos_ab) # Upper bound for positive Q

            model.Add(Q_ab <= -epsilon).OnlyEnforceIf(is_neg_ab)
            model.Add(Q_ab >= -M_q).OnlyEnforceIf(is_neg_ab) # Lower bound for negative Q

            model.Add(Q_ab == 0).OnlyEnforceIf(is_zero_ab)
            # model.Add(Q_ab > -epsilon).OnlyEnforceIf(is_zero_ab) # Redundant given Q_ab == 0
            # model.Add(Q_ab < epsilon).OnlyEnforceIf(is_zero_ab)  # Redundant given Q_ab == 0


    # 7. Create VX_prod_vars (d_dim x l_dim matrix for V @ X)
    VX_prod_vars_list = [[model.NewIntVar(-cp.INT_MAX, cp.INT_MAX, f'VX_{i}_{k}') for k in range(l_dim)] for i in range(d_dim)]
    M_vx_ik_abs_sum_V_row = [sum(abs(V_np[i,j]) for j in range(d_dim)) for i in range(d_dim)]


    for i in range(d_dim):
        for k in range(l_dim):
            current_VX_sum_terms = []
            for j_col_V in range(d_dim): # V is d_dim x d_dim, X is d_dim x l_dim
                V_ij_val = V_np[i, j_col_V]
                if V_ij_val == 0:
                    continue
                current_VX_sum_terms.append(V_ij_val * X_vars_list[j_col_V][k])

            if not current_VX_sum_terms:
                 model.Add(VX_prod_vars_list[i][k] == 0)
            else:
                model.Add(VX_prod_vars_list[i][k] == cp.LinearExpr.Sum(current_VX_sum_terms))

    # 8. Create F_vars (d_dim x l_dim matrix for (V @ X) @ S)
    F_vars_list = [[model.NewIntVar(-cp.INT_MAX, cp.INT_MAX, f'F_{i}_{j}') for j in range(l_dim)] for i in range(d_dim)]

    for i in range(d_dim): # F is d_dim x l_dim
        # Max possible value for VX_prod_vars_list[i][k] for a given i
        # This is sum over p of abs(V_np[i,p]) * max_abs_X, and max_abs_X is 1
        M_vx_ik = M_vx_ik_abs_sum_V_row[i]

        for j in range(l_dim):
            current_F_sum_terms = []
            for k in range(l_dim): # (V@X) is d_dim x l_dim, S is l_dim x l_dim
                VX_ik = VX_prod_vars_list[i][k]
                # S_kj = S_vars_list[k][j] # This is -1, 0, or 1

                # Linearize VX_ik * S_kj
                # Let Y = VX_ik and s_val = S_kj. We want Z = Y*s_val.
                s_is_1_kj = S_is_pos_vars[k][j] # model.NewBoolVar(f's_is_1_{k}_{j}_for_F')
                s_is_neg1_kj = S_is_neg_vars[k][j] # model.NewBoolVar(f's_is_neg1_{k}_{j}_for_F')
                # s_is_0_kj directly from S_is_zero_vars[k][j]

                # model.Add(S_vars_list[k][j] == 1).OnlyEnforceIf(s_is_1_kj) # Already defined
                # model.Add(S_vars_list[k][j] == -1).OnlyEnforceIf(s_is_neg1_kj) # Already defined
                # model.Add(S_vars_list[k][j] == 0).OnlyEnforceIf(S_is_zero_vars[k][j]) # Already defined
                # model.Add(s_is_1_kj + s_is_neg1_kj + S_is_zero_vars[k][j] == 1) # Already defined

                # Term for S_kj = 1
                Z_if_s_is_1 = model.NewIntVar(-M_vx_ik, M_vx_ik, f'Z_s1_{i}_{j}_{k}')
                model.Add(Z_if_s_is_1 == VX_ik).OnlyEnforceIf(s_is_1_kj)
                model.Add(Z_if_s_is_1 == 0).OnlyEnforceIf(s_is_1_kj.Not())

                # Term for S_kj = -1
                Z_if_s_is_neg1 = model.NewIntVar(-M_vx_ik, M_vx_ik, f'Z_s_neg1_{i}_{j}_{k}')
                model.Add(Z_if_s_is_neg1 == -VX_ik).OnlyEnforceIf(s_is_neg1_kj) # Note the negation
                model.Add(Z_if_s_is_neg1 == 0).OnlyEnforceIf(s_is_neg1_kj.Not())

                # If S_kj is 0, both Z_if_s_is_1 and Z_if_s_is_neg1 will be 0.
                # So term_prod_var = Z_if_s_is_1 + Z_if_s_is_neg1 correctly captures VX_ik * S_kj
                term_prod_var = Z_if_s_is_1 + Z_if_s_is_neg1
                current_F_sum_terms.append(term_prod_var)

            if not current_F_sum_terms: # Should not happen if l_dim > 0
                 model.Add(F_vars_list[i][j] == 0)
            else:
                model.Add(F_vars_list[i][j] == cp.LinearExpr.Sum(current_F_sum_terms))

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
    #    model.Maximize(cp.LinearExpr.Sum(objective_terms))


    # 6. Create solver
    solver = cp.CpSolver()

    # 7. Set time limit
    solver.parameters.max_time_in_seconds = 30.0
    solver.parameters.log_search_progress = True


    # 8. Call Solve
    status = solver.Solve(model)

    # 9. Print solver status
    print(f"\nSolver status: {solver.StatusName(status)}")

    # 10. If solution found
    if status == cp.OPTIMAL or status == cp.FEASIBLE:
        print("Solution found.")

        print("\nX_vars:")
        X_solution = np.zeros((d_sample, l_sample))
        for i in range(d_sample):
            row_str = []
            for j in range(l_sample):
                val = solver.Value(X_vars[i][j])
                X_solution[i,j] = val
                row_str.append(f"{val:3}")
            print(" ".join(row_str))

        # For verification, let's manually compute Q = X^T W X
        # Q_sol = X_solution.T @ W_sample @ X_solution
        # print("\nQ_sol (manual calculation from X_vars solution):")
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
                val = solver.Value(F_vars[i][j])
                row_str.append(f"{val:8.2f}") # F_vars can be large
            print(" ".join(row_str))

        # You can also retrieve other variables if create_milp_model returns them
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
