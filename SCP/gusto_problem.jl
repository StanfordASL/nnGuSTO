# ----------------------------------------
# -               CLASS                  -
# ----------------------------------------

export GuSTOProblem

mutable struct GuSTOProblem
    N
    dt

    omega
    Delta
    dimLinearConstraintsU
    dimSecondOrderConeConstraintsU

    solver_model

    X
    U
    Xp
    Up

    initial_constraint
end

function GuSTOProblem(model, N, dimLinearConstraintsU, dimSecondOrderConeConstraintsU, Xp, Up, solver=Ipopt.Optimizer)
    N     = N
    dt    = model.tf_guess / (N-1)
    omega = model.omega0
    Delta = model.Delta0

    solver_model = Model(with_optimizer(Ipopt.Optimizer, print_level=0))
    X = @variable(solver_model, X[1:model.x_dim,1:N  ])
    U = @variable(solver_model, U[1:model.u_dim,1:N-1])

    GuSTOProblem(N, dt,
                 omega, Delta, dimLinearConstraintsU, dimSecondOrderConeConstraintsU,
                 solver_model,
                 X, U, Xp, Up,
                 [])
end

function reset_problem(scp_problem::GuSTOProblem, model, solver=Ipopt.Optimizer)
    scp_problem.solver_model = Model(with_optimizer(solver, print_level=0))
    N = scp_problem.N
    X = @variable(scp_problem.solver_model, X[1:model.x_dim,1:N  ])
    U = @variable(scp_problem.solver_model, U[1:model.u_dim,1:N-1])
    scp_problem.X       = X
    scp_problem.U       = U
end

function set_parameters(scp_problem::GuSTOProblem, model,
                        Xp, Up, omega, Delta)
    scp_problem.Xp = Xp
    scp_problem.Up = Up
    scp_problem.omega = omega
    scp_problem.Delta = Delta
end

function accuracy_ratio(problem::GuSTOProblem, model, X, U, Xp, Up)
    N = length(X[1,:])

    num = 0.0
    den = 0.0

    # Dynamics
    for k in 1:N-1
        x_k  = X[:, k]
        u_k  = U[:, k]
        x_kp = Xp[:,k]
        u_kp = Up[:,k]

        f_k  = f_dyn(x_k,  u_k,  model)
        A_k  = A_dyn(x_k,  u_k,  model)
        B_k  = B_dyn(x_k,  u_k,  model)
        f_kp = f_dyn(x_kp, u_kp, model)
        A_kp = A_dyn(x_kp, u_kp, model)
        B_kp = B_dyn(x_kp, u_kp, model)

        linearized = f_kp + A_kp*(x_k-x_kp) + B_kp*(u_k-u_kp)
        num += norm(f_k - linearized, 2)
        den += norm(linearized,       2)
    end

    # Obstacles
    for k = 1:N
        for i = 1:length(model.obstacles)
            constraint            = obstacle_constraint(            model, X, U, Xp, Up, k, i)
            constraint_linearized = obstacle_constraint_convexified(model, X, U, Xp, Up, k, i)
            
            num += norm(constraint - constraint_linearized, 2)
            den += norm(constraint_linearized,              2)
        end
    end

    # Percentage ratio
    accuracy_ratio = num*100.0/den

    return accuracy_ratio
end



# ----------------------------------------
# -               COSTS                  -
# ----------------------------------------

function define_cost(scp_problem::GuSTOProblem, model)
    total_cost =  add_true_cost(scp_problem, model)
    total_cost += add_penalties(scp_problem, model)
    @objective(scp_problem.solver_model, Min, total_cost)
end

function add_true_cost(scp_problem::GuSTOProblem, model)
    solver_model = scp_problem.solver_model
    X, U, Xp, Up = scp_problem.X, scp_problem.U, scp_problem.Xp, scp_problem.Up
    return true_cost(model, X, U, Xp, Up)
end

function add_penalties(scp_problem::GuSTOProblem, model)
    solver_model = scp_problem.solver_model
    x_dim, u_dim = model.x_dim, model.u_dim
    omega, Delta = scp_problem.omega, scp_problem.Delta

    X, U, Xp, Up = scp_problem.X, scp_problem.U, scp_problem.Xp, scp_problem.Up
    N, dt        = length(X[1,:]), scp_problem.dt

    penalization = 0.

    # -----------------
    # STATE CONSTRAINTS
    # -----------------
    # Dual variables to reformulate max(a,0) constraints
    @variable(solver_model, lambdas_state_max_convex_constraints[i=1:x_dim, k=1:N])
    @variable(solver_model, lambdas_state_min_convex_constraints[i=1:x_dim, k=1:N])
    # Penalized constraints
    for k = 1:N
        for i = 1:x_dim
            lambda_max     = lambdas_state_max_convex_constraints[i,k]
            constraint_max = state_max_convex_constraints(model, X, U, Xp, Up, k, i)
            lambda_min     = lambdas_state_min_convex_constraints[i,k]
            constraint_min = state_min_convex_constraints(model, X, U, Xp, Up, k, i)

            @constraint(solver_model, lambda_max <= 0.)
            penalization += omega*(constraint_max-lambda_max)^2
            @constraint(solver_model, lambda_min <= 0.)
            penalization += omega*(constraint_min-lambda_min)^2
        end
    end
    # -----------------

    # -------------------------------------
    # SECOND ORDER CONE CONTROL CONSTRAINTS
    # -------------------------------------
    # @variable(solver_model, lambdas_second_order_cone_control_constraints[i=1:dimSecondOrderConeConstraintsU, k=1:N])
    # for k = 1:N-1
    #     for i = 1:dimSecondOrderConeConstraintsU
    #         lambda     = lambdas_second_order_cone_control_constraints[i,k]
    #         constraint = control_second_order_cone_constraints(model, X, U, Xp, Up, k, i)

    #         @constraint(solver_model, lambda <= 0.)
    #         penalization += omega*(constraint-lambda)^2
    #     end
    # end
    # ------------------------------------

    # -------------
    # TRUST REGIONS
    # -------------
    @variable(solver_model, lambdas_trust_max_convex_constraints[i=1:x_dim, k=1:N])
    @variable(solver_model, lambdas_trust_min_convex_constraints[i=1:x_dim, k=1:N])
    for k = 1:N
        for i = 1:x_dim
            lambda_max     = lambdas_trust_max_convex_constraints[i,k]
            constraint_max = trust_region_max_constraints(model, X, U, Xp, Up, k, i, Delta)
            lambda_min     = lambdas_trust_min_convex_constraints[i,k]
            constraint_min = trust_region_min_constraints(model, X, U, Xp, Up, k, i, Delta)

            @constraint(solver_model, lambda_max <= 0.)
            penalization += omega*(constraint_max-lambda_max)^2
            @constraint(solver_model, lambda_min <= 0.)
            penalization += omega*(constraint_min-lambda_min)^2
        end
    end
    # ------------

    # ---------
    # OBSTACLES
    # ---------

    # Spheres
    Nb_obstacles = length(model.obstacles)
    @variable(solver_model, lambdas_obstacles[i=1:Nb_obstacles, k=1:N])
    for k = 1:N
        for i = 1:Nb_obstacles
            lambda     = lambdas_obstacles[i,k]
            constraint = obstacle_constraint_convexified(model, X, U, Xp, Up, k, i)

            @constraint(solver_model, lambda <= 0.)
            # ε = model.epsilon # Soft obstacles
            # @constraint(solver_model, lambda <= ε)
            penalization += omega*(constraint-lambda)^2
        end
    end

    # Polygonal obstacles
    Nb_poly_obstacles = length(model.poly_obstacles)
    @variable(solver_model, lambdas_poly_obstacles[i=1:Nb_poly_obstacles, k=1:N])
    for k = 1:N
        for i = 1:Nb_poly_obstacles
            lambda     = lambdas_poly_obstacles[i,k]
            constraint = poly_obstacle_constraint_convexified(model, X, U, Xp, Up, k, i)

            @constraint(solver_model, lambda <= 0.)
            penalization += omega*(constraint-lambda)^2
        end
    end
    # --------

    return penalization
end

function satisfies_state_inequality_constraints(scp_problem::GuSTOProblem, model, X, U, Xp, Up, Delta)
    B_satisfies_constraints = true
    x_dim = model.x_dim
    N = scp_problem.N
    epsilon = model.epsilon
    
    # -----------------
    # STATE CONSTRAINTS
    # -----------------
    for k = 1:N
        for i = 1:x_dim
            constraint_max = state_max_convex_constraints(model, X, U, [], [], k, i)
            constraint_min = state_min_convex_constraints(model, X, U, [], [], k, i)
            if constraint_max > epsilon || constraint_min > epsilon
                B_satisfies_constraints = false
            end
        end
    end

    # -------------
    # TRUST REGIONS
    # -------------
    for k = 1:N
        for i = 1:x_dim
            constraint_max = trust_region_max_constraints(model, X, U, Xp, Up, k, i, Delta)
            constraint_min = trust_region_min_constraints(model, X, U, Xp, Up, k, i, Delta)
            if constraint_max > epsilon || constraint_min > epsilon
                B_satisfies_constraints = false
            end
        end
    end

    # ---------
    # OBSTACLES
    # ---------
    for k = 1:N
        # Spheres
        for i = 1:length(model.obstacles)
            constraint = obstacle_constraint(model, X, U, [], [], k, i)
            if constraint > epsilon
                B_satisfies_constraints = false
            end
        end

        # Polygonal obstacles
        # for i = 1:length(model.poly_obstacles)
        #     constraint = poly_obstacle_constraint(model, X, U, [], [], k, i)
        #     if constraint > 0.
        #         B_satisfies_constraints = false
        #     end
        # end
    end

    return B_satisfies_constraints
end
# ----------------------------------------



# --------------------------------------------
# -                CONSTRAINTS               -
# --------------------------------------------

function define_constraints(scp_problem::GuSTOProblem, model)
    add_initial_constraints(scp_problem, model)
    add_final_constraints(scp_problem, model)
    add_dynamics_constraints(scp_problem, model)
    add_control_constraints(scp_problem, model)
end

function add_initial_constraints(scp_problem::GuSTOProblem, model)
    solver_model = scp_problem.solver_model
    x_dim, u_dim = model.x_dim, model.u_dim
    omega, Delta = scp_problem.omega, scp_problem.Delta

    X, U, Xp, Up = scp_problem.X, scp_problem.U, scp_problem.Xp, scp_problem.Up

    constraint = state_initial_constraints(model, X, U, Xp, Up)
    scp_problem.initial_constraint = @constraint(solver_model, constraint .== 0.)
end

function add_final_constraints(scp_problem::GuSTOProblem, model)
    solver_model = scp_problem.solver_model
    x_dim, u_dim = model.x_dim, model.u_dim
    omega, Delta = scp_problem.omega, scp_problem.Delta

    X, U, Xp, Up = scp_problem.X, scp_problem.U, scp_problem.Xp, scp_problem.Up

    constraint = state_final_constraints(model, X, U, Xp, Up)
    @constraint(solver_model,  constraint - 0.001 .<= 0.)
    @constraint(solver_model, -constraint - 0.001 .<= 0.)
end

function add_dynamics_constraints(scp_problem::GuSTOProblem, model)
    solver_model = scp_problem.solver_model
    x_dim, u_dim = model.x_dim, model.u_dim
    omega, Delta = scp_problem.omega, scp_problem.Delta

    X, U, Xp, Up = scp_problem.X, scp_problem.U, scp_problem.Xp, scp_problem.Up
    N, dt        = length(X[1,:]), scp_problem.dt

    for k = 1:N-1
        X_knext  = X[:, k+1]
        X_k      = X[:, k]
        U_k      = U[:, k]
        X_kp     = Xp[:, k]
        U_kp     = Up[:, k]
        f_dyn_kp = model.f[k]
        A_kp     = model.A[k]
        B_kp     = model.B[k]

        # if k < N-2 # Trapezoidal rule
        #     U_knext  = U[:, k+1]
        #     X_knextp = Xp[:, k+1]
        #     U_knextp = Up[:, k+1]
        #     f_dyn_knext_p = model.f[k+1]
        #     A_knext_p     = model.A[k+1]
        #     B_knext_p     = model.B[k+1]

        #     constraint = X_knext - (
        #                     X_k + dt/2 * (  (   f_dyn_kp + 
        #                                         A_kp * (X_k-X_kp) + 
        #                                         B_kp * (U_k-U_kp)   )  
        #                                     + 
        #                                     (   f_dyn_knext_p + 
        #                                         A_knext_p * (X_knext-X_knextp) + 
        #                                         B_knext_p * (U_knext-U_knextp)   )
        #                                   )
        #                             )

        #     @constraint(solver_model, constraint .== 0.)
        # else # Euler
            constraint =  X_knext - ( X_k + dt * (  f_dyn_kp + 
                                                    A_kp * (X_k-X_kp) + 
                                                    B_kp * (U_k-U_kp)
                                                 )
                                    )
            @constraint(solver_model, constraint .== 0.)
        # end
    end
end

function add_control_constraints(scp_problem::GuSTOProblem, model)
    solver_model = scp_problem.solver_model
    x_dim, u_dim = model.x_dim, model.u_dim
    dimLinearConstraintsU, dimSecondOrderConeConstraintsU = scp_problem.dimLinearConstraintsU, scp_problem.dimSecondOrderConeConstraintsU

    X, U, Xp, Up = scp_problem.X, scp_problem.U, scp_problem.Xp, scp_problem.Up
    N            = length(X[1,:])

    for k = 1:N-1
        for i = 1:dimLinearConstraintsU
            # Linear constraints
            constraint = control_linear_constraints(model, X, U, Xp, Up, k, i)
            @constraint(solver_model, constraint <= 0.)
        end
        # for i = 1:dimSecondOrderConeConstraintsU
        #     # Second order cone constraints
        #     t, x = control_second_order_cone_constraints(model, X, U, Xp, Up, k, i)
        #     @constraint(solver_model, [t; x] in SecondOrderCone())
        # end
    end
end
# --------------------------------------------
