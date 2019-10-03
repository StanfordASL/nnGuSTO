export GuSTOProblem

mutable struct GuSTOProblem
    dt

    omega
    Delta

    problem

    X
    U
    Xp
    Up
end

function GuSTOProblem(model, N, Xp, Up, solver=Ipopt.Optimizer)
    dt    = model.tf_guess / (N-1)
    omega = model.omega0
    Delta = model.Delta0

    problem = Model(with_optimizer(Ipopt.Optimizer))
    JuMP.register(problem, :h_penalty, 1, h_penalty, autodiff=true)

    X = @variable(problem, X[1:model.x_dim,1:N  ])
    U = @variable(problem, U[1:model.u_dim,1:N-1])

    GuSTOProblem(dt,
                 omega, Delta,
                 problem,
                 X, U, Xp, Up)
end


# function define_all_constraints(model)
#         constraints = []
#             constraints += m.get_constraints(self.var['X'], self.var['U'], self.par['X_last'], self.par['U_last'])



# function add_constraints(model)



#### PENALTY function
function h_penalty(t)
    if t <= 0
        return 0.
    else
        return t^2
    end
end
# function ∇h_penalty(t)
#     if t <= 0
#         return 0.
#     else
#         return 2*t
#     end
# end


function set_parameters(problem::GuSTOProblem, model,
                        Xp, Up)
    problem.Xp = Xp
    problem.Up = Up
end

function define_problem(problem::GuSTOProblem, model)
    add_penalties(problem::GuSTOProblem, model)
end


function add_penalties(scp_problem::GuSTOProblem, model)
    problem = scp_problem.problem

    X, U, Xp, Up = scp_problem.X, scp_problem.U, scp_problem.Xp, scp_problem.Up

    N = length(X[1,:])
    omega = scp_problem.omega


    # k = 1
    # i = 1
    # state_max_convex_constraints_function = X -> state_max_convex_constraints(model, X, U, Xp, Up, k, i)
    # h_penalty_function = X -> h_penalty(state_max_convex_constraints_function(X))
    #my_expr = @NLexpression(problem, h_penalty(X[1,1]))
    # my_expr = @NLexpression(problem, h_penalty_function(X))


    # @NLexpression(problem, h_expr[k = 1:N], h_penalty(X[1,k]) )



    # @NLexpression(model, my_expr[i = 1:n], sin(x[i]))
    # @NLconstraint(model, my_constr[i = 1:n], my_expr[i] <= 0.5)



    # for k = 1:N
    #     for i = 1:model.x_dim
    #         @show i

    #         constraint = x_ki -> h_penalty(state_max_convex_constraints(model, x_ki, k, i))
    #         constraint_name = Symbol(string("constraint_",k,i))

    #         @show constraint_name

    #         JuMP.register(problem, constraint_name, 1, constraint, autodiff=true)

    #         @NLexpression(problem, my_expr[ii = 1:model.x_dim, kk = 1:N], 
    #                                 ( constraint(X[ii,kk]) )       )
    #         # @NLexpression(problem, h_expr[k = 1:N], h_penalty(X[1,k]) )

    #         # @NLobjective(problem, Min, constraint(X[i,k]))

    #         # constraint = state_max_convex_constraints(model, X, U, Xp, Up, k, i)
    #         # #@NLobjective(problem, Min, omega*h_penalty(constraint))
    #         # constraint = state_min_convex_constraints(model, X, U, Xp, Up, k, i)
    #         # #@NLobjective(problem, Min, omega*h_penalty(constraint))


    #         #my_expr = @NLexpression(problem, omega/*)
    #     end
    # end



    for k = 1:N
        for i = 1:model.x_dim


            cost_penalization = omega*max(X[1,2], 0.)

            @objective(solver_model, Min, cost_penalization)




            constraint = state_max_convex_constraints(model, X, U, Xp, Up, k, i)


            @objective(problem, Min, omega*pos(X[1,2], 0.))
            constraint = state_min_convex_constraints(model, X, U, Xp, Up, k, i)
            @objective(problem, Min, omega*pos(constraint, 0.))
        end
    end

    # for k = 1:N-1
    #     for i = 1:model.u_dim
    #         constraint = control_max_convex_constraints(model, X, U, Xp, Up, k, i)
    #         #@NLobjective(problem, Min, omega*h_penalty(constraint))
    #         constraint = control_min_convex_constraints(model, X, U, Xp, Up, k, i)
    #         #@NLobjective(problem, Min, omega*h_penalty(constraint))
    #     end
    # end
end