using LinearAlgebra
using Ipopt
using JuMP
using DifferentialEquations
using NLsolve

include("./Models/astrobee_se3.jl")
include("./SCP/gusto_problem.jl")



function solveRK4(scp_problem::GuSTOProblem,model,dynamics::Function,dim,N,tf,z0,t0=0.0)
    h = (tf - t0)/(N - 1)
    t = zeros(N)
    z = zeros(dim,N)
    dynTemp = zeros(dim)
    zTemp = zeros(dim)
    t[1] = t0
    z[:,1] = z0

    for i = 1:N-1
        t[i+1] = t[i] + h

        dynamics(scp_problem,model,dynTemp,z[:,i],t[i])
        z[:,i+1] = z[:,i] + h*dynTemp/6.0

        zTemp = z[:,i] + h*dynTemp/2.0
        dynamics(scp_problem,model,dynTemp,zTemp,t[i]+h/2.0)
        z[:,i+1] = z[:,i+1] + h*dynTemp/3.0

        zTemp = z[:,i] + h*dynTemp/2.0
        dynamics(scp_problem,model,dynTemp,zTemp,t[i]+h/2.0)
        z[:,i+1] = z[:,i+1] + h*dynTemp/3.0

        zTemp = z[:,i] + h*dynTemp
        dynamics(scp_problem,model,dynTemp,zTemp,t[i]+h)
        z[:,i+1] = z[:,i+1] + h*dynTemp/6.0
    end

    return t, z
end



function solve_shooting(scp_problem::GuSTOProblem,model,dual_variable)
    x_init, x_goal, tf = model.x_init, model.x_final, model.tf_guess
    dt_min             = scp_problem.dt
    p0                 = dual_variable

    # Set up shooting function
    shooting_eval! = (F, p_init) -> parameterized_shooting_eval!(F, scp_problem, model, p_init)

    # Run Newton method
    sol_newton = nlsolve(shooting_eval!, p0, iterations = 100, ftol=1e-6)
    @show sol_newton.f_converged

    x_shooting = []
    if sol_newton.f_converged
        x_shooting = solve_shooting_once(scp_problem, model, sol_newton.zero)
    else
        x_shooting = []
    end


    return x_shooting, sol_newton
end



function parameterized_shooting_eval!(F, scp_problem::GuSTOProblem,model,dual_variable)
    x_init, x_goal, tf = model.x_init, model.x_final, model.tf_guess
    dt_min             = scp_problem.dt

    x0 = vcat(x_init, dual_variable)

    #tspan = (0., tf)
    #shooting_ode_func! = (x_dot,x,unused_var,t) -> shooting_ode!(x_dot, scp_problem, model, x, t)

    prob = ODEProblem(shooting_ode_func!, x0, tspan, -1)
    sol = DifferentialEquations.solve(prob, dtmin=dt_min, force_dtmin=true, saveat=dt_min)

    sol_shooting = hcat(sol.u...)
    x_shooting, p_shooting = sol_shooting[1:x_dim,:], sol_shooting[x_dim+1:end,:]

    xN = x_shooting[:,end]

    for i = 1:x_dim
        F[i] = x_goal[i] - xN[i]
    end
end



# function solve_shooting_once(scp_problem::GuSTOProblem, model, dual_variable)
#     x_init, x_goal, tf = model.x_init, model.x_final, model.tf_guess
#     dt_min             = scp_problem.dt
#     p0                 = dual_variable

#     x0 = vcat(x_init, p0)
#     tspan = (0., tf)

#     shooting_ode_func! = (x_dot,x,unused_var,t) -> shooting_ode!(x_dot, scp_problem, model, x, t)

#     prob = ODEProblem(shooting_ode_func!, x0, tspan, -1)
#     # sol = DifferentialEquations.solve(prob)
#     sol = DifferentialEquations.solve(prob, dtmin=dt_min, force_dtmin=true, saveat=dt_min)

#     sol_shooting = hcat(sol.u...)
#     x_shooting, p_shooting = sol_shooting[1:x_dim,:], sol_shooting[x_dim+1:end,:]

#     return x_shooting
# end



function shooting_ode!(scp_problem::GuSTOProblem,model,xdot,x,t)
    x_dim = model.x_dim

    x, p = x[1:x_dim], x[x_dim+1:end]

    u = get_control_shooting(x, p, model)

    # Add contributions
    xdot[:] = zeros(2*x_dim)
    xdot[:]      = f_dyn_shooting(x, u, p, model)
    # xdot[14:26] += state_convex_constraints_penalty_shooting(model, x)
    # xdot[14:16] += obs_avoidance_penalty_grad_all_shooting(model, x)
end








# ---------------------
# -     PLOTTING      -
function plot_solutions(scp_problem::GuSTOProblem, model, X_all, U_all; x_shooting_all=[])
    N = length(X_all)

    idx = [1,2]
    local fig
    fig = plot(X_all[1][idx[1],:], X_all[1][idx[2],:])
    for iter = 2:length(X_all)
        X = X_all[iter]
        plot!(fig, X[idx[1],:], X[idx[2],:]; c=:blue)
    end

    for obs_i = 1:length(model.obstacles)
        p_obs, obs_radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]
        plot_circle(p_obs[idx], obs_radius; color=:red, fig=fig)
    end

    local fig
    for x_shooting in x_shooting_all
        if size(x_shooting,1) > 1
            plot!(fig, x_shooting[idx[1],:], x_shooting[idx[2],:]; c=:red)
        else
            plot!(fig, [0;0], [0;0])
        end
    end

    return fig
end


function plot_circle(position_2d, radius; color=:blue, fig=-1, lab="")
    # adapted from https://discourse.julialang.org/t/plot-a-circle-with-a-given-radius-with-plots-jl/23295
    function circleShape(h, k, r)
        theta = LinRange(0, 2*pi, 500)
        h .+ r*sin.(theta), k .+ r*cos.(theta)
    end

    if fig == -1 # undefined, plot new
        fig = plot(circleShape(position_2d[1],position_2d[2],radius), 
                seriestype = [:shape,], lw = 0.5,
                c = color, linecolor = :black,
                legend = false, fillalpha = 0.5, aspect_ratio = 1, label = lab)
    return fig
    else
        plot!(fig, circleShape(position_2d[1],position_2d[2],radius), 
                    seriestype = [:shape,], lw = 0.5,
                    c = color, linecolor = :black,
                    legend = false, fillalpha = 0.5, aspect_ratio = 1, label = lab)
        return fig
    end
end