using LinearAlgebra
using Ipopt
using JuMP
using DifferentialEquations
using NLsolve

include("./Models/polygonal_obstacles.jl")
include("./Models/landing.jl")
include("./SCP/gusto_problem.jl")



# Integration Scheme for ODE

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



# Plotting Functions

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

function plot_square(center_2d, width_2d; color=:blue, fig=-1, lab="")
    # adapted from https://discourse.julialang.org/t/plot-a-circle-with-a-given-radius-with-plots-jl/23295
    function squareShape(h, k, wh, wk)
        return Shape(h .+ [-wh/2, wh/2, wh/2, -wh/2], k .+ [-wk/2, -wk/2, wk/2, wk/2])
        # dh = LinRange(0, wh/2, 500)
        # dk = LinRange(0, wk/2, 500)
        # h .+ dh, k .+ dk
    end

    if fig == -1 # undefined, plot new
        fig = plot(squareShape(center_2d[1],center_2d[2],width_2d[1],width_2d[2]), 
                seriestype = [:shape,], lw = 0.5,
                c = color, linecolor = :black,
                legend = false, fillalpha = 0.5, aspect_ratio = 1, label = lab)
    return fig
    else
        plot!(fig, squareShape(center_2d[1],center_2d[2],width_2d[1],width_2d[2]), 
                    seriestype = [:shape,], lw = 0.5,
                    c = color, linecolor = :black,
                    legend = false, fillalpha = 0.5, aspect_ratio = 1, label = lab)
        return fig
    end
end