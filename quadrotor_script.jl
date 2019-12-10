using LinearAlgebra
using Ipopt
using JuMP
using DifferentialEquations
using NLsolve

include("./Models/polygonal_obstacles.jl")
include("./Models/quadrotor.jl")
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

    #pyplot(grid=true)

    idx = [1,2]
    local fig
    fig = plot(X_all[1][idx[1],:], X_all[1][idx[2],:],
        label="iter = 1",linewidth=2,
        xlims=(-0.5,3.),ylims=(-0.5,6.5),
        xlabel="x Position",ylabel="y Position")
    for iter = 2:length(X_all)
        X = X_all[iter]
        plot!(fig, X[idx[1],:], X[idx[2],:],
            label="iter = $iter",linewidth=2)
    end

    for obs_i = 1:length(model.obstacles)
        p_obs, obs_radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]
        plot_circle(p_obs[idx], obs_radius; color=:red, fig=fig)
    end

    return fig
end

function plot3D_solutions(scp_problem::GuSTOProblem, model, X_all, U_all)
    N = length(X_all)

    pyplot(grid=true)

    idx = [1,2,3]
    local fig
    fig = plot(X_all[1][idx[1],:],X_all[1][idx[2],:],X_all[1][idx[3],:],
        linewidth=2,label="iter = 1",
        xlims=(-0.1,0.5),ylims=(-0.3,0.3),zlims=(-0.2,0.2),
        xlabel="x Position",ylabel="y Position",zlabel="z Position")
    for iter = 2:length(X_all)
        X = X_all[iter]
        plot!(fig,X[idx[1],:],X[idx[2],:],X[idx[3],:],
            linewidth=2,label="iter = $iter")
    end

    for obs_i = 1:length(model.obstacles)
        p_obs, obs_radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]
        plot_sphere(p_obs[idx], obs_radius; color=:red, fig=fig)
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
                fillalpha = 0.5, aspect_ratio = 1, label = lab)
    return fig
    else
        plot!(fig, circleShape(position_2d[1],position_2d[2],radius),
                    seriestype = [:shape,], lw = 0.5,
                    c = color, linecolor = :black,
                    fillalpha = 0.5, aspect_ratio = 1, label = lab)
        return fig
    end
end

function plot_sphere(position_3d, radius; color=:blue, fig=-1, lab="")
    if fig == -1 # undefined, plot new
        u = LinRange(-pi/2.0,pi/2.0,50)
        v = LinRange(0,2*pi,100)
        x, y, z = [], [], []

        for i = 1:length(u)
            for j = 1:length(v)
                push!(x,position_3d[1] + radius*cos(u[i])*cos(v[j]))
                push!(y,position_3d[2] + radius*cos(u[i])*sin(v[j]))
                push!(z,position_3d[3] + radius*sin(u[i]))
            end
        end

        fig = plot(x,y,z,linetype=:surface,label=lab,colorbar=false,
            seriestype = [:shape,],lw = 0.5,
            c = color,fillalpha = 0.5)
    return fig
    else
        u = LinRange(-pi/2.0,pi/2.0,50)
        v = LinRange(0,2*pi,100)
        x, y, z = [], [], []

        for i = 1:length(u)
            for j = 1:length(v)
                push!(x,position_3d[1] + radius*cos(u[i])*cos(v[j]))
                push!(y,position_3d[2] + radius*cos(u[i])*sin(v[j]))
                push!(z,position_3d[3] + radius*sin(u[i]))
            end
        end

        plot!(fig,x,y,z,linetype=:surface,label=lab,colorbar=false,
            seriestype = [:shape,],lw = 0.5,
            c  = color,fillalpha = 0.5)
        return fig
    end
end