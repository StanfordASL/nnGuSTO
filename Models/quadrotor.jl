export Quadrotor

mutable struct Quadrotor
    # State: r, v
    x_dim
    # Control: u, Γ
    u_dim

    # Dynamics
    f
    A
    B

    # Model constants
    gravity
    uMin
    uMax
    θMax

    # Problem  
    x_init
    x_final
    tf_guess
    xMin
    xMax

    # Sphere obstacles [(x,y),r]
    obstacles
    # Polygons
    poly_obstacles

    # GuSTO parameters
    Delta0
    omega0
    omegamax
    epsilon
    rho0
    rho1
    beta_succ
    beta_fail
    gamma_fail
    convergence_threshold
end

function Quadrotor()
    x_dim = 6
    u_dim = 4

    # Model constants
    gravity = [0;0;9.81]
    uMin = 0.6
    uMax = 23.2
    θMax = pi/3.0

    # Problem
    x_init  = [0;0;0 ; 0;0;0]
    x_final = [2.5;6.;0 ; 0;0;0]
    tf_guess = 2.#1.99 # s
    myInf = 1.0e6
    xMin = [-0.1;-0.1;-myInf ; -myInf;-myInf;-myInf]
    xMax = [4;7;myInf ; myInf;myInf;myInf]

    # GuSTO parameters
    Delta0 = 100.
    omega0 = 1.
    omegamax = 1.0e9
    epsilon = 1.0e-3
    rho0 = 10.0
    rho1 = 20.0
    beta_succ = 2.
    beta_fail = 0.5
    gamma_fail = 5.
    convergence_threshold = 5.0


    # Cylindrical obstacles [(x,y),r (= datum given in the CSM paper)]
    obstacles = []
    obs = [[1.0,2.0],1.0/2.5]
    push!(obstacles, obs)
    obs = [[2.0,5.0],1.0/2.5]
    push!(obstacles, obs)

    # Polygonal obstacles
    poly_obstacles = []

    Quadrotor(x_dim, u_dim,
             [], [], [],
             gravity, uMin, uMax, θMax,
             x_init, x_final, tf_guess, xMin, xMax,
             obstacles,
             poly_obstacles,
             Delta0,
             omega0,
             omegamax,
             epsilon,
             rho0,
             rho1,
             beta_succ,
             beta_fail,
             gamma_fail,
             convergence_threshold)
end

function get_initial_gusto_parameters(m::Quadrotor)
    return m.Delta0, m.omega0, m.omegamax, m.epsilon, m.rho0, m.rho1, m.beta_succ, m.beta_fail, m.gamma_fail, m.convergence_threshold
end

function initialize_trajectory(model::Quadrotor, N::Int)
  x_dim,  u_dim   = model.x_dim, model.u_dim
  x_init, x_final = model.x_init, model.x_final
  dimR  = Int(model.x_dim/2)
  
  X = zeros(x_dim, N)
  x0 = x_init[1:dimR]
  x1 = x_final[1:dimR]
  X[1:dimR,:] = hcat(range(x0, stop=x1, length=N)...)
  for i = 1:N
    X[dimR+1:2*dimR,i] = zeros(dimR)#(x1-x0)/norm(x1-x0)
  end
  U = zeros(u_dim, N-1)

  return X, U
end

function convergence_metric(model::Quadrotor, X, U, Xp, Up)
    x_dim = model.x_dim
    N = length(X[1,:])

    # Normalized maximum relative error between iterations
    max_num, max_den = -Inf, -Inf
    for k in 1:N
        val = norm(X[1:x_dim,k] - Xp[1:x_dim,k])
        max_num = val > max_num ? val : max_num

        val = norm(X[1:x_dim,k])
        max_den = val > max_den ? val : max_den
    end

    # Percentage error
    return max_num*100.0/max_den
end



# --------------------------------------------
# -               OBJECTIVE                  -
# --------------------------------------------

function true_cost(model::Quadrotor, X, U, Xp, Up)
    x_dim, u_dim = model.x_dim, model.u_dim
    cost = 0.

    # ∫ Γ(t)^2 dt
    for k = 1:length(U[1,:])
        cost += U[u_dim,k]^2
    end

    return cost
end

# --------------------------------------------



# --------------------------------------------
# -               CONSTRAINTS                -
# --------------------------------------------

# Linear
# function control_linear_constraints(model::Quadrotor, X, U, Xp, Up, k, i)
#     x_dim, u_dim = model.x_dim, model.u_dim
#     uMin, uMax, θMax = model.uMin, model.uMax, model.θMax

#     if i == 1
#       return uMin - U[u_dim,k]
#     elseif i == 2
#       return U[u_dim,k] - uMax
#     elseif i == 3
#       return U[u_dim,k]*cos(θMax) - U[u_dim-1,k]
#     else
#       println("ERROR - TOO MANY LINEAR CONTROL CONSTRAINTS")
#     end
# end

function control_linear_constraints(model::Quadrotor, X, U, Xp, Up, k, i)
    x_dim, u_dim = model.x_dim, model.u_dim
    uMin, uMax, θMax = model.uMin, model.uMax, model.θMax

    if i == 1
      return uMin - U[u_dim,k]
    elseif i == 2
      return U[u_dim,k] - uMax
    elseif i == 3
      return U[u_dim,k]*cos(θMax) - U[u_dim-1,k]
    elseif i == 4
      return U[1,k] - U[u_dim,k]
    elseif i == 5
      return  -U[u_dim,k] - U[1,k]
    elseif i == 6
      return U[2,k] - U[u_dim,k]
    elseif i == 7
      return  -U[u_dim,k] - U[2,k]
    elseif i == 8
      return U[3,k] - U[u_dim,k]
    elseif i == 9
      return  -U[u_dim,k] - U[3,k]
    else
      println("ERROR - TOO MANY LINEAR CONTROL CONSTRAINTS")
    end
end

# Second order cone
# function control_second_order_cone_constraints(model::Quadrotor, X, U, Xp, Up, k, i)
#     x_dim, u_dim = model.x_dim, model.u_dim
#     uMin, uMax, θMax = model.uMin, model.uMax, model.θMax

#     if i == 1
#       return U[u_dim,k], U[1:u_dim-1,k]
#     else
#       println("ERROR - TOO MANY SECOND ORDER CONE CONTROL CONSTRAINTS")
#     end
# end

# Linearized second order cone
# function control_second_order_cone_constraints(model::Quadrotor, X, U, Xp, Up, k, i)
#     x_dim, u_dim = model.x_dim, model.u_dim
#     uMin, uMax, θMax = model.uMin, model.uMax, model.θMax
#     u, up, Γ, Γp = U[1:u_dim-1,k], Up[1:u_dim-1,k], U[u_dim,k], Up[u_dim,k]

#     if i == 1
#       return ( norm(up)^2 - Γp^2 + sum(2 * up[i] * (u[i] - up[i]) for i=1:u_dim-1) - 2 * Γp * (Γ - Γp) )
#     else
#       println("ERROR - TOO MANY SECOND ORDER CONE CONTROL CONSTRAINTS")
#     end
# end

function state_max_convex_constraints(model::Quadrotor, X, U, Xp, Up, k, i)
    return ( X[i, k] - model.xMax[i] )
end

function state_min_convex_constraints(model::Quadrotor, X, U, Xp, Up, k, i)
    return ( model.xMin[i] - X[i, k] )
end

function trust_region_max_constraints(model::Quadrotor, X, U, Xp, Up, k, i, Delta)
    return ( (X[i, k] - Xp[i, k]) - Delta )
end
function trust_region_min_constraints(model::Quadrotor, X, U, Xp, Up, k, i, Delta)
    return ( -Delta - (X[i, k] - Xp[i, k]) )
end

function is_in_trust_region(model::Quadrotor, X, U, Xp, Up, Delta)
    B_is_inside = true

    for k = 1:length(X[1,:])
        for i = 1:model.x_dim
            if trust_region_max_constraints(model, X, U, Xp, Up, k, i, Delta) > 0.
                B_is_inside = false
            end
            if trust_region_min_constraints(model, X, U, Xp, Up, k, i, Delta) > 0.
                B_is_inside = false
            end
        end
    end

    return B_is_inside
end

function state_initial_constraints(model::Quadrotor, X, U, Xp, Up)
    return (X[:,1] - model.x_init)
end

function state_final_constraints(model::Quadrotor, X, U, Xp, Up)
    return (X[:,end] - model.x_final)
end

function obstacle_constraint(model::Quadrotor, X, U, Xp, Up, k, obs_i)
    dimR  = Int(model.x_dim/2)
    p_obs, obs_radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]
    p_k  = X[1:dimR-1,k]
    
    dist = norm(p_k - p_obs, 2)
    # constraint = -( dist^2 - obs_radius^2 )
    constraint = -( dist - obs_radius )

    return constraint
end

function obstacle_constraint_convexified(model::Quadrotor, X, U, Xp, Up, k, obs_i)
    dimR  = Int(model.x_dim/2)
    p_obs, obs_radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]
    p_k  = X[1:dimR-1,k]
    p_kp = Xp[1:dimR-1,k]
    
    # dist_prev = norm(p_kp - p_obs, 2)
    # dir_prev = p_kp - p_obs
    # constraint = -( dist_prev^2 - obs_radius^2 + sum(2. * dir_prev[i] * (p_k[i] - p_kp[i]) for i=1:dimR-1) )

    dist_prev = norm(p_kp - p_obs, 2)
    dir_prev = (p_kp - p_obs)/dist_prev
    constraint = -( dist_prev - obs_radius + sum(dir_prev[i] * (p_k[i] - p_kp[i]) for i=1:dimR-1) )

    return constraint
end

# --------------------------------------------



# --------------------------------------------
# -                DYNAMICS                  -
# --------------------------------------------

# In continuous time, for all trajectory
function compute_dynamics(model, Xp, Up)
    N = length(Xp[1,:])

    f_all, A_all, B_all = [], [], []

    for k in 1:N-1
        x_k = Xp[:,k]
        u_k = Up[:,k]

        f_dyn_k, A_dyn_k, B_dyn_k = f_dyn(x_k, u_k, model), A_dyn(x_k, u_k, model), B_dyn(x_k, u_k, model)

        push!(f_all, f_dyn_k)
        push!(A_all, A_dyn_k)
        push!(B_all, B_dyn_k)
    end

    return f_all, A_all, B_all
end

function f_dyn(x::Vector, u::Vector, model::Quadrotor)
  x_dim, u_dim = model.x_dim, model.u_dim
  dimR = Int(model.x_dim/2)

  g = model.gravity
  f = zeros(x_dim)

  r, v = x[1:dimR], x[dimR+1:2*dimR]
  w, Γ = u[1:u_dim-1], u[u_dim]

  f[1:dimR] = v
  f[dimR+1:2*dimR] = w - g

  return f
end

function A_dyn(x::Vector, u::Vector, model::Quadrotor)
  x_dim, u_dim = model.x_dim, model.u_dim
  dimR = Int(model.x_dim/2)
  A = zeros(x_dim,x_dim)

  for i = 1:dimR
    A[i,dimR+i] = 1.0
  end

  return A
end

function B_dyn(x::Vector, u::Vector, model::Quadrotor)
  x_dim, u_dim = model.x_dim, model.u_dim
  dimR = Int(model.x_dim/2)
  B = zeros(x_dim,u_dim)

  for i = 1:dimR
    B[dimR+i,i] = 1.0
  end

  return B
end

# --------------------------------------------