export Astrobee

mutable struct Astrobee
    # state: r v p ω
    x_dim
    u_dim

    # dynamics
    f
    A
    B

    # model constants 
    model_radius
    mass
    J
    Jinv

    # constraints / limits
    x_max
    x_min
    u_max
    u_min

    # problem  
    x_init
    x_final
    tf_guess

    # sphere obstacles [(x,y),r]
    obstacles

    # GuSTO Parameters
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

function Astrobee()
    x_dim = 13
    u_dim = 6

    # model constants 
    model_radius = sqrt.(3)*0.5*0.305 # each side of cube is 30.5cm & inflate to sphere
    mass = 7.2
    J_norm = 0.1083
    J    = J_norm*Matrix(1.0I,3,3)
    Jinv = inv(J)

    hard_limit_vel   = 0.4 # Actual limit: 0.5
    hard_limit_accel = 0.1 
    hard_limit_omega = 45*3.14/180 
    hard_limit_alpha = 50*3.14/180 

    # constraints / limits
    x_max = [100.;100.;100.;    hard_limit_vel;hard_limit_vel;hard_limit_vel;   100.;100.;100.;100.;   hard_limit_omega;hard_limit_omega;hard_limit_omega]          
    u_max = [mass*hard_limit_accel;mass*hard_limit_accel;mass*hard_limit_accel;  J_norm*hard_limit_alpha;J_norm*hard_limit_alpha;J_norm*hard_limit_alpha]
    x_min = -x_max
    u_min = -u_max

    # problem 
    x_init  = [-0.25;0.4;0;  0;0;0;  0.;0.;0.; 1.;  0;0;0]
    x_final = [0.7 ;-0.5;0;  0;0;0;  0.;0.;0.; 1.;  0;0;0]


    tf_guess = 110.  # s

    # GuSTO Parameters
    Delta0 = 1000.
    omega0 = 1.
    omegamax = 1.0e9
    epsilon = 1.0e-3
    rho0 = 0.01
    rho1 = 100.
    beta_succ = 2.
    beta_fail = 0.5
    gamma_fail = 5.
    convergence_threshold = 1e-5


    # sphere obstacles [(x,y),r]
    obstacles = []
    obs = [[0.0,0.175,0.], 0.05]
    push!(obstacles, obs)
    obs = [[0.4,-0.10,0.], 0.05]
    push!(obstacles, obs)


    Astrobee(x_dim, u_dim,
             [], [], [],
             model_radius, mass, J, Jinv,
             x_max, x_min, u_max, u_min,
             x_init, x_final, tf_guess,
             obstacles,
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

function get_initial_gusto_parameters(m::Astrobee)
    return m.Delta0, m.omega0, m.omegamax, m.epsilon, m.rho0, m.rho1, m.beta_succ, m.beta_fail, m.gamma_fail, m.convergence_threshold
end

function initialize_trajectory(model::Astrobee, N::Int)
  x_dim,  u_dim   = model.x_dim, model.u_dim
  x_init, x_final = model.x_init, model.x_final
  
  X = hcat(range(x_init, stop=x_final, length=N)...)
  U = zeros(u_dim, N-1)

  return X, U
end

function convergence_metric(model::Astrobee, X, U, Xp, Up)
    N = length(X[1,:])

    # normalized maximum relative error between iterations
    max_num, max_den = -Inf, -Inf
    for k in 1:N
        val = norm(traj.X[:,k]-traj_prev.X[:,k])
        max_num = val > max_num ? val : max_num

        val = norm(traj.X[:,k])
        max_den = val > max_den ? val : max_den
    end
    return max_num/max_den
end


#### CONSTRAINTS
function state_max_convex_constraints(model::Astrobee, X, U, Xp, Up, k, i)
    return (X[i, k] - model.x_max[i])
end
function state_min_convex_constraints(model::Astrobee, X, U, Xp, Up, k, i)
    return (model.x_min[i] - X[i, k])
end
# function state_max_convex_constraints(model::Astrobee, x_ki, k, i)
#     return (x_ki - model.x_max[i])
# end
# function state_min_convex_constraints(model::Astrobee, x_ki, k, i)
#     return (model.x_min[i] - x_ki)
# end
function control_max_convex_constraints(model::Astrobee, X, U, Xp, Up, k, i)
    return (U[i, k] - model.u_max[i])
end
function control_min_convex_constraints(model::Astrobee, X, U, Xp, Up, k, i)
    return (model.u_min[i] - U[i, k])
end

# function state_convex_constraints_penalty(model::Astrobee, X, U, Xp, Up, omega, Delta) 

#     @NLobjective(problem, Min, h_penalty(aux))

# get_constraints(model::Astrobee, X, U, Xp, Up)








#### DYNAMICS

# in continuous time, for all trajectory
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




function f_dyn(x::Vector, u::Vector, model::Astrobee)
  x_dim = model.x_dim
  f = zeros(x_dim)

  r, v, ω = x[1:3], x[4:6], x[11:13]
  qw, qx, qy, qz = x[7:10]
  ωx, ωy, ωz = x[11:13]
  F, M = u[1:3], u[4:6]

  f[1:3] = v
  f[4:6] = F/model.mass

  # SO(3)
  f[7]  = 1/2*(-ωx*qx - ωy*qy - ωz*qz)
  f[8]  = 1/2*( ωx*qw - ωz*qy + ωy*qz)
  f[9]  = 1/2*( ωy*qw + ωz*qx - ωx*qz)
  f[10] = 1/2*( ωz*qw - ωy*qx + ωx*qy)
  f[11:13] = model.Jinv*(M - cross(ω,model.J*ω))

  return f
end

function A_dyn(x::Vector, u::Vector, model::Astrobee)
  x_dim = model.x_dim
  A = zeros(x_dim, x_dim)

  A[1:6,1:6] = kron([0 1; 0 0], Matrix(1.0I,3,3))

  Jxx, Jyy, Jzz = diag(model.J)
  qw, qx, qy, qz = x[7:10] 
  ωx, ωy, ωz = x[11:13]

  A[7,8] = -ωx/2
  A[7,9] = -ωy/2
  A[7,10] = -ωz/2
  A[7,11] = -qx/2
  A[7,12] = -qy/2
  A[7,13] = -qz/2

  A[8,7]  = ωx/2
  A[8,9]  = -ωz/2
  A[8,10] = ωy/2
  A[8,11] = qw/2
  A[8,12] = qz/2
  A[8,13] = -qy/2

  A[9,7]  = ωy/2
  A[9,8]  = ωz/2
  A[9,10] = -ωx/2
  A[9,11] = -qz/2
  A[9,12] = qw/2
  A[9,13] = qx/2

  A[10,7]  = ωz/2
  A[10,8]  = -ωy/2
  A[10,9]  = ωx/2
  A[10,11] = qy/2
  A[10,12] = -qx/2
  A[10,13] = qw/2

  # TODO: Change to account for nondiagonal inertia
  A[11,12] =  (Jyy-Jzz)*ωz/Jxx
  A[11,13] =  (Jyy-Jzz)*ωy/Jxx
  A[12,11] = -(Jxx-Jzz)*ωz/Jyy
  A[12,13] = -(Jxx-Jzz)*ωx/Jyy
  A[13,11] =  (Jxx-Jyy)*ωy/Jzz
  A[13,12] =  (Jxx-Jyy)*ωx/Jzz

  return A
end

function B_dyn(x::Vector, u::Vector, model::Astrobee)
  x_dim, u_dim = model.x_dim, model.u_dim

  B = zeros(x_dim, u_dim)

  B[4:6,1:3] = Matrix(1.0I,3,3)/model.mass
  B[11:13,4:6] = model.Jinv   # SO(3)

  return B
end