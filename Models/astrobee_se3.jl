export Astrobee
using ForwardDiff

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

    # Shooting parameters 
    lambda_min_bound
    lambda_max_bound
    num_constraints
end

function Astrobee()
    x_dim = 13
    u_dim = 6

    # model constants 
    model_radius = sqrt.(3)*0.5*0.305 # each side of cube is 30.5cm & inflate to sphere
    model_radius = 1e-5
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
    x_final = [0.7 ;-0.5;0;  0;0;0;  0.;0.;1.; 0.;  0;0;0]


    tf_guess = 110.  # s

    # GuSTO Parameters
    Delta0 = 1000.
    omega0 = 1.
    omegamax = 1.0e9
    epsilon = 1.0e-3
    rho0 = 0.01
    rho1 = 5.0
    beta_succ = 2.
    beta_fail = 0.1
    gamma_fail = 5.
    convergence_threshold = 1e-3

    # sphere obstacles [(x,y),r]
    obstacles = []
    obs = [[0.0,0.175,0.], 0.1]
    # obs = [[-0.2,0.0,0.], 0.1]
    push!(obstacles, obs)
    obs = [[0.4,-0.10,0.], 0.1]
    push!(obstacles, obs)

    # Shooting parameters 
    num_constraints = 2*x_dim + length(obstacles)
    lambda_min_bound = -1e3 
    lambda_max_bound = 0.0


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
                convergence_threshold,
                lambda_min_bound,
                lambda_max_bound,
                num_constraints)
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
        val = norm(X[:,k]-Xp[:,k])
        max_num = val > max_num ? val : max_num

        val = norm(X[:,k])
        max_den = val > max_den ? val : max_den
    end
    return max_num*100.0/max_den
end





# --------------------------------------------
# -               OBJECTIVE                  -
function true_cost(model::Astrobee, X, U, Xp, Up)
    cost = 0.
    for k = 1:length(U[1,:])
        cost += sum(U[i,k]^2 for i = 1:model.u_dim)
    end
    return cost
end
# --------------------------------------------





# --------------------------------------------
# -               CONSTRAINTS                -

function collect_nonlinear_constraints(model::Astrobee, X)
	x_dim        = model.x_dim
	num_obstacles = length(model.obstacles)
 #    g_dim = 2*x_dim + num_obstacles
 #    g = zeros(g_dim)
	# g[1 : x_dim] = X - model.x_max
 #    g[x_dim+1 : 2*x_dim] = model.x_min - X
 #    for obs_i = 1:num_obstacles
 #      g[2*x_dim + obs_i] = obstacle_constraint_at_a_point(model, X, obs_i)
 #    end
    g_dim = num_obstacles
    g = zeros(g_dim)
    for obs_i = 1:num_obstacles
      g[obs_i] = obstacle_constraint_at_a_point(model, X, obs_i)
    end





	# for i = 1:x_dim
        # g[i] = X[i] - model.x_max[i]			
	# end
	# for i = 1:x_dim
        # g[x_dim + i] = model.x_min[i] - X[i]
	# end
	# for obs_i = 1:num_obstacles
		# g[2*x_dim + obs_i] = obstacle_constraint_at_a_point(model, X, obs_i)
	# end
	return g
end

# function derivative_collect_nonlinear_constraints(model::Astrobee, X)
# 	dg = ForwardDiff.jacobian(z -> collect_nonlinear_constraints(model, z),X)
# 	return dg
# end

function derivative_collect_nonlinear_constraints(model::Astrobee, x)
    #return gradient(x -> Vfunc(x,param),z,GradientConfig(x -> Vfunc(x,param),x,Chunk{1}()),Val{false}())

    x_dim         = model.x_dim
    num_obstacles = length(model.obstacles)
    # g_dim         = 2*x_dim + num_obstacles
    g_dim         = num_obstacles

    ε = 1.0e-3
    
    xε = zeros(x_dim)
    x_ε = zeros(x_dim)
    x2ε = zeros(x_dim)
    x_2ε = zeros(x_dim)
    gε = zeros(g_dim)
    g_ε = zeros(g_dim)
    g2ε = zeros(g_dim)
    g_2ε = zeros(g_dim)

    ∇g = zeros(g_dim, x_dim)

    for i = 1:x_dim
        for j = 1:x_dim
            if j == i
                xε[j] = x[j] + ε
                x_ε[j] = x[j] - ε
                x2ε[j] = x[j] + 2*ε
                x_2ε[j] = x[j] - 2*ε
            else
                xε[j] = x[j]
                x_ε[j] = x[j]
                x2ε[j] = x[j]
                x_2ε[j] = x[j]
            end
        end

        gε = collect_nonlinear_constraints(model, xε)
        g_ε = collect_nonlinear_constraints(model, x_ε)
        g2ε = collect_nonlinear_constraints(model, x2ε)
        g_2ε = collect_nonlinear_constraints(model, x_2ε)

        ∇g[:,i] = (-g2ε + 8*gε - 8*g_ε + g_2ε)/(12*ε)
    end

    return ∇g
end

function state_max_convex_constraints(model::Astrobee, X, U, Xp, Up, k, i)
    return (X[i, k] - model.x_max[i])
end
function state_min_convex_constraints(model::Astrobee, X, U, Xp, Up, k, i)
    return (model.x_min[i] - X[i, k])
end
function control_max_convex_constraints(model::Astrobee, X, U, Xp, Up, k, i)
    return (U[i, k] - model.u_max[i])
end
function control_min_convex_constraints(model::Astrobee, X, U, Xp, Up, k, i)
    return (model.u_min[i] - U[i, k])
end

# function state_convex_constraints_penalty_shooting(model::Astrobee, x::Vector)
#     x_dim, x_max, x_min = model.x_dim, model.x_max, model.x_min

#     pdot = zeros(x_dim)
    
#     for i = 1:x_dim
#         if x[i] > x_max[i]
#             pdot[i] += 1
#         elseif x[i] < x_min[i]
#             pdot[i] += -1
#         end
#     end

#     return pdot
# end





# function trust_region_constraints(model::Astrobee, X, U, Xp, Up, k, Delta)
#     return ( sum(X[i,k]^2 for i = 1:model.x_dim) - Delta )
# end


# function trust_region_sphere_constraints(model::Astrobee, X, U, Xp, Up, k, Delta)
#     return ( sum(X[i,k]^2 for i = 1:model.x_dim) - Delta^2)
# end
# function is_in_trust_region(model::Astrobee, X, U, Xp, Up, Delta)
#     B_is_inside = true

#     for k = 1:length(X[1,:])
#         if trust_region_sphere_constraints(model, X, U, Xp, Up, k, Delta) > 0.
#             B_is_inside = false
#         end
#     end
#     return B_is_inside
# end
function trust_region_max_constraints(model::Astrobee, X, U, Xp, Up, k, i, Delta)
    return ( (X[i, k]-Xp[i, k]) - Delta )
end
function trust_region_min_constraints(model::Astrobee, X, U, Xp, Up, k, i, Delta)
    return ( -Delta - (X[i, k]-Xp[i, k]) )
end
function is_in_trust_region(model::Astrobee, X, U, Xp, Up, Delta)
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


function state_initial_constraints(model::Astrobee, X, U, Xp, Up)
    return (X[:,1] - model.x_init)
end
function state_final_constraints(model::Astrobee, X, U, Xp, Up)
    return (X[:,end] - model.x_final)
end

function obstacle_constraint_at_a_point(model::Astrobee, X, obs_i)
    p_obs, obs_radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]
    bot_radius        = model.model_radius
    total_radius      = obs_radius+ bot_radius

    p_k  = X[1:3]
    
    dist = norm(p_k - p_obs, 2)

    g = total_radius^2 - dist^2
    constraint = g
    return constraint
end

function obstacle_constraint(model::Astrobee, X, U, Xp, Up, k, obs_i)
    p_obs, obs_radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]
    bot_radius        = model.model_radius
    total_radius      = obs_radius+ bot_radius

    p_k  = X[1:3, k]
    
    dist = norm(p_k - p_obs, 2)
    g = total_radius^2 - dist^2
    constraint = g
    return constraint
end

# function obstacle_constraint_convexified(model::Astrobee, X, U, Xp, Up, k, obs_i)
#     p_obs, obs_radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]
#     bot_radius        = model.model_radius
#     total_radius      = obs_radius + bot_radius

#     p_k  = X[1:3, k]
#     p_kp = Xp[1:3, k]
    
#     dist_prev = norm(p_kp - p_obs, 2)
#     n_prev    = (p_kp-p_obs) / dist_prev

#     constraint = - ( dist_prev - total_radius + sum(n_prev[i] * (p_k[i]-p_kp[i]) for i=1:3) )
#     return constraint
# end
function obstacle_constraint_convexified(model::Astrobee, X, U, Xp, Up, k, obs_i)
    p_obs, obs_radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]
    bot_radius        = model.model_radius
    total_radius      = obs_radius + bot_radius

    p_k  = X[1:3, k]
    p_kp = Xp[1:3, k]

    dist_prev = norm(p_kp - p_obs, 2)
    g_kp = total_radius^2 - dist_prev^2 # scalar

    dg_kp = -2*(p_kp - p_obs) # vector

    constraint = g_kp + (dg_kp)' * (p_k - p_kp)
    
    return constraint
end

# function obs_avoidance_penalty_grad_all_shooting(model::Astrobee, x)
#     r_dot = zeros(3)

#     for obs_i = 1:length(model.obstacles)
#         p_obs, obs_radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]
#         bot_radius        = model.model_radius
#         total_radius      = obs_radius + bot_radius

#         p_k  = x[1:3]
    
#         dist = norm(p_k - p_obs, 2)
#         true_dist = dist - total_radius

#         n_hat = (p_k-p_obs) / dist

#         if true_dist < 0
#             r_dot += n_hat
#         end
#     end

#     return r_dot
# end
# --------------------------------------------







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

function f_dyn_shooting(x, u, p, model::Astrobee, X, U, Xp, Up, omega)
    fs = zeros(2*model.x_dim)

    r, v, w             = x[1:3], x[4:6], x[11:13]
    qw, qx, qy, qz      = x[7:10]
    wx, wy, wz          = x[11:13]

    F, M                = u[1:3], u[4:6]

    prx, pry, prz       = p[1:3]
    pqw, pqx, pqy, pqz  = p[7:10]

    # State variables
    fs[1:3] += v                                # rdot
    fs[4:6] += F/model.mass                      # vdot
    fs[7]   += 1/2*(-wx*qx - wy*qy - wz*qz)     # qdot
    fs[8]   += 1/2*( wx*qw - wz*qy + wy*qz)
    fs[9]   += 1/2*( wy*qw + wz*qx - wx*qz)
    fs[10]  += 1/2*( wz*qw - wy*qx + wx*qy)
    fs[11:13] = model.Jinv*(M - cross(w[:],(model.J*w)))    # wdot
	
	  # Dual variables pdot = - p*df/dx + 2*omega*(g-lambda)*(dg/dx) = T1 + T2
    # T1 = - p*df/dx
	  T1 = zeros(model.x_dim)    
    T1[4] += -prx
    T1[5] += -pry
    T1[6] += -prz

    T1[7] += -1/2*( pqx*wx + pqy*wy + pqz*wz)
    T1[8] += -1/2*(-pqw*wx + pqy*wz - pqz*wy)
    T1[9] += -1/2*(-pqw*wy - pqx*wz + pqz*wx)
    T1[10] += -1/2*(-pqw*wz + pqx*wy - pqy*wx)

    T1[11] += -1/2*(-pqw*qx + pqx*qw - pqy*qz + pqz*qy)
    T1[12] += -1/2*(-pqw*qy + pqx*qz + pqy*qw - pqz*qx)
    T1[13] += -1/2*(-pqw*qz - pqx*qy + pqy*qx + pqz*qw)
	
    # T2 = 2*omega*(g-lambda)*(dg/dx)
    T2 = zeros(model.x_dim)
    g = collect_nonlinear_constraints(model, x)
    dg = derivative_collect_nonlinear_constraints(model, x)
    lambda = get_lambda_shooting(model, x)
    # println(size(g-lambda))
    # println(size((g-lambda)'*dg))
    T2 = 2*omega*((g - lambda)'*dg)' # (g-lambda) is vector of size g_dim,1 and dg is matrix of size g_dim,x_dim

    # pdot = T1 + T2
    fs[14:26] = T1 + T2

    # OLD (incorrect?) Dual variables
    # fs[17] += -prx
    # fs[18] += -pry
    # fs[19] += -prz

    # fs[20] += -1/2*( pqx*wx + pqy*wy + pqz*wz)
    # fs[21] += -1/2*(-pqw*wx + pqy*wz - pqz*wy)
    # fs[22] += -1/2*(-pqw*wy - pqx*wz + pqz*wx)
    # fs[23] += -1/2*(-pqw*wz + pqx*wy - pqy*wx)

    # fs[24] += -1/2*(-pqw*qx + pqx*qw - pqy*qz + pqz*qy)
    # fs[25] += -1/2*(-pqw*qy + pqx*qz + pqy*qw - pqz*qx)
    # fs[26] += -1/2*(-pqw*qz - pqx*qy + pqy*qx + pqz*qw)

    return fs
end
# function f_dyn_shooting(x::Vector, u::Vector, p::Vector, model::Astrobee)
#     x_dot = zeros(2*model.x_dim)  
#     J, Jinv, mass = model.J, model.Jinv, model.mass


#     r, v, ω             = x[1:3], x[4:6], x[11:13]
#     qw, qx, qy, qz      = x[7:10]
#     ωx, ωy, ωz          = x[11:13]

#     F, M                = u[1:3], u[4:6]

#     prx, pry, prz       = p[1:3]
#     pqw, pqx, pqy, pqz  = p[7:10]
#     pω = p[11:13]

#   # State variables
#   xdot[1:3] += v                            # rdot
#   xdot[4:6] += 1/mass*F                     # vdot
#   xdot[7]  += 1/2*( ωz*qy - ωy*qz + ωx*qw)  # qdot
#   xdot[8]  += 1/2*(-ωz*qx + ωx*qz + ωy*qw)
#   xdot[9]  += 1/2*( ωy*qx - ωx*qy + ωz*qw)
#   xdot[10] += 1/2*(-ωx*qx - ωy*qy - ωz*qz)
#   xdot[11:13] += Jinv*(M - cross(ω,J*ω))    # ωdot

#   # Dual variables
#   # xdot[14:16] += Zeros(3)
#   xdot[17:19] += -pr
#   xdot[20] += -1/2*(-pqy*ωz + pqz*ωy - pqw*ωx)
#   xdot[21] += -1/2*( pqx*ωz - pqz*ωx - pqw*ωy)
#   xdot[22] += -1/2*(-pqx*ωy + pqy*ωx - pqw*ωz)
#   xdot[23] += -1/2*( pqx*ωx + pqy*ωy + pqz*ωz)

#   a1 = [-J[3,1]*ωy + J[2,1]*ωz
#          2*J[3,1]*ωx + J[3,2]*ωy - J[1,1]*ωz + J[3,3]*ωz
#         -2*J[2,1]*ωx + J[1,1]*ωy - J[2,2]*ωy - J[2,3]*ωz]
#   a2 = [-J[3,1]*ωx - 2*J[3,2]*ωy + J[2,2]*ωz - J[3,3]*ωz
#          J[3,2]*ωx - J[1,2]*ωz
#          J[1,1]*ωx - J[2,2]*ωx + 2*J[1,2]*ωy + J[1,3]*ωz]
#   a3 = [ J[2,1]*ωx + J[2,2]*ωy - J[3,3]*ωy + 2*J[2,3]*ωz
#         -J[1,1]*ωx + J[3,3]*ωx - J[1,2]*ωy - 2*J[1,3]*ωz
#         -J[2,3]*ωx + J[1,3]*ωy]

#   xdot[24] += -dot(pω, Jinv*a1) - 1/2*( pqx*qw + pqy*qz - pqz*qy - pqw*qx)
#   xdot[25] += -dot(pω, Jinv*a2) - 1/2*(-pqx*qz + pqy*qw + pqz*qx - pqw*qy)
#   xdot[26] += -dot(pω, Jinv*a3) - 1/2*( pqx*qy - pqy*qx + pqz*qw - pqw*qz)
#   return x_dot
# end
function get_control_shooting(x::Vector, p::Vector, model::Astrobee)
    pv, pw = p[4:6], p[11:13]

    F = pv / (2. * model.mass)
    M = model.Jinv*pw/2.

    us = vcat(F, M)
    
    return us
end

function get_lambda_shooting(model::Astrobee, x)
	g = collect_nonlinear_constraints(model, x)
	x_dim = model.x_dim
	num_obstacles = length(model.obstacles)
	# g_dim = 2*x_dim + num_obstacles
    g_dim = num_obstacles
	lambda = zeros(g_dim)
    lambda_min, lambda_max = model.lambda_min_bound, model.lambda_max_bound
	
	for i = 1:g_dim
		if g[i] <= lambda_min
			lambda[i] = lambda_min
		elseif g[i] >= lambda_max
			lambda[i] = lambda_max
		else
			lambda[i] = g[i]
		end
	end
    return lambda
end
