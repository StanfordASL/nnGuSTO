export Landing

mutable struct Landing
    # State: r, v, q, ω, m
    x_dim
    # Control: u, Γ
    u_dim

    # Dynamics
    f
    A
    B

    # Model constants
    gravity
    J
    Jinv
    rTB
    α
    ρMin
    ρMax
    δMax
    θMax
    mDry
    γMax
    ωMax

    # Problem
    myInf 
    x_init
    x_final
    tf_guess
    xMin
    xMax

    # State constraints (seen as obstacles)
    obstacles

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

function Landing()
    x_dim = 14
    u_dim = 3

    # Model constants
    gravity = [0 ; 0 ; -3.7114]
    J = zeros(3,3); J[1,1] = 13600.; J[2,2] = 13600.; J[3,3] = 19150.
    Jinv = zeros(3,3); Jinv[1,1] = 1.0/13600.; Jinv[2,2] = 1.0/13600.; Jinv[3,3] = 1.0/19150.
    rTB = [0 ; 0 ; -0.25]
    Isp = 225.
    α = 1.0/(9.806*Isp)
    ρMin = 5200.
    ρMax = 22500.
    δMax = 0.349
    θMax = 0.698
    mDry = 2100.
    γMax = 0.349
    ωMax = 0.2

    # Problem
    myInf = 1.0e4
    x_init  = [150.;75.;433. ; 30.;-5.;-15. ; myInf;myInf;myInf;myInf ; 0;0;0 ; 3250.]
    x_final = [0;0;30. ; 0;0;-1. ; 1.;0;0;0 ; 0;0;0 ; myInf]
    tf_guess = 33. # s
    xMin = [-500.;-500.;-500. ; -myInf;-myInf;-myInf ; -1.0;-1.0;-1.0;-1.0 ; -ωMax;-ωMax;-ωMax ; mDry]
    xMax = [1000.;1000.;1000. ; myInf;myInf;myInf ; 1.0;1.0;1.0;1.0 ; ωMax;ωMax;ωMax ; myInf]

    # Number of state constraints
    obstacles = []
    obs = 0.
    push!(obstacles, obs)
    obs = 0.
    push!(obstacles, obs)

    # GuSTO parameters
    Delta0 = 1.0e6
    omega0 = 1.
    omegamax = 1.0e6
    epsilon = 1.0e-1
    rho0 = 11000.0
    rho1 = 11000.0
    beta_succ = 2.
    beta_fail = 0.5
    gamma_fail = 5.
    convergence_threshold = 1.0

    Landing(x_dim, u_dim,
             [], [], [],
             gravity,J,Jinv,rTB,α,ρMin,ρMax,δMax,θMax,mDry,γMax,ωMax,
             myInf,x_init,x_final,tf_guess,xMin,xMax,
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

function get_initial_gusto_parameters(m::Landing)
    return m.Delta0, m.omega0, m.omegamax, m.epsilon, m.rho0, m.rho1, m.beta_succ, m.beta_fail, m.gamma_fail, m.convergence_threshold
end

function initialize_trajectory(model::Landing, N::Int)
  x_dim,  u_dim   = model.x_dim, model.u_dim
  x_init, x_final = model.x_init, model.x_final
  ρMin, ρMax = model.ρMin, model.ρMax
  dimR  = 3
  
  X = zeros(x_dim, N)
  x0 = x_init[1:dimR]
  x1 = x_final[1:dimR]
  X[1:dimR,:] = hcat(range(x0, stop=x1, length=N)...)
  for i = 1:N
    X[2*dimR+1:2*dimR+1+3,i] = [1. ; 1. ; 1. ; 1.]/norm([1. ; 1. ; 1. ; 1.])
    X[x_dim,i] = x_init[14]
  end
  U = ( (ρMin + ρMax)/2.0 )*ones(u_dim, N-1)/norm(ones(u_dim, N-1))

  return X, U
end

function convergence_metric(model::Landing, X, U, Xp, Up)
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

function true_cost(model::Landing, X, U, Xp, Up)
    x_dim, u_dim = model.x_dim, model.u_dim
    N = length(X[1,:])

    # -m(tf)
    return -X[x_dim,N]

    # ∫ Γ(t) dt
    # cost = 0.0
    # for k = 1:length(U[1,:])
    #     cost += U[u_dim,k]
    # end
    # return cost
end

# --------------------------------------------



# --------------------------------------------
# -               CONSTRAINTS                -
# --------------------------------------------

function control_linear_constraints(model::Landing, X, U, Xp, Up, k, i)
    x_dim, u_dim = model.x_dim, model.u_dim
    ρMin, ρMax, δMax = model.ρMin, model.ρMax, model.δMax
    T1, T2, T3, T = U[1,k], U[2,k], U[3,k], U[1:3,k]
    T1p, T2p, T3p, Tp = Up[1,k], Up[2,k], Up[3,k], Up[1:3,k]

    if i == 1
      constraint_prev = ρMin - norm(Tp)
      grad_prev = -Tp/norm(Tp)

      return ( constraint_prev + sum(grad_prev[i] * ( T[i] - Tp[i] ) for i=1:length(T)) )
    elseif i == 2
      constraint_prev = norm(Tp) - ρMax
      grad_prev = Tp/norm(Tp)

      return ( constraint_prev + sum(grad_prev[i] * ( T[i] - Tp[i] ) for i=1:length(T)) )
    elseif i == 3
      constraint_prev = norm(Tp)*cos(δMax) - T3p
      grad_prev = [T1p*cos(δMax)/norm(Tp) ; T2p*cos(δMax)/norm(Tp) ; T3p*cos(δMax)/norm(Tp) - 1]

      return ( constraint_prev + sum(grad_prev[i] * ( T[i] - Tp[i] ) for i=1:length(T)) )
    # if i == 1
    #   return ρMin - Γ
    # elseif i == 2
    #   return Γ - ρMax
    # elseif i == 3
    #   return Γ*cos(δMax) - T3
    # elseif i == 4
    #   T1p, T2p, T3p, Tp, Γp = Up[1,k], Up[2,k], Up[3,k], Up[1:3,k], Up[u_dim,k]

    #   constraint_prev = norm(Tp)^2 - Γp^2
    #   gradT_prev = 2*Tp
    #   gradΓ_prev = -2*Γp

    #   return ( constraint_prev + sum(gradT_prev[i] * ( T[i] - Tp[i] ) for i=1:length(T)) + gradΓ_prev*( Γ - Γp ) )
    # elseif i == 5
    #   T1p, T2p, T3p, Tp, Γp = Up[1,k], Up[2,k], Up[3,k], Up[1:3,k], Up[u_dim,k]

    #   constraint_prev = norm(Tp)^2 - Γp^2
    #   gradT_prev = 2*Tp
    #   gradΓ_prev = -2*Γp

    #   return -( constraint_prev + sum(gradT_prev[i] * ( T[i] - Tp[i] ) for i=1:length(T)) + gradΓ_prev*( Γ - Γp ) )

    #   return U[1,k] - U[u_dim,k]
    # elseif i == 5
    #   return  -U[u_dim,k] - U[1,k]
    # elseif i == 6
    #   return U[2,k] - U[u_dim,k]
    # elseif i == 7
    #   return  -U[u_dim,k] - U[2,k]
    # elseif i == 8
    #   return U[3,k] - U[u_dim,k]
    # elseif i == 9
    #   return  -U[u_dim,k] - U[3,k]
    else
      println("ERROR - TOO MANY LINEAR CONTROL CONSTRAINTS")
    end
end

function state_max_convex_constraints(model::Landing, X, U, Xp, Up, k, i)
    return ( X[i, k] - model.xMax[i] )
end

function state_min_convex_constraints(model::Landing, X, U, Xp, Up, k, i)
    return ( model.xMin[i] - X[i, k] )
end

function trust_region_max_constraints(model::Landing, X, U, Xp, Up, k, i, Delta)
    return ( (X[i, k] - Xp[i, k]) - Delta )
end
function trust_region_min_constraints(model::Landing, X, U, Xp, Up, k, i, Delta)
    return ( -Delta - (X[i, k] - Xp[i, k]) )
end

function is_in_trust_region(model::Landing, X, U, Xp, Up, Delta)
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

function state_initial_constraints(model::Landing, X, U, Xp, Up)
    x_dim = model.x_dim
    result = []

    for i = 1:x_dim
      if model.x_init[i] < model.myInf
        push!(result, X[i,1] - model.x_init[i])
      end
    end

    return result
end

function state_final_constraints(model::Landing, X, U, Xp, Up)
    x_dim = model.x_dim
    result = []

    for i = 1:x_dim
      if model.x_final[i] < model.myInf
        push!(result, X[i,end] - model.x_final[i])
      end
    end

    return result
end

function obstacle_constraint(model::Landing, X, U, Xp, Up, k, obs_i)
    if obs_i == 1

      θMax = model.θMax
      q3_k, q4_k = X[9,k], X[10,k]
      partial_q = [q3_k ; q4_k]

      dist = norm(partial_q, 2)
      constraint = ( dist - (1. - cos(θMax))/2. )

      return constraint

    else

      γMax = model.γMax
      r2_k, r3_k = X[2,k], X[3,k]
      partial_r = [r2_k ; r3_k]

      dist = norm(partial_r, 2)
      constraint = ( dist - sec(γMax)*r3_k )

      return constraint

    end
end

function obstacle_constraint_convexified(model::Landing, X, U, Xp, Up, k, obs_i)
    if obs_i == 1

      θMax = model.θMax
      q3_k, q4_k = X[9,k], X[10,k]
      partial_q = [q3_k ; q4_k]
      q3_kp, q4_kp = Xp[9,k], Xp[10,k]
      partial_q_p = [q3_kp ; q4_kp]

      dist_prev = norm(partial_q_p, 2)
      dir_prev = partial_q_p/dist_prev
      constraint = ( ( dist_prev - (1. - cos(θMax))/2. ) + sum(dir_prev[i] * (partial_q[i] - partial_q_p[i]) for i=1:length(partial_q)) )

      return constraint

    else

      γMax = model.γMax
      r2_k, r3_k = X[2,k], X[3,k]
      partial_r = [r2_k ; r3_k]
      r2_kp, r3_kp = Xp[2,k], Xp[3,k]
      partial_r_p = [r2_kp ; r3_kp]

      dist_prev = norm(partial_r_p, 2)
      dir_prev = partial_r_p/dist_prev
      constraint = ( ( dist_prev - sec(γMax)*r3_kp ) + sum(dir_prev[i] * (partial_r[i] - partial_r_p[i]) for i=1:length(partial_r)) )

      return constraint

    end
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

function CIB_Func(qw, qx, qy, qz)
  result = zeros(3,3)
  a, b, c, d = qw, qx, qy, qz

  result[1,1] = 2*a^2 + 2*b^2 - 1
  result[1,2] = 2*b*c + 2*a*d
  result[1,3] = 2*b*d - 2*a*c

  result[2,1] = 2*b*c - 2*a*d
  result[2,2] = 2*a^2 + 2*c^2 - 1
  result[2,3] = 2*c*d + 2*a*b

  result[3,1] = 2*b*d + 2*a*c
  result[3,2] = 2*c*d - 2*a*b
  result[3,3] = 2*a^2 + 2*d^2 - 1

  return result[:,:]
end

function dCIB_dqw_Func(qw, qx, qy, qz)
  result = zeros(3,3)
  a, b, c, d = qw, qx, qy, qz

  result[1,1] = 4*a^2
  result[1,2] = 2*d
  result[1,3] = -2*c

  result[2,1] = -2*d
  result[2,2] = 4*a
  result[2,3] = 2*b

  result[3,1] = 2*c
  result[3,2] = -2*b
  result[3,3] = 4*a
  
  return result[:,:]
end

function dCIB_dqx_Func(qw, qx, qy, qz)
  result = zeros(3,3)
  a, b, c, d = qw, qx, qy, qz

  result[1,1] = 4*b
  result[1,2] = 2*c
  result[1,3] = 2*d

  result[2,1] = 2*c
  result[2,2] = 0
  result[2,3] = 2*a

  result[3,1] = 2*d
  result[3,2] = -2*a
  result[3,3] = 0
  
  return result[:,:]
end

function dCIB_dqy_Func(qw, qx, qy, qz)
  result = zeros(3,3)
  a, b, c, d = qw, qx, qy, qz

  result[1,1] = 0
  result[1,2] = 2*b
  result[1,3] = -2*a

  result[2,1] = 2*b
  result[2,2] = 4*c
  result[2,3] = 2*d

  result[3,1] = 2*a
  result[3,2] = 2*d
  result[3,3] = 0
  
  return result[:,:]
end

function dCIB_dqz_Func(qw, qx, qy, qz)
  result = zeros(3,3)
  a, b, c, d = qw, qx, qy, qz

  result[1,1] = 0
  result[1,2] = 2*a
  result[1,3] = 2*b

  result[2,1] = -2*a
  result[2,2] = 0
  result[2,3] = 2*c

  result[3,1] = 2*b
  result[3,2] = 2*c
  result[3,3] = 4*d^2
  
  return result[:,:]
end

function f_dyn(x::Vector, u::Vector, model::Landing)
  x_dim = model.x_dim
  f = zeros(x_dim)
  CIB = zeros(3,3)

  g = model.gravity
  J = model.J
  Jinv = model.Jinv
  rTB = model.rTB
  α = model.α

  r, v, ω, m = x[1:3], x[4:6], x[11:13], x[14]
  qw, qx, qy, qz = x[7:10]
  ωx, ωy, ωz = x[11:13]
  T = u[1:3]

  CIB = CIB_Func(qw, qx, qy, qz)

  f[1:3] = v
  f[4:6] = CIB*T/m + g
  f[11:13] = Jinv*(cross(rTB,T) - cross(ω,model.J*ω))
  f[14] = -α*norm(T)

  # SO(3)
  f[7]  = 1/2*(-ωx*qx - ωy*qy - ωz*qz)
  f[8]  = 1/2*( ωx*qw - ωz*qy + ωy*qz)
  f[9]  = 1/2*( ωy*qw + ωz*qx - ωx*qz)
  f[10] = 1/2*( ωz*qw - ωy*qx + ωx*qy)

  return f[:]
end

function A_dyn(x::Vector, u::Vector, model::Landing)
  x_dim = model.x_dim
  A = zeros(x_dim, x_dim)
  dCIB_dqw = zeros(3,3)
  dCIB_dqx = zeros(3,3)
  dCIB_dqy = zeros(3,3)
  dCIB_dqz = zeros(3,3)

  g = model.gravity
  J = model.J; Jxx = J[1,1]; Jyy = J[2,2]; Jzz = J[3,3];
  Jinv = model.Jinv
  rTB = model.rTB
  α = model.α

  r, v, ω, m = x[1:3], x[4:6], x[11:13], x[14]
  qw, qx, qy, qz = x[7:10]
  ωx, ωy, ωz = x[11:13]
  T = u[1:3]

  dCIB_dqw = dCIB_dqw_Func(qw, qx, qy, qz)
  dCIB_dqx = dCIB_dqx_Func(qw, qx, qy, qz)
  dCIB_dqy = dCIB_dqy_Func(qw, qx, qy, qz)
  dCIB_dqz = dCIB_dqz_Func(qw, qx, qy, qz)

  # Related to r'
  A[1:3,4:6] = Matrix(1.0I,3,3)

  # Related to v'
  A[4:6,7] = dCIB_dqw*T/m
  A[4:6,8] = dCIB_dqx*T/m
  A[4:6,9] = dCIB_dqy*T/m
  A[4:6,10] = dCIB_dqz*T/m
  
  # Related to q'
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

  # Related to ω'
  A[11,12] =  ( Jyy - Jzz )*ωz/Jxx
  A[11,13] =  ( Jyy - Jzz )*ωy/Jxx
  A[12,11] = -( Jxx - Jzz )*ωz/Jyy
  A[12,13] = -( Jxx - Jzz )*ωx/Jyy
  A[13,11] =  ( Jxx - Jyy )*ωy/Jzz
  A[13,12] =  ( Jxx - Jyy )*ωx/Jzz

  # Related to m': nothing to do

  return A[:,:]
end

function B_dyn(x::Vector, u::Vector, model::Landing)
  x_dim, u_dim = model.x_dim, model.u_dim
  B = zeros(x_dim, u_dim)
  CIB = zeros(3,3)

  g = model.gravity
  J = model.J; Jxx = J[1,1]; Jyy = J[2,2]; Jzz = J[3,3];
  Jinv = model.Jinv
  rTB = model.rTB
  α = model.α

  r, v, ω, m = x[1:3], x[4:6], x[11:13], x[14]
  qw, qx, qy, qz = x[7:10]
  ωx, ωy, ωz = x[11:13]
  T = u[1:3]

  CIB = CIB_Func(qw, qx, qy, qz)

  # Related to r': nothing to do

  # Related to v'
  B[4:6,1:3] = CIB/m

  # Related to q': nothing to do

  # Related to ω':
  B[11:13,1] = [0/Jxx ; rTB[3]/Jyy ; -rTB[2]/Jzz]
  B[11:13,2] = [-rTB[3]/Jxx ; 0/Jyy ; rTB[1]/Jzz]
  B[11:13,3] = [rTB[2]/Jxx ; -rTB[1]/Jyy ; 0/Jzz]

  # Related to m':
  B[14,1] = -α*T[1]/norm(T)
  B[14,2] = -α*T[2]/norm(T)
  B[14,3] = -α*T[3]/norm(T)

  return B[:,:]
end

# --------------------------------------------