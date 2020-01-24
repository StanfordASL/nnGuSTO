export PolygonalObstacle

mutable struct PolygonalObstacle
    c::Vector # center

    # half widths
    dx
    dy
    dz

    # normals
    n_vec::Vector

    # positions of corners
    p_vec::Vector
end
        
# widths_vec: width along each dimension (2*radius)
function PolygonalObstacle(center::Vector, widths_vec::Vector)
  c = center
  dx = widths_vec[1] / 2.
  dy = widths_vec[2] / 2.
  dz = widths_vec[3] / 2.

  p_vec = compute_corners_positions(c[1], c[2], c[3], dx, dy, dz)
  n_vec = compute_surfaces_normals(c[1], c[2], c[3], dx, dy, dz)
  PolygonalObstacle(c, dx, dy, dz,
                     n_vec, p_vec)
end





function compute_surfaces_normals(cx, cy, cz, dx, dy, dz)
  n1 = [ 0; -1;  0]
  n2 = [ 1;  0;  0]
  n3 = [ 0;  1;  0]
  n4 = [-1;  0;  0]
  n5 = [ 0;  0;  1]
  n6 = [ 0;  0; -1]
  return [n1, n2, n3, n4, n5, n6]
end
function get_surfaces_normals(obs::PolygonalObstacle)
  n = obs.n_vec
  return n[1], n[2], n[3], n[4], n[5], n[6]
end


function compute_corners_positions(cx, cy, cz, dx, dy, dz)
  p125 = [cx + dx; cy - dy; cz + dz]
  p235 = [cx + dx; cy + dy; cz + dz]
  p345 = [cx - dx; cy + dy; cz + dz]
  p145 = [cx - dx; cy - dy; cz + dz]
  p126 = [cx + dx; cy - dy; cz - dz]
  p236 = [cx + dx; cy + dy; cz - dz]
  p346 = [cx - dx; cy + dy; cz - dz]
  p146 = [cx - dx; cy - dy; cz - dz]
  return [p125, p235, p345, p145, p126, p236, p346, p146]
end
function get_corners_positions(obs::PolygonalObstacle)
  p = obs.p_vec
  p125, p235, p345, p145, p126, p236, p346, p146 = 
        p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]
  return p125, p235, p345, p145, p126, p236, p346, p146
end


function B_is_in_zone_i(x::Vector, i::Int, obs::PolygonalObstacle)
  c = obs.c
  n = obs.n_vec[i]
  dx, dy, dz = obs.dx, obs.dy, obs.dz
  if i == 1 || i == 3
    return (dot(x-c,n) >= dy)
  elseif i == 2 || i == 4
    return (dot(x-c,n) >= dx)
  elseif i == 5 || i == 6
    return (dot(x-c,n) >= dz)
  else
    error("[polygonal_obstacles.jl::is_in_zone_i] i not recognized.")
  end
end



function signed_distance(x::Vector, obs::PolygonalObstacle)
  # extract values
  c                                              = obs.c
  cx, cy, cz                                     = obs.c[1], obs.c[2], obs.c[3]
  dx, dy, dz                                     = obs.dx,   obs.dy,   obs.dz
  p125, p235, p345, p145, p126, p236, p346, p146 = get_corners_positions(obs)
  n1, n2, n3, n4, n5, n6                         = get_surfaces_normals(obs)


  # check in which zones it is
  Bz_vec = falses(6) # vector of booleans: i-th value is true if x is in i-th zone
  for i = 1:6;  Bz_vec[i] = B_is_in_zone_i(x, i, obs);  end

  # x is outside a corner
  if Bz_vec[1] && Bz_vec[2] && Bz_vec[5] # in 125
    return norm(x-p125)
  elseif Bz_vec[2] && Bz_vec[3] && Bz_vec[5] # in 235
    return norm(x-p235)
  elseif Bz_vec[3] && Bz_vec[4] && Bz_vec[5] # in 345
    return norm(x-p345)
  elseif Bz_vec[1] && Bz_vec[4] && Bz_vec[5] # in 145
    return norm(x-p145)
  elseif Bz_vec[1] && Bz_vec[2] && Bz_vec[6] # in 126
    return norm(x-p126)
  elseif Bz_vec[2] && Bz_vec[3] && Bz_vec[6] # in 236
    return norm(x-p236)
  elseif Bz_vec[3] && Bz_vec[4] && Bz_vec[6] # in 346
    return norm(x-p346)
  elseif Bz_vec[1] && Bz_vec[4] && Bz_vec[6] # in 146
    return norm(x-p146)

  # x is outside an edge
  elseif Bz_vec[1] && Bz_vec[2] # in 12
    xhat = [x[1]; x[2]; cz+dz]; return norm(xhat-p125)
  elseif Bz_vec[2] && Bz_vec[3] # in 23
    xhat = [x[1]; x[2]; cz+dz]; return norm(xhat-p235)
  elseif Bz_vec[3] && Bz_vec[4] # in 34
    xhat = [x[1]; x[2]; cz+dz]; return norm(xhat-p345)
  elseif Bz_vec[1] && Bz_vec[4] # in 14
    xhat = [x[1]; x[2]; cz+dz]; return norm(xhat-p145)
  elseif Bz_vec[1] && Bz_vec[5] # in 15
    xhat = [x[1]+dx; x[2]; x[3]]; return norm(xhat-p125)
  elseif Bz_vec[2] && Bz_vec[5] # in 25
    xhat = [x[1]; x[2]-dy; x[3]]; return norm(xhat-p125)
  elseif Bz_vec[3] && Bz_vec[5] # in 35
    xhat = [x[1]+dx; x[2]; x[3]]; return norm(xhat-p235)
  elseif Bz_vec[4] && Bz_vec[5] # in 45
    xhat = [x[1]; x[2]+dy; x[3]]; return norm(xhat-p345)
  elseif Bz_vec[1] && Bz_vec[6] # in 16
    xhat = [x[1]+dx; x[2]; x[3]]; return norm(xhat-p126)
  elseif Bz_vec[2] && Bz_vec[6] # in 26
    xhat = [x[1]; x[2]-dy; x[3]]; return norm(xhat-p126)
  elseif Bz_vec[3] && Bz_vec[6] # in 36
    xhat = [x[1]+dx; x[2]; x[3]]; return norm(xhat-p236)
  elseif Bz_vec[4] && Bz_vec[6] # in 46
    xhat = [x[1]; x[2]+dy; x[3]]; return norm(xhat-p346)

  # x is outside a flat edge (not on border)
  elseif Bz_vec[1] # in 1
    return (dot(x-c,n1) - dy)
  elseif Bz_vec[2] # in 2
    return (dot(x-c,n2) - dx)
  elseif Bz_vec[3] # in 3
    return (dot(x-c,n3) - dy)
  elseif Bz_vec[4] # in 4
    return (dot(x-c,n4) - dx)
  elseif Bz_vec[5] # in 5
    return (dot(x-c,n5) - dz)
  elseif Bz_vec[6] # in 6
    return (dot(x-c,n6) - dz)

  # is inside
  else
    return (norm(x-c) - sqrt(dx^2+dy^2+dz^2)) # not quite accurate, but good enough for obstacle avoidance
  end 
end



function ∇signed_distance(x::Vector, obs::PolygonalObstacle)

    x_dim = length(x)

    ε = 1.0e-2
    
    xε   = zeros(x_dim)
    x_ε  = zeros(x_dim)
    x2ε  = zeros(x_dim)
    x_2ε = zeros(x_dim)
    dsε   = 0.
    ds_ε  = 0.
    ds2ε  = 0.
    ds_2ε = 0.

    ∇ds = zeros(x_dim)

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

        dsε   = signed_distance(xε  , obs)
        ds_ε  = signed_distance(x_ε , obs)
        ds2ε  = signed_distance(x2ε , obs)
        ds_2ε = signed_distance(x_2ε, obs)

        ∇ds[i] = (-ds2ε + 8*dsε - 8*ds_ε + ds_2ε)/(12*ε)
    end

  return ∇ds
end




####  PLOTTING
function plot_polyObs_corners_2d(fig, obs; dims=[1,2])
    edges = hcat(obs.p_vec...)
    scatter!(fig, edges[dims[1],:],edges[dims[2],:])
end
function plot_polyObs_corners_3d(fig, obs)
    edges = hcat(obs.p_vec...)
    scatter3d!(fig, edges[1,:],edges[2,:],edges[3,:])
end

function plot_distances(x, obs)
    distance = signed_distance(x,obs)
    @show distance
    
    edges = hcat(obs.p_vec...)
    fig = scatter3d(edges[1,:],edges[2,:],edges[3,:])
    fig = scatter3d([x[1],x[1]],[x[2],x[2]],[x[3],x[3]])
    plot_polyObs_corners_3d(fig, obs)
end

# include("astrobee_se3_script.jl")
# x = [0;0;0.]
# plot_distances(x, obs)