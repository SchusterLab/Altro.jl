"""
backwardpass.jl
"""

"""
Calculates the optimal feedback gains K,d as well as the 2nd Order approximation of the
Cost-to-Go, using a backward Riccati-style recursion. (non-allocating)
"""
function backwardpass!(solver::iLQRSolver{IR}) where {IR<:QuadratureRule}
    # initialize
    model = solver.model
    stage_cost = solver.obj.stage_cost
    terminal_cost = solver.obj.terminal_cost
    ix = solver.ix
    iu = solver.iu
    X = solver.X
    U = solver.U
    ts = solver.ts
    N = solver.N
    K = solver.K
    d = solver.d
    D = solver.D
    A = solver.A
    B = solver.B
    G = solver.G
    E = solver.E
    Qxx = solver.Qxx
    Quu = solver.Quu
    Quu_reg = solver.Quu_reg
    Qux = solver.Qux
    Qux_reg = solver.Qux_reg
    Qx = solver.Qx
    Qu = solver.Qu
    P = solver.P
    P_ = solver.P_
    p = solver.p
    p_ = solver.p_
    ΔV = solver.ΔV
    J = 0

    # compute cost
    for k = 1:N-1
        # cost
        J += TO.stage_cost(stage_cost, X[k], U[k])
    end
    J += TO.stage_cost(terminal_cost, X[N])

    # terminal (cost and action-value) expansions
    TO.gradient!(E, terminal_cost, X[N])
    TO.hessian!(E, terminal_cost, X[N])
    P .= P_N = E.Q
    p .= p_N = E.q

    k = N-1
    while k > 0
	# dynamics and cost expansions
        dt = ts[k + 1] - ts[k]
	RobotDynamics.discrete_jacobian!(D, A, B, IR, model, X[k], U[k], ts[k], dt, ix, iu)
        TO.gradient!(E, stage_cost, X[k], U[k])
        TO.hessian!(E, stage_cost, X[k], U[k])

	# action-value expansion
        Qxx .= E.Q + A' * P * A
        Quu .= E.R + B' * P * B
        Qux .= E.H + B' * P * A
        Qx .= E.q + A' * p
        Qu .= E.r + B' * p

	# regularization
	Quu_reg .= Quu + solver.ρ[1] * I
	Qux_reg .= Qux

	if solver.opts.bp_reg
	    vals = eigvals(Hermitian(Quu_reg))
	    if minimum(vals) <= 0
	        @warn "Backward pass regularized"
                regularization_update!(solver, :increase)
                k = N-1
                ΔV1 = 0
                ΔV2 = 0
                P .= P_N
                p .= p_N
                continue
            end
        end

        # gains
        _calc_gains!(K[k], d[k], Quu_reg, Qux_reg, Qu)
        
	# cost-to-go (using unregularized Quu and Qux)
	_calc_ctg!(ΔV, P, P_, p, p_, K[k], d[k], Qxx, Quu, Qux, Qx, Qu)

        k -= 1
    end

    regularization_update!(solver, :decrease)
    
    return nothing
end


function static_backwardpass!(solver::iLQRSolver{T,QUAD,L,O,n,n̄,m}) where {T,QUAD<:QuadratureRule,L,O,n,n̄,m}
	N = solver.N

    # Objective
    obj = solver.obj
    model = solver.model

    # Extract variables
    Z = solver.Z; K = solver.K; d = solver.d;
    G = solver.G
    S = solver.S
    Quu_reg = SMatrix(solver.Quu_reg)
    Qux_reg = SMatrix(solver.Qux_reg)

    # Terminal cost-to-go
	# Q = error_expansion(solver.Q[N], model)
	Q = solver.Q[N]
	Sxx = SMatrix(Q.Q)
	Sx = SVector(Q.q)

    # Initialize expected change in cost-to-go
    ΔV = @SVector zeros(2)

	k = N-1
    while k > 0
        ix = Z[k]._x
        iu = Z[k]._u

		# Get error state expanions
		fdx,fdu = TO.error_expansion(solver.D[k], model)
		fdx,fdu = SMatrix(fdx), SMatrix(fdu)
		Q = TO.static_expansion(solver.Q[k])
		# Q = error_expansion(solver.Q[k], model)
		# Q = solver.Q[k]

		# Calculate action-value expansion
		Q = _calc_Q!(Q, Sxx, Sx, fdx, fdu)

		# Regularization
		Quu_reg, Qux_reg = _bp_reg!(Q, fdx, fdu, solver.ρ[1], solver.opts.bp_reg_type)

	    if solver.opts.bp_reg
	        vals = eigvals(Hermitian(Quu_reg))
	        if minimum(vals) <= 0
	            @warn "Backward pass regularized"
	            regularization_update!(solver, :increase)
	            k = N-1
	            ΔV = @SVector zeros(2)
	            continue
	        end
	    end

        # Compute gains
		K_, d_ = _calc_gains!(K[k], d[k], Quu_reg, Qux_reg, Q.u)

		# Calculate cost-to-go (using unregularized Quu and Qux)
		Sxx, Sx, ΔV_ = _calc_ctg!(Q, K_, d_)
		# k >= N-2 && println(diag(Sxx))
		if solver.opts.save_S
			S[k].xx .= Sxx
			S[k].x .= Sx
			S[k].c .= ΔV_
		end
		ΔV += ΔV_
        k -= 1
    end

    regularization_update!(solver, :decrease)

    return ΔV
end

function _bp_reg!(Quu_reg::SizedMatrix{m,m}, Qux_reg, Q, fdx, fdu, ρ, ver=:control) where {m}
    if ver == :state
        Quu_reg .= Q.uu #+ solver.ρ[1]*fdu'fdu
		mul!(Quu_reg, Transpose(fdu), fdu, ρ, 1.0)
        Qux_reg .= Q.ux #+ solver.ρ[1]*fdu'fdx
		mul!(Qux_reg, fdu', fdx, ρ, 1.0)
    elseif ver == :control
        Quu_reg .= Q.uu #+ solver.ρ[1]*I
		Quu_reg .+= ρ*Diagonal(@SVector ones(m))
        Qux_reg .= Q.ux
    end
end

function _bp_reg!(Q, fdx, fdu, ρ, ver=:control)
    if ver == :state
		Quu_reg = Q.uu + ρ * fdu'fdu
		Qux_reg = Q.ux + ρ * fdu'fdx
    elseif ver == :control
		Quu_reg = Q.uu + ρ * I
        Qux_reg = Q.ux
    end

	Quu_reg, Qux_reg
end

# function _calc_Q!(Q::TO.StaticExpansion, Sxx, Sx, fdx::SMatrix, fdu::SMatrix)
# 	Qx = Q.x + fdx'Sx
# 	Qu = Q.u + fdu'Sx
# 	Qxx = Q.xx + fdx'Sxx*fdx
# 	Quu = Q.uu + fdu'Sxx*fdu
# 	Qux = Q.ux + fdu'Sxx*fdx
# 	TO.StaticExpansion(Qx,Qxx,Qu,Quu,Qux)
# end


function _calc_gains!(K::AbstractMatrix, d::AbstractVector, Quu::AbstractMatrix,
                      Qux::AbstractMatrix, Qu::AbstractVector)
    LAPACK.potrf!('U', Quu)
    K .= Qux
    d .= Qu
    LAPACK.potrs!('U', Quu, K)
    LAPACK.potrs!('U', Quu, d)
    K .*= -1
    d .*= -1
end


# function _calc_gains!(K, d, Quu::SMatrix, Qux::SMatrix, Qu::SVector)
# 	K_ = -Quu\Qux
# 	d_ = -Quu\Qu
# 	K .= K_
# 	d .= d_
# 	return K_,d_
# end


function _calc_ctg!(ΔV, P, P_, p, p_, K, d, Qxx, Quu, Qux, Qx, Qu)
    # p = Qx + K' * Quu * d +K' * Qu + Qxu * d
    p .= Qx
    mul!(p_, Quu, d)
    mul!(p, K', p_, 1.0, 1.0)
    mul!(p, K', Qu, 1.0, 1.0)
    mul!(p, Qux', d, 1.0, 1.0)

    # P = Qxx + K' * Quu * K + K' * Qux + Qxu * K
    P .= Qxx
    mul!(P_, Quu, K)
    mul!(P, K', P_, 1.0, 1.0)
    mul!(P, K', Qux, 1.0, 1.0)
    mul!(P, Qux', K, 1.0, 1.0)
    transpose!(Qxx, P)
    P .+= Qxx
    P .*= 0.5

    # calculated change is cost-to-go over entire trajectory
    t1 = dot(d, Qu)
    mul!(Qu, Quu, d)
    t2 = 0.5 * dot(d, Qu)
    ΔV[1] += t1
    ΔV[2] += t2

    return nothing
end


# function _calc_ctg!(Q::TO.StaticExpansion, K::SMatrix, d::SVector)
# 	Sx = Q.x + K'Q.uu*d + K'Q.u + Q.ux'd
# 	Sxx = Q.xx + K'Q.uu*K + K'Q.ux + Q.ux'K
# 	Sxx = 0.5*(Sxx + Sxx')
# 	# S.x .= Sx
# 	# S.xx .= Sxx
# 	t1 = d'Q.u
# 	t2 = 0.5*d'Q.uu*d
# 	return Sxx, Sx, @SVector [t1, t2]
# end
