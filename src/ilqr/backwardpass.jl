"""
backwardpass.jl
"""

"""
Calculates the optimal feedback gains K,d as well as the 2nd Order approximation of the
Cost-to-Go, using a backward Riccati-style recursion. (non-allocating)
"""
function backwardpass!(solver::iLQRSolver{IR}) where {IR}
    # initialize
    model = solver.model
    ix = solver.ix
    iu = solver.iu
    X = solver.X
    U = solver.U
    ts = solver.ts
    m = solver.m
    N = solver.N
    K = solver.K
    K_dense = solver.K_dense
    d = solver.d
    D = solver.D
    A = solver.A
    B = solver.B
    G = solver.G
    E = solver.E
    Qxx = solver.Qxx
    Qxx_tmp = solver.Qxx_tmp
    Quu = solver.Quu
    Quu_dense = solver.Quu_dense
    Quu_reg = solver.Quu_reg
    Qux = solver.Qux
    Qux_tmp = solver.Qux_tmp
    Qux_reg = solver.Qux_reg
    Qx = solver.Qx
    Qu = solver.Qu
    P = solver.P
    P_tmp = solver.P_tmp
    p = solver.p
    p_tmp = solver.p_tmp
    ΔV = solver.ΔV

    # terminal (cost and action-value) expansions
    ΔV .= 0
    TO.cost_derivatives!(E, solver.obj, X, U, N)
    P .= E.Q
    p .= E.q

    k = N-1
    while k > 0
	# dynamics and cost expansions
        dt = ts[k + 1] - ts[k]
	RD.discrete_jacobian!(D, A, B, IR, model, X[k], U[k], ts[k], dt, ix, iu)
        TO.cost_derivatives!(E, solver.obj, X, U, k)

	# action-value expansion
        _calc_Q!(Qxx, Qxx_tmp, Quu, Qux, Qux_tmp, Qx, Qu, E, A, B, P, p)

	# regularization
        reg_flag = _bp_reg!(Quu, Quu_reg, Qux, Qux_reg, A, B, solver.ρ[1], solver.opts.bp_reg_type)
        if solver.opts.bp_reg && reg_flag
            @warn "Backward pass regularized"
            println("bp regularized")
            regularization_update!(solver, :increase)
            k = N-1
            ΔV .= 0
            TO.cost_derivatives!(E, solver.obj, N, X[N])
            P .= E.Q
            p .= E.q
            continue
        end

        # gains
        _calc_gains!(K[k], K_dense, d[k], Quu_reg, Quu_dense, Qux_reg, Qu)
        
	# cost-to-go (using unregularized Quu and Qux)
	_calc_ctg!(ΔV, P, P_tmp, p, p_tmp, K[k], d[k], Qxx, Quu, Qux, Qx, Qu)

        k -= 1
    end

    regularization_update!(solver, :decrease)
    
    return nothing
end

function _bp_reg!(Quu, Quu_reg, Qux, Qux_reg, A, B, ρ, type_)
    reg_flag = false
    # perform regularization
    Quu_reg .= Quu
    for i = 1:size(Quu_reg, 1)
        Quu_reg[i, i] += ρ
    end
    mul!(Qux_reg, Transpose(B), A)
    for i in eachindex(Qux_reg)
        Qux_reg[i] = Qux[i] + ρ * Qux_reg[i]
    end
    # check for ill-conditioning
    vals = eigvals(Hermitian(Quu_reg))
    if minimum(vals) <= 0
        reg_flag = true
    end
    return reg_flag
end

function _calc_Q!(Qxx::AbstractMatrix, Qxx_tmp::AbstractMatrix, Quu::AbstractMatrix,
                  Qux::AbstractMatrix, Qux_tmp::AbstractMatrix, Qx::AbstractVector,
                  Qu::AbstractVector, E::TO.QuadraticCost, A::AbstractMatrix, B::AbstractMatrix,
                  P::AbstractMatrix, p::AbstractVector)
    # Qxx
    mul!(Qxx_tmp, Transpose(A), P)
    mul!(Qxx, Qxx_tmp, A)
    Qxx .+= E.Q
    # Quu
    mul!(Qux_tmp, Transpose(B), P)
    mul!(Quu, Qux_tmp, B)
    Quu .+= E.R
    # Qux
    mul!(Qux_tmp, Transpose(B), P)
    mul!(Qux, Qux_tmp, A)
    Qux .+= E.H
    # Qx
    mul!(Qx, Transpose(A), p)
    Qx .+= E.q
    # Qu
    mul!(Qu, Transpose(B), p)
    Qu .+= E.r
    return nothing
end

function _calc_gains!(K::AbstractMatrix, K_dense::AbstractMatrix,
                      d::AbstractVector, Quu::AbstractMatrix,
                      Quu_dense::AbstractMatrix,
                      Qux::AbstractMatrix, Qu::AbstractVector)
    # compute cholesky decomp of Quu
    Quu_dense .= Quu
    LAPACK.potrf!('U', Quu_dense)
    # compute K
    K_dense .= Qux
    LAPACK.potrs!('U', Quu_dense, K_dense)
    K .= K_dense
    K .*= -1
    # compute d
    d .= Qu
    LAPACK.potrs!('U', Quu_dense, d)
    d .*= -1
    return nothing
end

function _calc_ctg!(ΔV, P, P_, p, p_, K, d, Qxx, Quu, Qux, Qx, Qu)
    # p = Qx + K' * Quu * d + K' * Qu + Qxu * d
    p .= Qx
    mul!(p_, Quu, d)
    mul!(p, Transpose(K), p_, 1.0, 1.0)
    mul!(p, Transpose(K), Qu, 1.0, 1.0)
    mul!(p, Transpose(Qux), d, 1.0, 1.0)

    # P = Qxx + K' * Quu * K + K' * Qux + Qxu * K
    P .= Qxx
    mul!(P_, Quu, K)
    mul!(P, Transpose(K), P_, 1.0, 1.0)
    mul!(P, Transpose(K), Qux, 1.0, 1.0)
    mul!(P, Transpose(Qux), K, 1.0, 1.0)
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
