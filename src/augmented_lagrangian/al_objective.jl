"""
al_objective.jl
"""

struct ALObjective{O<:Objective} <: AbstractObjective
    obj::O
    convals::Vector{Vector{ConVal}}
end

# constructors
function ALObjective(prob::Problem, opts::SolverOptions)
    obj = prob.obj
    cons = prob.constraints
    n, m, N = size(prob)
    M = prob.M
    V = prob.V
    convals = Vector{ConVal}[]
    for k = 1:N
        convals_ = ConVal[]
        for i in 1:length(cons)
            con = cons.constraints[i]
            knot_points = cons.inds[i]
            if k in knot_points
                conval = ConVal(con, n, m, M, V, opts.penalty_scaling, opts.penalty_initial,
                                opts.penalty_max, opts.dual_max)
                insert!(convals_, length(convals_) + 1, conval)
            end
        end
        insert!(convals, length(convals) + 1, convals_)
    end
    O = typeof(obj)
    return ALObjective{O}(obj, convals)
end

# methods
Base.copy(obj::ALObjective{O}) where {O} = ALObjective{O,C}(copy(obj.obj),
                                                            copy(obj.convals))
@inline Base.length(obj::ALObjective) = length(obj.obj)
@inline RD.state_dim(obj::ALObjective) = RD.state_dim(obj.obj)
@inline RD.control_dim(obj::ALObjective) = RD.control_dim(obj.obj)

# evaluations
const _U_NULL = Array{Float64,1}(undef, 0)

function _cost!(obj::ALObjective, k::Int, x::AbstractVector, u::AbstractVector)
    J = 0.
    # compute value for constraints
    for conval in obj.convals[k]
        # compute and store (in conval) value for constraint functions
        TO.evaluate!(conval.c, conval.con, x, u)
        # determine active constraints
        if TO.sense(conval.con) isa Inequality
            for i = 1:length(conval.con)
                conval.a[i] = ((conval.c[i] >= 0.) | (conval.λ[i] > 0.))
            end
        end
        # compute value for augmented lagrangian
        J += conval.λ'conval.c
        # 1//2 * c' * I_μ * c
        for i = 1:length(conval.con)
            J += 0.5 * conval.c[i]^2 * conval.μ[i] * conval.a[i]
        end
    end
    return J
end

function TO.cost(obj::ALObjective, k::Int, x::AbstractVector)
    # compute value for cost functions
    J = TO.cost(obj.obj, k, x)
    # compute value for constraints
    return J + _cost!(obj, k, x, _U_NULL)
end

function TO.cost(obj::ALObjective, k::Int, x::AbstractVector, u::AbstractVector)
    # compute value for cost functions
    J = TO.cost(obj.obj, k, x, u)
    # compute value for constraints
    return J + _cost!(obj, k, x, u)
end

function _cost_derivatives!(E::QuadraticCost, obj::ALObjective, k::Int, x::AbstractVector,
                            u::AbstractVector)
    # compute derivatvies for constraints
    for conval in obj.convals[k]
        Cx = conval.con.Cx
        Cu = conval.con.Cu
        XP_tmp = conval.con.XP_tmp
        UP_tmp = conval.con.UP_tmp
        p_tmp1 = conval.con.p_tmp[1]
        p_tmp2 = conval.con.p_tmp[2]
        # compute derivatives for constraint function
        # assumes that if the constraint jacobian is constant
        # that it has already been computed
        if !conval.con.const_jac
            TO.jacobian!(Cx, Cu, conval.con, x, u)
        end
        # compute derivatives for augmented lagrangian
        p_tmp1 .= conval.μ
        p_tmp1 .*= conval.a
        p_tmp2 .= conval.c
        p_tmp2 .*= p_tmp1
        p_tmp2 .+= conval.λ # p_tmp2 = Iμ * c + λ
        Iμ = Diagonal(p_tmp1)
        if conval.con.state_expansion
            mul!(XP_tmp, Transpose(Cx), Iμ)
            mul!(E.Q, XP_tmp, Cx, 1., 1.)
            mul!(E.q, Transpose(Cx), p_tmp2, 1., 1.)
        end
        if conval.con.control_expansion
            mul!(UP_tmp, Transpose(Cu), Iμ)
            mul!(E.R, UP_tmp, Cu, 1., 1.)
            mul!(E.r, Trasponse(Cu), p_tmp2, 1., 1.)
        end
        if conval.con.coupled_expansion
            mul!(UP_tmp, Transpose(Cu), Iμ)
            mul!(E.H, UP_tmp, Cx, 1., 1.)
        end
    end
end

function TO.cost_derivatives!(E::QuadraticCost, obj::ALObjective, k::Int, x::AbstractVector,
                              u::AbstractVector)
    # compute derivatives for cost functions
    TO.cost_derivatives!(E, obj.obj, k, x, u)
    # compute derivatvies for constraints
    _cost_derivatives!(E, obj, k, x, u)
    return nothing
end

function TO.cost_derivatives!(E::QuadraticCost, obj::ALObjective, k::Int, x::AbstractVector)
    # compute derivatives for cost functions
    TO.cost_derivatives!(E, obj.obj, k, x)
    # compute derivatvies for constraints
    _cost_derivatives!(E, obj, k, x, _U_NULL)
    return nothing
end
