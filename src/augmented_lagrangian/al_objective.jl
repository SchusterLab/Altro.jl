"""
al_objective.jl
"""

struct ALObjective{O<:Objective} <: TO.AbstractObjective
    obj::O
    convals::Vector{Vector{TO.ConVal}}
end

# constructors
function ALObjective(prob::Problem, opts::SolverOptions)
    O = typeof(prob.obj)
    return ALObjective{O}(prob.obj, prob.convals)
end

# methods
Base.copy(obj::ALObjective{O}) where {O} = ALObjective{O}(copy(obj.obj),
                                                          copy(obj.convals))
@inline Base.length(obj::ALObjective) = length(obj.obj)
@inline RD.state_dim(obj::ALObjective) = RD.state_dim(obj.obj)
@inline RD.control_dim(obj::ALObjective) = RD.control_dim(obj.obj)

# evaluations
function TO.cost(obj::ALObjective, X::AbstractVector, U::AbstractVector, k::Int)
    # compute value for cost functions
    cost_ = TO.cost(obj.obj, X, U, k)
    # compute value for constraints
    for conval in obj.convals[k]
        if conval.con.direct
            continue
        end
        # compute and store (in conval) value for constraint functions
        TO.evaluate!(conval.c, conval.con, X, U, k)
        # determine active constraints
        TO.update_active!(conval.a, conval.con, conval.c, conval.λ, 0.)
        # compute value for augmented lagrangian
        cost_ += conval.λ'conval.c
        # 1//2 * c' * I_μ * c
        for i = 1:length(conval.con)
            cost_ += 0.5 * conval.c[i]^2 * conval.μ[i] * conval.a[i]
        end
    end
    return cost_
end

function TO.cost_derivatives!(E::QuadraticCost, obj::ALObjective, X::AbstractVector,
                              U::AbstractVector, k::Int)
    # compute derivatives for cost functions
    # ASSUMPTION: E is being overwritten in the below function
    TO.cost_derivatives!(E, obj.obj, X, U, k)
    # compute derivatvies for constraints
    for conval in obj.convals[k]
        if conval.con.direct
            continue
        end
        Cx = conval.con.Cx
        Cu = conval.con.Cu
        XP_tmp = conval.con.XP_tmp
        UP_tmp = conval.con.UP_tmp
        p_tmp1 = conval.con.p_tmp[1]
        p_tmp2 = conval.con.p_tmp[2]
        # compute derivatives for constraint function
        # ASSUMPTION: jacobian! has already been called on the constraint
        if !conval.con.const_jac
            TO.jacobian!(Cx, Cu, conval.con, x, u)
        end
        # compute derivatives for augmented lagrangian
        p_tmp1 .= conval.μ
        p_tmp1 .*= conval.a # p_tmp1 Iμ
        p_tmp2 .= conval.c
        p_tmp2 .*= p_tmp1
        p_tmp2 .+= conval.λ # p_tmp2 = Iμ * c + λ
        tIμ = Diagonal(p_tmp1)
        if conval.con.state_expansion
            mul!(XP_tmp, Transpose(Cx), tIμ)
            mul!(E.Q, XP_tmp, Cx, 1., 1.)
            mul!(E.q, Transpose(Cx), p_tmp2, 1., 1.)
        end
        if conval.con.control_expansion
            mul!(UP_tmp, Transpose(Cu), tIμ)
            mul!(E.R, UP_tmp, Cu, 1., 1.)
            mul!(E.r, Transpose(Cu), p_tmp2, 1., 1.)
        end
        if conval.con.coupled_expansion
            mul!(UP_tmp, Transpose(Cu), tIμ)
            mul!(E.H, UP_tmp, Cx, 1., 1.)
        end
    end
    return nothing
end
