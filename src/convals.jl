"""
convals.jl
"""

"""
	ConVal{C,V,M,W}

Holds information about a constraint
"""
struct ConVal{S,C,Tc,Tic}
    con::C
    # constraint function value
    c::Tc
    # dual
    λ::Tc
    # penalty multiplier
    μ::Tc
    # active constraints
    a::Tc
    # index for this conval's `c` in
    # the global linearized constraint
    c_ginds::Tic
end

# constructors
function ConVal(con::AbstractConstraint{S}, n::Int, m::Int, c_ginds::Tic,
                M, V) where {S,Tic}
    p = length(con)
    c = V(zeros(p))
    λ = V(zeros(p))
    μ = V(fill(con.params.μ0, p))
    a = V(zeros(Bool, p))
    Tc = typeof(c)
    C = typeof(con)
    return ConVal{S,C,Tc,Tic}(con, c, λ, μ, a, c_ginds)
end

# methods
function update_active!(conval::ConVal{EQUALITY})
    conval.a .= true
    return nothing
end

function update_active!(conval::ConVal{INEQUALITY})
    a = conval.a
    c = conval.c
    λ = conval.λ
    a_tol = conval.con.params.a_tol
    con = conval.con
    for i = 1:length(a)
        a[i] = ((c[i] >= a_tol) | (abs(λ[i]) > a_tol))
    end
    return nothing
end

@inline violation(con::AbstractConstraint{EQUALITY}, c::AbstractVector) = (
    norm(c, Inf)    
)

@inline violation(con::AbstractConstraint{INEQUALITY}, c::AbstractVector) = (
    max(0., maximum(c))    
)

function update_dual_penalty!(convals::Vector{Vector{ConVal}})
    for (k, convals_) in enumerate(convals)
        for conval in convals_
            # update dual
            λ_max = conval.con.params.λ_max
            λ_min = sense(conval.con) == EQUALITY ? -λ_max : zero(λ_max)
            for i in eachindex(conval.λ)
                conval.λ[i] = clamp(conval.λ[i] + conval.μ[i] * conval.c[i], λ_min, λ_max)
            end
            # update penalty
            for i in eachindex(conval.μ)
                conval.μ[i] = clamp(conval.con.params.ϕ * conval.μ[i], 0, conval.con.params.μ_max)
            end
        end
    end
    return nothing
end

function max_violation_penalty(convals::Vector{Vector{ConVal}})
    max_violation = 0.
    max_penalty = 0.
    for (k, convals_) in enumerate(convals)
        for conval in convals_
            viol = violation(conval.con, conval.c)
            max_violation = max(max_violation, viol)
            max_penalty = max(max_penalty, maximum(conval.μ))
        end
    end
    return max_violation, max_penalty
end

function max_violation_info(convals::Vector{Vector{ConVal}})
    max_viol = -Inf
    info_str = ""
    for (k, convals_) in enumerate(convals)
        for conval in convals_
            max_viol_, info_str_ = max_violation_info(conval.con, conval.c, k)
            if max_viol_ > max_viol
                max_viol = max_viol_
                info_str = info_str_
            end
        end
    end
    return max_viol, info_str
end


# build list of convals from constraint list
function convals_from_constraint_list(cons::ConstraintList)
    # indices for tracking global concatenated constraint
    c_gind = 0
    convals = Vector{ConVal}[]
    # build a list of convals at each knot point
    for k = 1:cons.N
        convals_ = ConVal[]
        # iterate over each constraint
        for i in 1:length(cons)
            con = cons.constraints[i]
            knot_points = cons.inds[i]
            # if this constraint is active at the current knot point,
            # add a conval for it to the current list
            if k in knot_points
                p = length(con)
                c_ginds = cons.V((1:p) .+ c_gind)
                c_gind += p
                conval = ConVal(con, cons.n, cons.m, c_ginds,
                                cons.M, cons.V)
                push!(convals_, conval)
            end
        end
        push!(convals, convals_)
    end
    return convals
end
