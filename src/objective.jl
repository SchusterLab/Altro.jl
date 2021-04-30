"""
objective.jl
"""

############################################################################################
#                              OBJECTIVES                                                  #
############################################################################################

abstract type AbstractObjective end
state_dim(obj::AbstractObjective) = throw(ErrorException("state_dim not implemented"))
control_dim(obj::AbstractObjective) = throw(ErrorException("control_dim not implemented"))

"""$(TYPEDEF)
Objective: stores stage cost(s) and terminal cost functions

Constructors:
```julia
Objective(cost, cost_term, N)
```
"""
struct Objective{Tc} <: AbstractObjective
    cost::Vector{Tc}
    N::Int
end

# constructors
function Objective(cost::Vector{Tc}, N::Int, checks=true) where {
    Tc<:CostFunction}
    if checks
        @assert length(cost) == N
        for k = 1:N-1
            @assert cost[k].terminal == false
        end
        @assert cost[N].terminal == true
    end
    return Objective{Tc}(cost, N)
end

# methods
Base.copy(obj::Objective) = Objective(copy(obj.cost), obj.N)
Base.show(io::IO, obj::Objective) = print(io,"Objective")

@inline control_dim(obj::Objective) = control_dim(obj.cost[1])
@inline state_dim(obj::Objective) = state_dim(obj.cost[1])

@inline cost(obj::Objective, X::AbstractVector, U::AbstractVector, k::Int) = (
    cost(obj.cost[k], X, U, k)
)

@inline cost_derivatives!(E::QuadraticCost, obj::Objective, X::AbstractVector,
                          U::AbstractVector, k::Int) = (
                              cost_derivatives!(E, obj.cost[k], X, U, k)
)

# LQR objective
function LQRObjective(Q::AbstractMatrix, Qf::AbstractMatrix, R::AbstractMatrix,
                      xf::AbstractVector, n::Int, m::Int, N::Int, M, V)
    stage = LQRCost(Q, xf, R, M, V)
    terminal = LQRCost(Qf, xf, R, M, V; use_R=false, terminal=true)
    Tc = typeof(stage)
    cost = Vector{Tc}(undef, N)
    for k = 1:N-1
        cost[k] = stage
    end
    cost[N] = terminal
    return Objective(cost, N)
end
