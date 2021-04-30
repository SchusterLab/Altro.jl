"""
pn.jl
"""

"""
$(TYPEDEF)
Projected Newton Solver
Direct method developed by the REx Lab at Stanford University
Achieves machine-level constraint satisfaction by projecting onto the feasible subspace.
It can also take a full Newton step by solving the KKT system.
This solver is to be used exlusively for solutions that are close to the optimal solution.
It is intended to be used as a "solution polishing" method for augmented Lagrangian methods.
"""
struct ProjectedNewtonSolver{T,Tm,To,Tx,Tix,Tu,Tiu,TH,Tg,TE,TD,Td} <: ConstrainedSolver{T}
    # problem info
    n::Int
    m::Int
    N::Int
    model::Tm
    obj::To
    convals::Vector{Vector{ConVal}}
    # trajectory
    X::Vector{Tx}
    X_tmp::Vector{Tx}
    # state indices in global constraint
    x_ginds::Vector{Tix}
    U::Vector{Tu}
    U_tmp::Vector{Tu}
    # control indices in global constraint
    u_ginds::Vector{Tiu}
    # global derivatives
    H::TH
    g::Tg
    E::TE
    # global constraints
    # constraint jacobian
    D::TD
    # constraint values
    d::Td
    # duals
    λ::Td
    # active set
    a::Vector{Bool}
    # misc
    opts::SolverOptions{T}
    stats::SolverStats{T}
    solver_name::Symbol
end

function ProjectedNewtonSolver(prob::Problem{IR,T}, opts::SolverOptions,
                               stats::SolverStats) where {IR,T}
    n, m, N = size(prob)
    NZ = n * N + m * (N - 1)
    M = prob.M
    V = prob.V
    # derivatives
    H = spzeros(NZ, NZ)
    g = zeros(NZ)
    # count number of constraint values
    NP = 0
    for convals_ in prob.convals
        for conval in convals_
            NP += length(conval.con)
        end
    end
    # constraints
    D = spzeros(NP, NZ)
    d = zeros(NP)
    λ = zeros(NP)
    a = zeros(Bool, NP)
    # build x_ginds and u_ginds
    x_ginds = [V((1:n) .+ k * (n + m)) for k = 0:N - 1]
    u_ginds = [V((1:m) .+ (n + k * (n + m))) for k = 0:N - 2]
    # put it all together
    Tm = typeof(prob.model)
    To = typeof(prob.obj)
    Tx = eltype(prob.X)
    Tix = eltype(x_ginds)
    Tu = eltype(prob.U)
    Tiu = eltype(u_ginds)
    TH = typeof(H)
    Tg = typeof(g)
    TE = typeof(prob.E)
    TD = typeof(D)
    Td = typeof(d)
    return ProjectedNewtonSolver{T,Tm,To,Tx,Tix,Tu,Tiu,TH,Tg,TE,TD,Td}(
        n, m, N, prob.model, prob.obj, prob.convals, prob.X, prob.X_tmp, x_ginds,
        prob.U, prob.U_tmp, u_ginds, H, g, prob.E, D, d, λ, a, opts, stats, :ProjectedNewton
    )
end

# methods
Base.size(solver::ProjectedNewtonSolver) = solver.n, solver.m, solver.N
