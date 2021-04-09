"""
ilqr.jl
"""

using TrajectoryOptimization
const TO = TrajectoryOptimization

struct iLQRSolver{IR,Tm,To,Tix,Tiu,Txx,Tuu,Tux,Txu,Tx,Tu,TD,TG,T
                  } <: UnconstrainedSolver{T}
    model::Tm
    obj::To
    ix::Tix
    iu::Tiu
    X::Vector{Tx}
    X_tmp::Vector{Tx}
    U::Vector{Tu}
    U_tmp::Vector{Tu}
    ts::Vector{T}
    n::Int
    m::Int
    N::Int
    opts::SolverOptions{T}
    stats::SolverStats{T}
    # gains
    # state feedback gains (m, n) x N
    K::Vector{Tux}
    # feedforward gains (m) x N
    d::Vector{Tu}
    # discrete dynamics jacobians
    # block jacobian (n, n + m)
    D::TD
    # w.r.t. state (n, n)
    A::Txx
    # w.r.t. control (n, m)
    B::Txu
    # state difference jacobian (n̄, n)
    G::TG
    # quadratic expansion of obj
    E::TO.QuadraticCost{Txx,Tuu,Tux,Tx,Tu,T}
    Qxx::Txx
    Qxx_tmp::Txx
    Quu::Tuu
    Quu_reg::Tuu
    Qux::Tux
    Qux_tmp::Tux
    Qux_reg::Tux
    Qx::Tx
    Qu::Tu
    # cost-to-go
    P::Txx
    P_tmp::Tux
    p::Tx
    p_tmp::Tu
    ΔV::Vector{T}
    # regularization
    ρ::Vector{T}   
    # regularization rate of change
    dρ::Vector{T}
    # gradient
    grad::Vector{T}
    logger::SolverLogger
end


function iLQRSolver(prob::Problem{IR,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,T},
                    opts::SolverOptions=SolverOptions(), 
                    stats::SolverStats=SolverStats(parent=solvername(iLQRSolver));
                    kwarg_opts...) where {IR,T}
    set_options!(opts; kwarg_opts...)
    n, m, N = size(prob)
    n̄ = RobotDynamics.state_diff_size(prob.model)
    M = prob.M
    V = prob.V
    X_tmp = [V(zeros(T, n)) for k = 1:N+1] # 1 extra
    U_tmp = [V(zeros(T, m)) for k = 1:N+1] # 2 extra
    K = [M(zeros(T, m, n̄)) for k = 1:N-1]
    d = [V(zeros(T, m)) for k = 1:N-1]
    D = M(zeros(T, n, n + m))
    A = M(zeros(T, n, n))
    B = M(zeros(T, n, m))
    G = M(zeros(T, n̄, m))
    Qxx = M(zeros(T, n, n))
    Qxx_tmp = copy(Qxx)
    Quu = M(zeros(T, m, m))
    Quu_reg = copy(Quu)
    Qux = M(zeros(T, m, n))
    Qux_tmp = copy(Qux)
    Qux_reg = copy(Qux)
    Qx = V(zeros(T, n))
    Qu = V(zeros(T, m))
    E = QuadraticCost(copy(Qxx), copy(Quu), copy(Qux), copy(Qx), copy(Qu), 0.; checks=false)
    P = M(zeros(T, n, n))
    P_tmp = M(zeros(T, m, n))
    p = V(zeros(T, n))
    p_tmp = V(zeros(T, m))
    ΔV = zeros(T, 2)
    ρ = zeros(T, 1)
    dρ = zeros(T, 1)
    grad = zeros(T, N-1)
    logger = SolverLogging.default_logger(opts.verbose >= 2)
    Tm = typeof(prob.model)
    To = typeof(prob.obj)
    Tix = typeof(prob.ix)
    Tiu = typeof(prob.iu)
    Txx = typeof(Qxx)
    Tuu = typeof(Quu)
    Tux = typeof(Qux)
    Txu = typeof(B)
    Tx = typeof(Qx)
    Tu = typeof(Qu)
    TD = typeof(D)
    TG = typeof(G)
    solver = iLQRSolver{IR,Tm,To,Tix,Tiu,Txx,Tuu,Tux,Txu,Tx,Tu,TD,TG,T}(
        prob.model, prob.obj, prob.ix, prob.iu, prob.X, X_tmp, prob.U, U_tmp, prob.ts,
        n, m, N, opts, stats, K, d, D, A, B, G, E, Qxx, Qxx_tmp, Quu, Quu_reg, Qux,
        Qux_tmp, Qux_reg, Qx, Qu, P, P_tmp, p, p_tmp, ΔV, ρ, dρ, grad, logger)
    reset!(solver)
    return solver
end

# methods
Base.size(solver::iLQRSolver) = solver.n, solver.m, solver.N
@inline TO.get_objective(solver::iLQRSolver) = solver.obj
@inline TO.get_model(solver::iLQRSolver) = solver.model
@inline get_initial_state(solver::iLQRSolver) = solver.X[1]
@inline TO.integration(solver::iLQRSolver{QUAD}) where {QUAD} = QUAD
solvername(::Type{<:iLQRSolver}) = :iLQR

log_level(::iLQRSolver) = InnerLoop

function reset!(solver::iLQRSolver{T}) where T
    # reset stats
    reset!(solver.stats, solver.opts.iterations, :iLQR)
    # reset regularization
    solver.ρ[1] = 0.0
    solver.dρ[1] = 0.0
    return nothing
end
