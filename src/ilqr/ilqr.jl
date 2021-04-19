"""
ilqr.jl
"""

using TrajectoryOptimization
const TO = TrajectoryOptimization

struct iLQRSolver{IR,Tm,To,Tix,Tiu,Txx,Tuu,Tuud,Tux,Tuxd,Txu,Tx,Tu,TD,TG,T
                  } <: UnconstrainedSolver{T}
    # problem info
    n::Int
    m::Int
    N::Int
    model::Tm
    obj::To
    ix::Tix
    iu::Tiu
    # TODO: mem bottleneck
    # state at each time step: n X N
    X::Vector{Tx}
    X_tmp::Vector{Tx}
    U::Vector{Tu}
    U_tmp::Vector{Tu}
    ts::Vector{T}
    # gains
    # TODO: mem bottleneck
    # state feedback gains (m, n) x N
    K::Vector{Tux}
    K_dense::Tuxd
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
    Quu_dense::Tuud
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
    # misc
    opts::SolverOptions{T}
    stats::SolverStats{T}
    logger::SolverLogger
    solver_name::Symbol
end


function iLQRSolver(prob::Problem{IR,T}, opts::SolverOptions, stats::SolverStats) where {IR,T}
    n, m, N = size(prob)
    n̄ = RobotDynamics.state_diff_size(prob.model)
    M = prob.M
    Md = prob.Md
    V = prob.V
    K = [M(zeros(T, m, n̄)) for k = 1:N-1]
    K_dense = Md(zeros(T, m, n̄))
    d = [V(zeros(T, m)) for k = 1:N-1]
    D = M(zeros(T, n, n + m))
    A = M(zeros(T, n, n))
    B = M(zeros(T, n, m))
    G = M(zeros(T, n̄, m))
    Qxx = M(zeros(T, n, n))
    Qxx_tmp = copy(Qxx)
    Quu = M(zeros(T, m, m))
    Quu_dense = Md(zeros(T, m, m))
    Quu_reg = copy(Quu)
    Qux = M(zeros(T, m, n))
    Qux_tmp = copy(Qux)
    Qux_reg = copy(Qux)
    Qx = V(zeros(T, n))
    Qu = V(zeros(T, m))
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
    Tuud = typeof(Quu_dense)
    Tux = typeof(Qux)
    Tuxd = typeof(K_dense)
    Txu = typeof(B)
    Tx = typeof(Qx)
    Tu = typeof(Qu)
    TD = typeof(D)
    TG = typeof(G)
    solver = iLQRSolver{IR,Tm,To,Tix,Tiu,Txx,Tuu,Tuud,Tux,Tuxd,Txu,Tx,Tu,TD,TG,T}(
        prob.n, prob.m, prob.N, prob.model, prob.obj, prob.ix, prob.iu, prob.X,
        prob.X_tmp, prob.U, prob.U_tmp, prob.ts,
        K, K_dense, d, D, A, B, G, prob.E, Qxx, Qxx_tmp, Quu, Quu_dense,
        Quu_reg, Qux, Qux_tmp, Qux_reg, Qx, Qu, P, P_tmp,
        p, p_tmp, ΔV, ρ, dρ, grad, opts, stats, logger, :iLQR)
    reset!(solver)
    return solver
end

# methods
Base.size(solver::iLQRSolver) = solver.n, solver.m, solver.N

log_level(::iLQRSolver) = InnerLoop

function reset!(solver::iLQRSolver{T}) where T
    # reset regularization
    solver.ρ[1] = 0.0
    solver.dρ[1] = 0.0
    return nothing
end
