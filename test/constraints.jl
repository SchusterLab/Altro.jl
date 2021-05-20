"""
constraints.jl
"""

using Altro
using LinearAlgebra

@inline bound_constraint_eval(x, u, x_max, x_min, u_max, u_min) = ([
    x - x_max;
    x_min - x;
    u - u_max;
    u_min - u;
])

function bound_constraint_jacobian(x, u, x_max, x_min, u_max, u_min)
    n = length(x)
    m = length(u)
    p = 2 * n + 2 * m
    Cx = zeros(p, n)
    for i = 1:n
        Cx[i, i] = 1.
        Cx[i + n, i] = -1.
    end
    Cu = zeros(p, m)
    for i = 1:m
        Cu[i + 2 * n, i] = 1.
        Cu[i + 2 * n + m, i] = -1.
    end
    return Cx, Cu
end

function test_bound_constraint()
    big = 1
    big_ = 1000
    n = 2
    m = 2
    M(x) = x
    V(x) = x
    for i = 1:big
        x_max = rand(n)
        x_min = -rand(n)
        u_max = rand(m)
        u_min = -rand(m)
        bc = BoundConstraint(x_max, x_min, u_max, u_min, n, m, M, V)
        p = 2 * n + 2 * m
        c = zeros(p)
        Cx = zeros(p, n)
        Cu = zeros(p, m)
        for j = 1:big_
            x = 2 .* rand(n) .- 1
            X = [x]
            u = 2 .* rand(m) .- 1
            U = [u]
            k = 1
            Altro.evaluate!(c, bc, X, U, k)
            c_ = bound_constraint_eval(x, u, x_max, x_min, u_max, u_min)
            @assert c ≈ c_
            Altro.jacobian!(Cx, Cu, bc, X, U, k)
            Cx_, Cu_ = bound_constraint_jacobian(x, u, x_max, x_min, u_max, u_min)
            @assert Cx ≈ Cx_
            @assert Cu ≈ Cu_
        end
    end
end
