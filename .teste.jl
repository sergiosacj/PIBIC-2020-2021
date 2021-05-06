# HS071
# min x1 * x4 * (x1 + x2 + x3) + x3
# st  x1 * x2 * x3 * x4 >= 25
#     x1^2 + x2^2 + x3^2 + x4^2 = 40
#     1 <= x1, x2, x3, x4 <= 5
# Start at (1,5,5,1)
# End at (1.000..., 4.743..., 3.821..., 1.379...)

using Ipopt
using LinearAlgebra

n = 4
x_L = fill(-Inf, n)
x_U = fill(Inf, n)

m = 0
g_L = Float64[]
g_U = Float64[]

function eval_f(x)
    return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
end

function eval_g(x, g)
    g[1] = x[1]   * x[2]   * x[3]   * x[4]
    g[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
    return
end

function eval_grad_f(x, grad_f)
    grad_f[1] = x[1] * x[4] + x[4] * (x[1] + x[2] + x[3])
    grad_f[2] = x[1] * x[4]
    grad_f[3] = x[1] * x[4] + 1
    grad_f[4] = x[1] * (x[1] + x[2] + x[3])
    println("gradf = $(grad_f)")
    println("gradf norm = $(norm(grad_f))")
    return
end

function eval_jac_g(x, mode, rows, cols, values)
end

function eval_h(x, mode, rows, cols, obj_factor, lambda, values)
    if mode == :Structure
        # Symmetric matrix, fill the lower left triangle only
        idx = 1
        for row = 1:4
            for col = 1:row
                rows[idx] = row
                cols[idx] = col
                idx += 1
            end
        end
    else
        # Again, only lower left triangle
        # Objective
        values[1] = obj_factor * (2*x[4])  # 1,1
        values[2] = obj_factor * (  x[4])  # 2,1
        values[3] = 0                      # 2,2
        values[4] = obj_factor * (  x[4])  # 3,1
        values[5] = 0                      # 3,2
        values[6] = 0                      # 3,3
        values[7] = obj_factor * (2*x[1] + x[2] + x[3])  # 4,1
        values[8] = obj_factor * (  x[1])  # 4,2
        values[9] = obj_factor * (  x[1])  # 4,3
        values[10] = 0                     # 4,4
    end
end

prob = createProblem(
    n,
    x_L,
    x_U,
    m,
    g_L,
    g_U,
    0,
    10,
    eval_f,
    eval_g,
    eval_grad_f,
    eval_jac_g,
    eval_h,
)

# Set starting solution
prob.x = [1.0, 5.0, 5.0, 1.0]

# Solve
status = solveProblem(prob)

println(Ipopt.ApplicationReturnStatus[status])
println(prob.x)
println(prob.obj_val)
