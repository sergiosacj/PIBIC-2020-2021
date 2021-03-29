using NLPModels
using LinearAlgebra
using Ipopt
using LinearOperators
"""
Model
    f(x) + ∇f * s + ∇²f * s² / 2 + sigma/(p+1) * ‖s‖^(p+1)
"""

mutable struct RegNLP <: AbstractNLPModel
    inner :: AbstractNLPModel
    objx
    gradx
    hessx
    sigma
    p
    x
end

function RegNLP(nlp :: AbstractNLPModel, sigma, x, p = 2)
  return RegNLP(nlp,
                obj(nlp, x),
                grad(nlp, x),
                hess(nlp, x),
                sigma,
                p,
                x)
end

function solve_subproblem(nlp :: AbstractNLPModel, sigma, x)

    rnlp = RegNLP(nlp, sigma, x)

    function eval_f(x::Vector{Float64})
        if x == 0
            return rnlp.objx
        end

        return rnlp.objx
                + sum(x.*rnlp.gradx)
                + sum(transpose(x)*(rnlp.hessx*x)) / 2
                + rnlp.sigma / (rnlp.p+1) * norm(x)^(rnlp.p+1)
    end

    function eval_g(x, g)
    # not necessary
    end

    function eval_grad_f(x::Vector{Float64}, grad_f::Vector{Float64})
        grad_f = rnlp.gradx + rnlp.hessx * x + rnlp.sigma * norm(x)^(rnlp.p-1) * x
        return
    end

    function eval_jac_g(x, mode, rows, cols, values)
    # not necessary
    end

    function eval_h(
      x::Vector{Float64},         # Current solution
      mode::Symbol,               # Either :Structure or :Values
      rows::Vector{Int32},        # Sparsity structure - row indices
      cols::Vector{Int32},        # Sparsity structure - column indices
      obj_factor::Float64,        # Lagrangian multiplier for objective
      lambda::Vector{Float64},    # Multipliers for each constraint
      values::Vector{Float64},    # The values of the Hessian
    )
        n = rnlp.inner.meta.nvar
        H = rnlp.hessx + (rnlp.p - 1) * rnlp.sigma * norm(x)^(rnlp.p - 2) * sum(x .* x)
        idx = 1
        for i = 1:n
            for j = 1:n
                rows[idx] = i
                cols[idx] = j
                values[idx] = H[i, j]
                idx = idx+1
            end
        end
        return           
    end

    problem = createProblem(
        rnlp.inner.meta.nvar,   # Number of variables
        rnlp.inner.meta.lvar,   # Variable lower bounds
        rnlp.inner.meta.uvar,   # Variable upper bounds
        rnlp.inner.meta.ncon,   # Number of constraints
        rnlp.inner.meta.lcon,   # Constraint lower bounds
        rnlp.inner.meta.ucon,   # Constraint upper bounds
        rnlp.inner.meta.nnzj,   # Number of non-zeros in Jacobian
        rnlp.inner.meta.nnzh,   # Number of non-zeros in Hessian
        eval_f,                 # Callback: objective function
        eval_g ,                # Callback: constraint evaluation
        eval_grad_f,            # Callback: objective function gradient
        eval_jac_g,             # Callback: Jacobian evaluation
        eval_h,                 # Callback: Hessian evaluation
      )

    # Set starting solution
    problem.x = nlp.meta.x0

    finalize(rnlp)

    return solveProblem(problem), problem
end
