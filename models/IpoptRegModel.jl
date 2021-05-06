using NLPModels
using LinearAlgebra
using Ipopt
using SparseArrays
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

    function eval_f(s)
        return rnlp.objx + sum(s.*rnlp.gradx) + sum(s.*(rnlp.hessx*s)) / 2 + rnlp.sigma / (rnlp.p+1) * norm(s)^(rnlp.p+1)
    end

    function eval_g(x, g)
    # not necessary
    end

    function eval_grad_f(s::Vector{Float64}, grad_f::Vector{Float64})
        grad_f = rnlp.gradx + rnlp.hessx * s + rnlp.sigma * norm(s)^(rnlp.p-1) .* s
        return
    end

    function eval_jac_g(x, mode, rows, cols, values)
    # not necessary
    end

    function eval_h(
      s::Vector{Float64},         # Current solution
      mode::Symbol,               # Either :Structure or :Values
      rows::Vector{Int32},        # Sparsity structure - row indices
      cols::Vector{Int32},        # Sparsity structure - column indices
      obj_factor::Float64,        # Lagrangian multiplier for objective
      lambda::Vector{Float64},    # Multipliers for each constraint
      values::Vector{Float64},    # The values of the Hessian
    )
        n = rnlp.inner.meta.nvar
        index = 1
        if mode == :Structure
            for i = 1:n
                for j = 1:i
                    rows[index] = i
                    cols[index] = j
                    index = index+1
                end
            end
        else
            for i = 1:n
                for j = 1:i
                    values[index] = rnlp.hessx[i,j] + rnlp.sigma * s[i]*s[j]
                    index = index+1
                end
            end
        end
        return           
    end

    n = rnlp.inner.meta.nvar

    problem = createProblem(
        n,                      # Number of variables
        fill(-Inf, n),          # Variable lower bounds
        fill(Inf, n),           # Variable upper bounds
        0,                      # Number of constraints
        Float64[],              # Constraint lower bounds
        Float64[],              # Constraint upper bounds
        0,                      # Number of non-zeros in Jacobian
        div(((n+1)*n), 2),      # Number of non-zeros in Hessian
        eval_f,                 # Callback: objective function
        eval_g ,                # Callback: constraint evaluation
        eval_grad_f,            # Callback: objective function gradient
        eval_jac_g,             # Callback: Jacobian evaluation
        eval_h,                 # Callback: Hessian evaluation
      )

    # Set starting solution
    nvar = nlp.meta.nvar
    problem.x = fill(0, nvar)

    # finalize(rnlp)

    return rnlp, problem, solveProblem(problem)
end
