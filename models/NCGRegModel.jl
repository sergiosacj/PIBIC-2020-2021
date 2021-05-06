using NLPModels
using LinearAlgebra
using SparseArrays
include("../algorithms/NewtonCG.jl")
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
end

mutable struct problem
    obj
    grad
    hess
    size
end

function RegNLP(nlp :: AbstractNLPModel, sigma, x, p = 2)
  return RegNLP(nlp,
                obj(nlp, x),
                grad(nlp, x),
                hess(nlp, x),
                sigma,
                p)
end

function solve_subproblem(nlp :: AbstractNLPModel, sigma, x)
    rnlp = RegNLP(nlp, sigma, x)

    function eval_f(s)
        return rnlp.objx + sum(s.*rnlp.gradx) + sum(s.*(rnlp.hessx*s)) / 2 + rnlp.sigma / (rnlp.p+1) * norm(s)^(rnlp.p+1)
    end

    function eval_g(s)
        return rnlp.gradx + rnlp.hessx * s + rnlp.sigma * norm(s)^(rnlp.p-1) .* s
    end

    function eval_h(s)
        return rnlp.hessx + (rnlp.p-1) * rnlp.sigma * s * transpose(s) * norm(s)^(rnlp.p-2)
    end

    problemTools = problem(eval_f, eval_g, eval_h, size(x))

    return rnlp, problemTools, newtoncg(problemTools)
end
