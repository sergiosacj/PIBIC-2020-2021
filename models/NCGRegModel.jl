using NLPModels
using CUTEst
using LinearAlgebra

mutable struct problem
    obj
    grad
    hess
    size
end

function problemTools(nlp)
    function eval_f(x)
        return obj(nlp, x)
    end

    function eval_g(x)
        return grad(nlp, x)
    end

    function eval_h(x)
        return hess_op(nlp, x)
    end

    problem(eval_f, eval_g, eval_h, size(nlp.meta.x0))
end
