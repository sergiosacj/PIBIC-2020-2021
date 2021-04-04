using CUTEst
using NLPModels
include("IpoptRegModel.jl")

function ARp(nlp::AbstractNLPModel;
             e = 1e-8,
             kMAX = 1000,
             sigma_min = 1e-8,
             sigma = sigma_min,
             theta = 100,
             gama1 = 0.5,
             gama2 = 10,
             J = 0,
             eta1 = 10,
             eta2 = 2)

    k = 0
    x = nlp.meta.x0
    gradient = grad(nlp, x)
    hessian = hess_op(nlp, x)
    p = 0

    while k<kMAX
        # step 1
        if p >= eta1
            x = x+s
            gradient = grad(nlp, x)
            if sqrt(sum(gradient.*gradident)) <= e
                break
            end
            hessian = hess_op(nlp, x)
        end
        # step 2
        stats, problem = solve_subproblem(nlp, sigma, x)
        s = problem.x
        # step 3
        objx = obj(nlp, x)
        objxs = obj(nlp, x+s)
        p = objx - objxs
        p /= objx - Taylor(s, objxs, gradient, hessian)
        # step 4
        if p >= eta2
            sigma = maximum(sigma_min, gama1*sigma)
        elseif p < eta1
            sigma = gama2*sigma
        end
        k+=1
        println("K = $(k)")
    end
end

function Taylor(s :: Array, obj :: Float64, gradient :: Array, hessian)
    return obj + sum(s.*gradient) + sum(s.*(hessian*s)) / 2
end

