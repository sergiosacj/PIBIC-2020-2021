using CUTEst
using NLPModels, NLPModelsIpopt
include("RegularizationModel.jl")
using .RegularizationModel

function ARp(nlp::AbstractNLPModel;
             e = 1e-8,
             kMAX = 1e3,
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
    p = 0

    while k<kMAX
        # step 1
        if p >= eta1
            gradient = grad(nlp, x)
            gradientNorm = sqrt(sum(gradient*grandient))
            if gradientNorm < e
                break
            end
        end
        # step 2
        rnlp = RegNLP(nlp, sigma, x)
        println(rnlp)
        stats = ipopt(rnlp)
        println("STATS SOLUTION")
        println(stats)
        s = stats.solution
        # step 3
        objx = obj(nlp, x)
        objxs = obj(nlp, x+s)
        p = objx - objxs
        p /= objx - Taylor(s, objxs, gradient, hess_op(nlp, x))
        if p >= eta1
            x = x+s
        end
        # step 4
        if p >= eta2
            sigma = maximum(sigma_min, gama1*sigma)
        elseif p < eta1
            sigma = gama2*sigma
        end
        k+=1
        finalize(rnlp)
    end
end

function Taylor(s :: Array, obj :: Float64, gradient :: Array, hessian)
    sT = transpose(s)
    return obj + sum(sT*gradient) + sum(sT*(hessian*s)) / 2
end

