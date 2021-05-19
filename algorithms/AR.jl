using CUTEst
using NLPModels
include("../helpers/print.jl")
include("../models/NCGRegModel.jl")

function ARp(nlp::AbstractNLPModel;
             e = 1e-8,
             e_rel = 1e-9,
             kMAX = 500,
             sigma_min = 1e-8,
             theta = 100,
             gama1 = 0.5,
             gama2 = 10,
             J = 20,
             alpha = 1e-8,
             eta1 = 1000,
             eta2 = 3)
    # step 0
    j = 0
    p = 0.0
    sigma = 0.0
    x = nlp.meta.x0
    s = fill(0.0, size(x))
    gradient = grad(nlp, x)
    hessian = hess(nlp, x)
    gradient_norm = sqrt(sum(gradient.*gradient))

    file = open("OUTPUTS/output.txt", "w")
    printHeader(file)

    k = 0
    while k<kMAX
        # step 1
        if p >= eta1
            x = x.+s
            gradient = grad(nlp, x)
            hessian = hess(nlp, x)
            gradient_norm = sqrt(sum(gradient.*gradient))
            if sqrt(sum(gradient_norm)) <= e
                printEach([k, objx, gradient_norm, sigma, p, eta1], file)
                break
            end
        end

        # step 2
        rnlp, problem, solution = solve_subproblem(nlp, sigma, x)
        s = solution[1]

        # step 3
        objx = problem.obj(s-s)
        objxs = problem.obj(x.+s)
        p = objx - objxs
        p /= objx - Taylor(s, objxs, gradient, hessian)

        # step 4
        if p >= eta2
            sigma = maximum(sigma_min, gama1*sigma)
        elseif p < eta1
            sigma = gama2*sigma
        end
        k+=1

        printEach([k, objxs, gradient_norm, sigma, p, eta1], file)
        println("s = $(s)")
        if k%40 == 0
            printHeader(file)
        end
    end

    objx = obj(nlp, x)
    gradx = grad(nlp, x)
    gradx_norm = sqrt(sum(gradx.*gradx))
    printProblemInfo([objx, gradx_norm, k], file)
    close(file)
end

function Taylor(s, obj, gradient, hessian)
    return obj + sum(s.*gradient) + sum(s.*(hessian*s))/ 2
end
