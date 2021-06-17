using CUTEst
using NLPModels
include("../helpers/print.jl")
include("../models/ARpRegModel.jl")

function ARp(nlp::AbstractNLPModel;
             p = 2,
             e = 1e-8,
             alpha = 1e-8,
             eta1 = 1000,
             eta2 = 3,
             sigma_low = 1e-8,
             theta = 100,
             J = 20,
             gama1 = 0.5,
             gama2 = 10,
             kMAX = 500)
    sigma_ini = sigma_low
    k = 0

    x = nlp.meta.x0
    s = fill(0.0, size(x))
    gradx = grad(nlp, x)

    file = open("OUTPUTS/output.txt", "w")
    printHeader(file)

    # step 1
    sigma = 0.0
    j = 0

    while k<kMAX
        println("iter      = $(k)")
        println("x         = $(x)")
        println("s         = $(s)")
        println("grad_norm = $(sqrt(sum(gradx.*gradx)))")
        println("infnorm = $(infnorm(gradx))")

        # stop criteria
        if infnorm(gradx) <= e 
            break
        end

        # step 2
        rnlp, problem, solution = solve_subproblem(nlp, sigma, x, p)
        stop_status = solution[2]

        if j == 0 && stop_status != 0
            j = 1
            sigma = sigma_ini
            rnlp, problem, solution = solve_subproblem(nlp, sigma, x, p)
        end

        s = solution[1]
        objective = problem.obj(s)
        gradient = problem.grad(s)

        objx = rnlp.objx
        gradx = rnlp.gradx
        hessx = rnlp.hessx

        # step 3 conditions
        eta1_condition = objx - taylor(s, objx, gradx, hessx)
        eta1_condition /= max(1, abs(objx))

        eta2_condition = norm(s) / max(1, norm(x))

        # step 3 and 4
        if (j >= J || (eta1_condition <= eta1 && eta2_condition <= eta2)) && objective <= objx - alpha*norm(s)^(p+1)
            # step 5
            x = x .+ s
            sigma_ini = gama1 * (sigma == 0.0 ? sigma_ini : sigma)
            # step 1
            sigma = 0.0
            j = 0
            k+=1
        else
            # step 3 and 4 (Otherwise)
            sigma = max(sigma_ini, gama2*sigma)
            j += 1
        end
    end

    # Print results
    objx = obj(nlp, x)
    gradx = grad(nlp, x)
    gradx_norm = sqrt(sum(gradx.*gradx))
    printProblemInfo([objx, gradx_norm, k], file)
    close(file)
end

function taylor(s, objective, gradient, hessian)
    return objective + sum(s.*gradient) + sum(s.*(hessian*s))/2
end

function infnorm(v)
    n = size(v)[1]
    biggest = v[1]
    for i = 2:n
        vi = abs(v[i])
        biggest = biggest < vi ? vi : biggest
    end
    return biggest
end
