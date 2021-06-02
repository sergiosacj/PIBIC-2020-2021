using CUTEst
using NLPModels
include("../helpers/print.jl")
include("../models/NCGRegModel.jl")

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
    x = nlp.meta.x0
    sigma_ini = sigma_low
    k = 0
    s = fill(0.0, size(x))
    gradient = grad(nlp, x)

    file = open("OUTPUTS/output.txt", "w")
    printHeader(file)

    # step 1
    j = 0
    sigma = 0.0

    while k<kMAX
        println("iter      = $(k)")
        println("x         = $(x)")
        println("s         = $(s)")
        println("grad_norm = $(sqrt(sum(gradient.*gradient)))")

        # stop criteria
        if p >= eta1
            grad_norm = sqrt(sum(gradient.*gradient))
            if grad_norm <= e
                printEach([k, objx, grad_norm, sigma, p, eta1], file)
                break
            end
        end

        # step 2
        rnlp, problem, solution = solve_subproblem(nlp, sigma, x)
        stop_status = solution[2]

        if j == 0 && stop_status != 0
            j = 1
            sigma = sigma_ini
            rnlp, problem, solution = solve_subproblem(nlp, sigma, x)
        end

        s = solution[1]
        objective = problem.obj(s)
        gradient = problem.grad(s)

        # step 3 conditions
        eta1_condition = rnlp.objx - taylor(s, rnlp.objx, rnlp.gradx, rnlp.hessx)
        eta1_condition /= max(1, abs(rnlp.objx))

        eta2_condition = norm(s) / max(1, norm(x))

        # step 3 and 4
        if (j >= J || (eta1_condition <= eta1 && eta2_condition <= eta2)) && objective <= rnlp.objx - alpha*norm(s)^(p+1)
            # step 5
            x = x .+ s
            sigma_ini = gama1 * (sigma == 0.0 ? sigma_ini : sigma)
        else
            sigma = max(sigma_ini, gama2*sigma)
            j += 1
        end

        k+=1
    end

    # Print results
    objx = obj(nlp, x)
    gradx = grad(nlp, x)
    gradx_norm = sqrt(sum(gradx.*gradx))
    printProblemInfo([objx, gradx_norm, k], file)
    close(file)
end

function taylor(s, objective, gradient, hessian)
    return objective + sum(s.*gradient) + sum(s.*(hessian*s))/ 2
end
