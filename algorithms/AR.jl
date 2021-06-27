using CUTEst
using NLPModels
include("../helpers/print.jl")
include("../models/ARpRegModel.jl")

function ARp(nlp::AbstractNLPModel;
             p = 2,
             e = 1e-6,
             alpha = 1e-8,
             eta1 = 1000,
             eta2 = 3,
             sigma_low = 1e-8,
             theta = 100,
             J = 20,
             gama1 = 0.5,
             gama2 = 10,
             kMAX = 500)
    # output variables
    fcnt = gcnt = hcnt = 0
    stop = 1
    allf = zeros(Float64, Integer(kMAX))
    allg = zeros(Float64, Integer(kMAX))
    allsigma = zeros(Float64, Integer(kMAX))

    # essencial variables
    sigma_ini = sigma_low
    k = 1
    x = nlp.meta.x0
    s = fill(0.0, size(x))
    gradx = grad(nlp, x)
    gcnt += 1

    # step 1
    sigma = 0.0
    j = 0

    while k<kMAX
        println("iteration: $(k)")
        allsigma[k] = sigma

        # stop criteria
        if norm(gradx) <= e
            stop = 0
            break
        end

        # step 2
        rnlp, problem, solution = solve_subproblem(nlp, sigma, x, p)
        fcnt += 1
        gcnt += 1
        hcnt += 1
        stop_status = solution[2]

        if j == 0 && stop_status != 0
            j = 1
            sigma = sigma_ini
            rnlp, problem, solution = solve_subproblem(nlp, sigma, x, p)
        end

        s = solution[1]

        objective = problem.obj(s)
        allf[k] = objective
        fcnt += 1

        gradient = problem.grad(s)
        allg[k] = norm(gradient)
        gcnt += 1

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
            x = x + s
            sigma_ini = gama1 * (sigma == 0.0 ? sigma_ini : sigma)
            # step 1
            sigma = 0.0
            j = 0
        else
            # step 3 and 4 (Otherwise)
            sigma = max(sigma_ini, gama2*sigma)
            j += 1
        end

        k+=1
    end
    return stop, [k, fcnt, gcnt, hcnt, allf, allg, allsigma]
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
