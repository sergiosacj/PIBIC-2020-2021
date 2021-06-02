using NLPModels
using TimerOutputs
include("./backtrack_line_search.jl")

# NewtonCG with backtrack line search.
# outputs
# fcnt:    how many times the function has been evaluated.
# gcnt:    how many times the gradient has been evaluated.
# hcnt:    how many times the hessian has been evaluated.
# it:      problem iterations.
# itSUB:   subproblem(linear solver) iterations.
# itLS:    line search iterations.
# time:    total run time.
# timeSUB: subproblem(linear solver) run time.
# timeLS:  line search run time.
# SUBf:    how many times linear solver has failed.
# LSf:     how many times line search has failed.
# stop:    0 convergence has been achieved.
#          1 NewtonCG failed to converge.
#          2 time limit exceeded.

function newtoncg(problemTools; tle = 10, e = 1e-8, itMAX = 1e3)
	to = TimerOutput()
	@timeit to "newton_modified" begin
		# output variables
        fcnt = gcnt = hcnt = 0
        it = itSUB = itBLS = 0
        stop = 1
        SUBf = BLSf = 0
		allobj = zeros(Float64, Integer(itMAX))
		all∇f = zeros(Float64, Integer(itMAX))
		allalpha = zeros(Float64, Integer(itMAX))
		allpnorm = zeros(Float64, Integer(itMAX))
		x = fill(0.0, problemTools.size)
		file = open("OUTPUTS/tmp.txt", "a+")

		∇f = ∇fnorm = 0
		while it<itMAX
			# stop criteria
			∇f = problemTools.grad(x)
			∇fnorm = sqrt(sum(∇f.*∇f))
            gcnt += 1
            if ∇fnorm < e
                stop = 0
                break
			end

			p, j, failure = @timeit to "linear_solver" conjugategradient(-∇f, problemTools.hess(x))
			itSUB += j
			hcnt += 1
			SUBf += failure

			alpha, i, failure = @timeit to "backtrack_line_search" backtracklinesearch(x, problemTools, p, ∇f)
			itBLS += i
			BLSf += failure

			x = x + alpha.*p
			it += 1

            # saving data
            allobj[it] = problemTools.obj(x)
            all∇f[it] = ∇fnorm
            allalpha[it] = alpha
			allpnorm[it] = sqrt(sum(p.*p))
			println(file, "obj   -> $(allobj[it])")
			println(file, "grad  -> $(all∇f[it])")
			println(file, "alpha -> $(allalpha[it])")
			println(file, "pnorm -> $(allpnorm[it])")
		end
	end
	println(file, "stop -> $(stop)")
	close(file)
    values = [allobj, all∇f, allalpha, allpnorm]
	fcnt += itBLS
	return x, stop
end

function conjugategradient(r, B)
	# output: search direction, iterations, failure
	d = r
	z = zeros(Float64, size(d,1)) # is z = d-d faster?
	rip = ripold = sum(r.*r) # r inner product
	∇fnorm = sqrt(rip)
	e = min(0.5, sqrt(∇fnorm))*∇fnorm
	
	jMAX = 1000
	for j = 1:jMAX
		Bd = B*d
		dBd = sum(transpose(d)*Bd)
		if dBd <= 0
			if j == 1
				return d, j, 0
			else
				return z, j, 0
			end
		end

		alpha = rip/dBd
		z = z + alpha.*d
		r = r - alpha.*Bd

		rip = sum(r.*r)

		if sqrt(rip) <= e
			return z, j, 0
		end

		ß = rip/ripold
		d = r + ß.*d

		ripold = rip
	end
	
	return z, jMAX, 1
end
