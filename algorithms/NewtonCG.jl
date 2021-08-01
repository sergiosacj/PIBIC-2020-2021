using NLPModels
using TimerOutputs

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
    # output variables
    fcnt = gcnt = hcnt = 0
    stop = 1
    x = fill(0.0, problemTools.size)
    it = 0

	to = TimerOutput()
	@timeit to "newton_modified" begin

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
			hcnt += 1

			alpha, i, failure = @timeit to "backtrack_line_search" backtracklinesearch(x, problemTools, p, ∇f)
            fcnt += 1

			x = x + alpha.*p
			it += 1
		end
	end

	return x, stop, fcnt, gcnt, hcnt
end

function conjugategradient(r, B)
	# output: search direction, iterations, failure
	d = r
	z = d-d
	rip = ripold = sum(r.*r)
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

function backtracklinesearch(x::Array, problemTools, p::Array, ∇f::Array)
	# output: alpha, iterations, failure
	alpha = 1
	c = 1e-4
	p_lo = 0.1
	p_hi = 0.5
	i = 0
	limit = 1000
	failure = 0

	∇fTp = sum(∇f.*p)
	objx = problemTools.obj(x)
	while problemTools.obj(x + alpha.*p) > objx + c*alpha*∇fTp
		alpha = min(alpha, alpha*p_hi) # avoid too big reductions
		alpha = max(alpha, alpha*p_lo) # avoid too small reductions

		i+=1
		if i>=limit
			failure = 1
			break
		end
	end

	return alpha, i, failure
end
