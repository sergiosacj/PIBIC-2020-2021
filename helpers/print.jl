using Printf

function printHeader(file)
    @printf(file, "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ superscription ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n")
    @printf(file, "iter....: current iteration\n")
    @printf(file, "f(x*)...: objective function avaluated at x(iter)\n")
    @printf(file, "‖∇f(x)‖.: gradient norm used to calculate x(iter), so x in ‖∇f(x)‖ is equal to x(iter-1)\n")
    @printf(file, "alpha...: step calculated by backtrack line search\n")
    @printf(file, "‖d‖.....: search direction norm\n")
    @printf(file, "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n")

    println(file, repeat("_", 90))
    @printf(file, "%-5s  %-15s  %-15s %-15s\n", "iter", "f(x*)", "‖∇f(x)‖", "sigma")
    println(file, repeat("‾", 90))
end

function printEach(output::Array, file)
    k = output[1]
    allf = output[5]
    allg = output[6]
    allsigma = output[7]

    for i=1:k-1
        @printf(file, "%-5d  %-15e  %-15e %-15e\n", i, allf[i], allg[i], allsigma[i])
    end
end

function printProblemInfo(output::Array, file, stop)
    k = output[1]
    fcnt = output[2]
    gcnt = output[3]
    hcnt = output[4]
    allf = output[5]
    allg = output[6]
    allsigma = output[7]

    println("k = $(k)")

    @printf(file, "\n\n")
    @printf(file, "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ summary statistics ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n")
    @printf(file, "Total iterations......: %d\n", k)
    if k == 1
        k+=1
    end
    @printf(file, "AF....................: %d\n", fcnt)
    @printf(file, "AG....................: %d\n", gcnt)
    @printf(file, "AH....................: %d\n", hcnt)
    @printf(file, "f(x*).................: %e\n", allf[k-1])
    @printf(file, "‖∇f(x)‖...............: %e\n", allg[k-1])
    @printf(file, "sigma.................: %e\n", allsigma[k-1])
    @printf(file, "stop..................: %s\n", stop)
    @printf(file, "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n\n")
end
