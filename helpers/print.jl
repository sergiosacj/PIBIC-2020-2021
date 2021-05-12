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
    @printf(file, "%-6s  %-15s  %-15s  %-15s  %-15s  %-15s\n",
                  "iter", "f(x*)", "‖∇f(x)‖", "sigma", "p", "eta1")
    println(file, repeat("‾", 90))
end

function printEach(output::Array, file)
    k = output[1]
    obj = output[2]
    grad = output[3]
    sigma = output[4]
    p = output[5]
    eta1 = output[6]

    @printf(file, "%-6d  %-15e  %-15e  %-15e  %-15e  %-15e\n",
            k, obj, grad, sigma, p, eta1)
end

function printProblemInfo(output::Array, file)
    @printf(file, "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ summary statistics ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n")
    @printf(file, "Objective.............: %e\n", output[1])
    @printf(file, "Gradient norm.........: %e\n", output[2])
    @printf(file, "Total iterations......: %d\n", output[3])
    @printf(file, "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n\n")
end