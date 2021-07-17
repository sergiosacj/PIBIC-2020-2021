using CUTEst, Printf, TimerOutputs
include("../algorithms/AR.jl")

function tests()
    name = "ROSENBR"
    nlp = CUTEstModel(name)
    stop, output = ARp(nlp)

    out = open("Tests/ARp/$(name).out", "w")
    printHeader(out)
    printEach(output, out)
    printProblemInfo(output, out, stop)
    close(out)

    finalize(nlp)
end

tests()
