using CUTEst, Printf, TimerOutputs
include("../algorithms/AR.jl")
include("./print.jl")

function tests(inputFile)
    io = open(inputFile, "r")
    # skip first line
    readline(io)
    for i = 1:35
        # reading and parsing
        name, number, n, m = split(readline(io))
        number = parse(Int64, number)
        n = parse(Int64, n)
        m = parse(Int64, m)

        # running
        println("-----------------------------------------------")
        println("Started $(name)")

        nlp = CUTEstModel(name)
        stop, output = ARp(nlp)

        stop == 0 ? println("Converged") : println("Max iteration reached")
        out = open("Tests/ARp/$(name).out", "w")
        printHeader(out)
        printEach(output, out)
        printProblemInfo(output, out)
        close(out)

        finalize(nlp)
        println("-----------------------------------------------")
    end
    close(io)
end

tests("./helpers/mgh_problems.txt")
