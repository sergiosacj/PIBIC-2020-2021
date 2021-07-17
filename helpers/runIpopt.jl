using CUTEst, Printf, TimerOutputs
using NLPModelsIpopt
using Suppressor
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

        if name == "ARGAUSS" || name == "ARGLCLE"
            continue
        end

        # running
        println("-----------------------------------------------")
        println("Started $(name)")

        nlp = CUTEstModel(name)
        output = @capture_out ipopt(nlp, tol=1e-6)

        open("Tests/Ipopt/$(name).out", "w") do io
            write(io, output)
        end

        finalize(nlp)
        println("-----------------------------------------------")
    end
    close(io)
end

tests("./helpers/mgh_problems.txt")
