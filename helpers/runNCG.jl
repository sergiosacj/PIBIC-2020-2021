using CUTEst, Printf, TimerOutputs
include("../models/NCGRegModel.jl")
include("../algorithms/NewtonCG.jl")

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

        println("Started $(name)")
        nlp = CUTEstModel(name)
        newtoncg(problemTools(nlp))
        # print results here
        finalize(nlp)
    end
    closed(io)
end

tests("./helpers/mgh_problems.txt")
