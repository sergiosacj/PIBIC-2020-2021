using CUTEst, Printf, TimerOutputs
include("../algorithms/AR.jl")

problems =
[
"ROSENBR" ,
"POWELLBS" ,
"BROWNBS" ,
"BEALE" ,
"HELIX" ,
"BARD" ,
"BOX3" ,
"POWELLSG" ,
"WOODS" ,
"WATSON" ,
"SROSENBR" ,
"VARDIM" ,
"ARGTRIG" ,
"BROWNAL" ,
"BDVALUE" ,
"INTEGREQ" ,
"BROYDN3D" ,
"ARGLINA" ,
]

function tests(name)
    solver = "ARp"
    nlp = CUTEstModel(name)
    stop, output = ARp(nlp)

    out = open("Tests/$(solver)/$(name).out", "w")
    printHeader(out)
    printEach(output, out)
    printProblemInfo(output, out, stop)
    close(out)

    finalize(nlp)
end

for problem in problems
    tests(problem)
end
