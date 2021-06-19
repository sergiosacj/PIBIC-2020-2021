using CUTEst, Printf, TimerOutputs
include("../algorithms/AR.jl")

function tests()
  nlp = CUTEstModel("ROSENBR")
  # rnlp = RegNLP(nlp, rand(), rand(2))
  # ipopt(rnlp)
  # ipopt(nlp)
  ARp(nlp)
  finalize(nlp)
end

tests()
