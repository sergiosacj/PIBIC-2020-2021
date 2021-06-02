using LinearAlgebra, NLPModels, Printf, SparseArrays
using CUTEst
include("../algorithms/oldAR.jl")

function tests()
  nlp = CUTEstModel("ROSENBR")
  # rnlp = RegNLP(nlp, rand(), rand(2))
  # ipopt(rnlp)
  # ipopt(nlp)
  ARp(nlp)
  finalize(nlp)
end

tests()
