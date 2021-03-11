using LinearAlgebra, NLPModels, NLPModelsIpopt, Printf, SparseArrays
using CUTEst
include("./AR.jl")

function tests()
  nlp = CUTEstModel("ROSENBR")
  # rnlp = RegNLP(nlp, rand(), rand(2))
  # ipopt(rnlp)
  # ipopt(nlp)
  ARp(nlp)
  finalize(nlp)
end

tests()
