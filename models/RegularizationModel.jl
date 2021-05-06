module RegularizationModel

using NLPModels, LinearAlgebra

export RegNLP

"""
Model
    f(x) + ∇f * s + ∇²f * s² / 2 + sigma/(p+1) * ‖s‖^(p+1)
"""

mutable struct RegNLP <: AbstractNLPModel
  meta :: NLPModelMeta
  inner :: AbstractNLPModel
  objx
  gradx
  hessx
  sigma
  p
  x
end

function Base.show(io :: IO, nlp :: RegNLP)
  println(io, "RegNLP - Regularized model")
  show(io, nlp.meta)
  show(io, nlp.inner.counters)
end

function RegNLP(nlp :: AbstractNLPModel, sigma, x, p = 2)
  n = nlp.meta.nvar
  nnzh = nlp.meta.nnzh + n
  return RegNLP(NLPModelMeta(n, x0=[0, 0], nnzh=nnzh),
                nlp,
                obj(nlp, x),
                grad(nlp, x),
                hess_op(nlp, x),
                sigma,
                p,
                x)
end

@default_counters RegNLP inner

function NLPModels.obj(nlp :: RegNLP, s :: AbstractVector)
  sigma = nlp.sigma
  p = nlp.p

  if s == 0
    return nlp.objx
  end

  return nlp.objx
         + sum(s.*nlp.gradx)
         + sum(transpose(s)*(nlp.hessx*s)) / 2
         + sigma / (p+1) * norm(s)^(p+1)
end

function NLPModels.grad!(nlp :: RegNLP, s :: AbstractVector, g :: AbstractVector)
  grad!(nlp.inner, s, g)
  g = nlp.gradx + nlp.hessx * s + nlp.sigma * norm(s)^(nlp.p-1) * s
  return g
end

function NLPModels.hess_structure!(nlp :: RegNLP, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  nz = nlp.inner.meta.nnzh
  n = nlp.meta.nvar
  @views hess_structure!(nlp.inner, rows[1:nz], cols[1:nz])
  rows[nz+1:end] .= 1:n
  cols[nz+1:end] .= 1:n
  return rows, cols
end

function NLPModels.hess_coord!(nlp :: RegNLP, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Real=1.0)
  nz = nlp.inner.meta.nnzh
  @views hess_coord!(nlp.inner, x, vals[1:nz]; obj_weight=obj_weight)
  # vals[nz+1:end] .= 1 * obj_weight
  return vals
end

end # module
