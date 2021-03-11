# Versão do Julia

> julia/testing,now 1.5.3+dfsg-3 amd64

O motivo de eu estar usando essa é porque a versão mais antiga (1.3 se não me
engano) possuia um bug bem chato. Não lembro que bug era mas já estava corrigido
na versão mais recente. Tudo está sendo feito no Debian Testing.

# Como rodar

A princípio, apenas rodando:

```
$ julia runtests.jl
```

deve ser suficiente. Na primeira vez esse comando pode falhar por não ter
instalado todas as depedências, para instalar elas basta rodar:

```
$ julia
julia> ]
(@v1.5) pkg> add CUTEst, NLPModels, NLPModelsIpopt, LinearAlgebra, Printf, SparseArrays
```
