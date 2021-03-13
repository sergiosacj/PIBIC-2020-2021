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

# Informações extras

Como eu também não tenho muita familiaridade com todas as ferramentas de Julia, vou colocar nessa seção uma lista de materiais que estão me ajudando a entender melhor certos conceitos.

Link que explica o que é "@views":

```
https://discourse.julialang.org/t/could-you-explain-what-are-views/17535/2
```

Legenda dos termos utilizados no NLPModels:

```
https://github.com/JuliaSmoothOptimizers/NLPModels.jl#attributes
```