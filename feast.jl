module feastUtil
    using LinearAlgebra
    using Random
    export linquad, 
            trapezoidal,
            getMaxResInside,
            Quadrrsolve,
            beyn,
            nseig,
            getInsideIndex
    include("feast_util.jl")
end

module feastCore
    using LinearAlgebra
    using Random
    using ..feastUtil
    export feast_core, feastNS_core
    include("feast_core.jl")
end

module feastLinear
    using SparseArrays
    using LinearAlgebra
    using Random
    using ..feastCore,..feastUtil,IterativeSolvers
    export feast_linear,ifeast_linear,feastNS_linear,ifeastNS_linear
    include("feast_linear.jl")
end

