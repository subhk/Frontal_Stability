using LazyGrids
using BlockArrays
using Printf
using StaticArrays
#using Interpolations
using SparseArrays
using SparseMatrixDicts
using SpecialFunctions
using FillArrays
using Parameters
using Test
using MAT
using BenchmarkTools

using Serialization
#using Pardiso
using Arpack
using LinearMaps
using ArnoldiMethod: partialschur, partialeigen, LR, LI, LM
using KrylovKit

struct ShiftAndInvert{TA,TB,TT}
    A_lu::TA
    B::TB
    temp::TT
end

function (M::ShiftAndInvert)(y, x)
    mul!(M.temp, M.B, x)
    ldiv!(y, M.A_lu, M.temp)
end

function construct_linear_map(A, B)
    a = ShiftAndInvert( factorize(A), B, Vector{eltype(A)}(undef, size(A,1)) )
    LinearMap{eltype(A)}(a, size(A,1), ismutating=true)
end

function nearestval_idx(a, x)
    idx::Int = 0
    for it =1:length(a)
        if a[it] == x
            idx = it
        end
    end
    return idx
end

function Eigs(𝓛, ℳ; σ::Float64, maxiter::Int)
    λₛ⁻¹, _, info = eigsolve(construct_linear_map(𝓛- σ*ℳ, ℳ), 
                                    rand(ComplexF64, size(𝓛,1)), 
                                    1, :LR, 
                                    maxiter=50, 
                                    krylovdim=250, 
                                    verbosity=0)

    if length(λₛ⁻¹) > 0
        λₛ⁰ = @. 1.0 / λₛ⁻¹ + σ
        idx = nearestval_idx(real(λₛ⁰), maximum(real(λₛ⁰)));
        λₛ = λₛ⁰[idx]
        #println(λₛ⁰)
    else
        λₛ = 0.0 + 0.0im
    end

    return λₛ, info.converged
end

function EigSolver_shift_invert_krylov_checking(𝓛, ℳ; σ₀::ComplexF64, α::Float64)
    info::Int = 1
    λₛ = []
    count::Int = -1
    λₛ₀ = zeros(ComplexF64, 1)
    λₛ₀[1] = σ₀
    try 
        push!(λₛ, λₛ₀[1])
        while info > 0
            λₛₜ = λₛ₀[1].re + α * λₛ₀[1].re
            @printf "target eigenvalue (α=%0.04f) λ: %f \n" α λₛₜ
            λₛ₀, info = Eigs(𝓛, ℳ; σ=λₛₜ, maxiter=20)
            if info > 0; push!(λₛ, λₛ₀[1]); end
            count += 1
        end
    catch error
        λₛ = Array(λₛ)
        if length(λₛ) > 1
            λₛ = sort_evals_(λₛ, "R")
        end
        #@printf "found eigenvalue (α=%0.04f): %f + im %f \n" α λₛ[1].re λₛ[1].im
    end
    λₛ = Array(λₛ)
    if length(λₛ) > 1
        λₛ = sort_evals_(λₛ, "R")
    end
    @printf "found eigenvalue (α=%0.04f): %f + im %f \n" α λₛ[1].re λₛ[1].im
    return λₛ[1]
end


function EigSolver_shift_invert_krylov(𝓛, ℳ; σ₀::Float64)
    maxiter::Int = 20
    try 
        σ = 0.50σ₀
        @printf "sigma: %f \n" real(σ) 
        λₛ, _ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
        @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
        return λₛ #, Χ
    catch error
        try
            σ = 0.75σ₀
            @printf "(first didn't work) sigma: %f \n" real(σ) 
            λₛ, _ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
            @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
            return λₛ #, Χ
        catch error
            try
                σ = 0.99σ₀
                @printf "(second didn't work) sigma: %f \n" real(σ)
                λₛ, _ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                return λₛ #, Χ
            catch error
                try
                    σ = 0.96σ₀
                    @printf "(third didn't work) sigma: %f \n" real(σ)
                    λₛ, _ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                    @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                    return λₛ #, Χ
                catch error
                    try
                        σ = 0.92σ₀
                        @printf "(fourth didn't work) sigma: %f \n" real(σ)
                        λₛ, _ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                        @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                        return λₛ #, Χ 
                    catch error
                        try
                            σ = 0.90σ₀
                            @printf "(fifth didn't work) sigma: %f \n" real(σ)
                            λₛ, _ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                            @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                            return λₛ #, Χ
                        catch error
                            try
                                σ = 0.85σ₀
                                @printf "(sixth didn't work) sigma: %f \n" real(σ)
                                λₛ, _ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                                return λₛ #, Χ
                            catch error
                                try
                                    σ = 0.82σ₀
                                    @printf "(seventh didn't work) sigma: %f \n" real(σ) 
                                    λₛ, _ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                    @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                                    return λₛ #, Χ
                                catch error
                                    try
                                        σ = 0.78σ₀
                                        @printf "(eighth didn't work) sigma: %f \n" real(σ) 
                                        λₛ, _ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                        @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                                        return λₛ #, Χ
                                    catch error
                                        try
                                            σ = 0.75σ₀
                                            @printf "(ninth didn't work) sigma: %f \n" real(σ) 
                                            λₛ, _ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                            @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                                            return λₛ #, Χ
                                        catch error
                                            try
                                                σ = 0.72σ₀
                                                @printf "(tenth didn't work) sigma: %f \n" real(σ) 
                                                λₛ, _ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                                @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                                                return λₛ #, Χ
                                            catch error
                                                σ = 0.69σ₀
                                                @printf "(eleventh didn't work) sigma: %f \n" real(σ)
                                                λₛ, _ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                                @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                                                return λₛ #, Χ
                                            end    
                                        end   
                                    end
                                end    
                            end
                        end                    
                    end          
                end    
            end
        end
    end
end

