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
using LinearAlgebra

function sort_evals_(λₛ, which, sorting="lm")
    @assert which ∈ ["M", "I", "R"]

    if sorting == "lm"
        if which == "I"
            idx = sortperm(λₛ, by=imag, rev=true) 
        end
        if which == "R"
            idx = sortperm(λₛ, by=real, rev=true) 
        end
        if which == "M"
            idx = sortperm(λₛ, by=abs, rev=true) 
        end
    else
        if which == "I"
            idx = sortperm(λₛ, by=imag, rev=false) 
        end
        if which == "R"
            idx = sortperm(λₛ, by=real, rev=false) 
        end
        if which == "M"
            idx = sortperm(λₛ, by=abs, rev=false) 
        end
    end
    return λₛ[idx] 
end

function Eigs(𝓛, ℳ; σ::Float64, maxiter::Int)
    λₛ, _, info = Arpack.eigs(𝓛, ℳ, nev=1, 
                                    tol=1e-7, 
                                    maxiter=10, 
                                    which=:LR, 
                                    sigma=σ,
                                    check=0)
    return λₛ, info
end

function EigSolver_shift_invert_arpack_checking(𝓛, ℳ; σ₀::ComplexF64, α::Float64)
    converged = true
    λₛ = []
    count::Int = -1
    λₛ₀ = zeros(ComplexF64, 1)
    λₛ₀[1] = σ₀
    try 
        push!(λₛ, λₛ₀[1])
        while converged
            if count > -1; push!(λₛ, λₛ₀[1]); end
            λₛₜ = λₛ₀[1].re + α * λₛ₀[1].re
            @printf "target eigenvalue λ: %f \n" λₛₜ
            λₛ₀, info = Eigs(𝓛, ℳ; σ=λₛₜ, maxiter=20)
            count += 1
        end
    catch error
        λₛ = Array(λₛ)
        if length(λₛ) > 1
            λₛ = sort_evals_(λₛ, "R")
        end
        #λₛ, info = Eigs(𝓛, ℳ; σ=0.99λₛ[1].re, maxiter=20)
        @printf "found eigenvalue (α=%0.02f): %f + im %f \n" α λₛ[1].re λₛ[1].im
    end
    return λₛ[1]
end

function EigSolver_shift_invert1(𝓛, ℳ; σ₀::Float64)
    maxiter::Int = 20
    try 
        σ = 1.25σ₀
        @printf "sigma: %f \n" σ.re
        λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
        @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
        return λₛ #, Χ
    catch error
        try 
            σ = 1.15σ₀
            @printf "(first didn't work) sigma: %f \n" real(σ) 
            λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
            @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
            return λₛ #, Χ
        catch error
            try 
                σ = 1.10σ₀
                @printf "(second didn't work) sigma: %f \n" real(σ) 
                λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                return λₛ #, Χ
            catch error
                try
                    σ = 0.99σ₀
                    @printf "(third didn't work) sigma: %f \n" real(σ) 
                    λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                    @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                    return λₛ #, Χ
                catch error
                    try
                        σ = 0.95σ₀
                        @printf "(fourth didn't work) sigma: %f \n" real(σ) 
                        λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                        @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                        return λₛ #, Χ
                    catch error
                        try
                            σ = 0.90σ₀
                            @printf "(fifth didn't work) sigma: %f \n" real(σ) 
                            λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                            @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                            return λₛ #, Χ   
                        catch error
                            try
                                σ = 0.85σ₀
                                @printf "(sixth didn't work) sigma: %f \n" real(σ) 
                                λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                                return λₛ , Χ  
                            catch error
                                σ = 0.80σ₀
                                @printf "(seventh didn't work) sigma: %f \n" real(σ)
                                λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
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

function EigSolver_shift_invert(𝓛, ℳ; σ₀::Float64)
    maxiter::Int = 20
    try 
        σ = 1.10σ₀
        @printf "sigma: %f \n" σ
        λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
        @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
        return λₛ #, Χ
    catch error
        try
            σ = 1.05σ₀
            @printf "(first didn't work) sigma: %f \n" real(σ) 
            λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
            @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
            return λₛ #, Χ
        catch error
            try
                σ = 0.95σ₀
                @printf "(second didn't work) sigma: %f \n" real(σ)
                λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                return λₛ #, Χ
            catch error
                try
                    σ = 0.90σ₀
                    @printf "(third didn't work) sigma: %f \n" real(σ)
                    λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                    @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                    return λₛ #, Χ
                catch error
                    try
                        σ = 0.87σ₀
                        @printf "(fourth didn't work) sigma: %f \n" real(σ)
                        λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                        @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                        return λₛ #, Χ 
                    catch error
                        try
                            σ = 0.85σ₀
                            @printf "(fifth didn't work) sigma: %f \n" real(σ)
                            λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                            @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                            return λₛ #, Χ
                        catch error
                            try
                                σ = 0.80σ₀
                                @printf "(sixth didn't work) sigma: %f \n" real(σ)
                                λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                                return λₛ #, Χ
                            catch error
                                try
                                    σ = 0.70σ₀
                                    @printf "(seventh didn't work) sigma: %f \n" real(σ) 
                                    λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                    @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                                    return λₛ #, Χ
                                catch error
                                    try
                                        σ = 0.65σ₀
                                        @printf "(eighth didn't work) sigma: %f \n" real(σ) 
                                        λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                        @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                                        return λₛ #, Χ
                                    catch error
                                        try
                                            σ = 0.60σ₀
                                            @printf "(ninth didn't work) sigma: %f \n" real(σ) 
                                            λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                            @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                                            return λₛ #, Χ
                                        catch error
                                            try
                                                σ = 0.55σ₀
                                                @printf "(tenth didn't work) sigma: %f \n" real(σ) 
                                                λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                                @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                                                return λₛ #, Χ
                                            catch error
                                                σ = 0.50σ₀
                                                @printf "(eleventh didn't work) sigma: %f \n" real(σ)
                                                λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
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

function EigSolver_shift_invert_2(𝓛, ℳ; σ₀::Float64)
    maxiter::Int = 20
    try 
        σ = 0.90σ₀
        @printf "sigma: %f \n" σ
        λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
        @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
        return λₛ #, Χ
    catch error
        try
            σ = 0.87σ₀
            @printf "(first didn't work) sigma: %f \n" real(σ) 
            λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
            @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
            return λₛ #, Χ
        catch error
            try
                σ = 0.84σ₀
                @printf "(second didn't work) sigma: %f \n" real(σ)
                λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                return λₛ #, Χ
            catch error
                try
                    σ = 0.81σ₀
                    @printf "(third didn't work) sigma: %f \n" real(σ)
                    λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                    @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                    return λₛ #, Χ
                catch error
                    try
                        σ = 0.78σ₀
                        @printf "(fourth didn't work) sigma: %f \n" real(σ)
                        λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                        @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                        return λₛ #, Χ 
                    catch error
                        try
                            σ = 0.75σ₀
                            @printf "(fifth didn't work) sigma: %f \n" real(σ)
                            λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                            @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                            return λₛ #, Χ
                        catch error
                            try
                                σ = 0.70σ₀
                                @printf "(sixth didn't work) sigma: %f \n" real(σ)
                                λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                                return λₛ #, Χ
                            catch error
                                try
                                    σ = 0.65σ₀
                                    @printf "(seventh didn't work) sigma: %f \n" real(σ) 
                                    λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                    @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                                    return λₛ #, Χ
                                catch error
                                    try
                                        σ = 0.60σ₀
                                        @printf "(eighth didn't work) sigma: %f \n" real(σ) 
                                        λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                        @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                                        return λₛ #, Χ
                                    catch error
                                        try
                                            σ = 0.55σ₀
                                            @printf "(ninth didn't work) sigma: %f \n" real(σ) 
                                            λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                            @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                                            return λₛ #, Χ
                                        catch error
                                            try
                                                σ = 0.50σ₀
                                                @printf "(tenth didn't work) sigma: %f \n" real(σ) 
                                                λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                                @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
                                                return λₛ #, Χ
                                            catch error
                                                σ = 0.45σ₀
                                                @printf "(eleventh didn't work) sigma: %f \n" real(σ)
                                                λₛ, info = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
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