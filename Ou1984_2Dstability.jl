#=
Stability of a 2D front: an example setup
=#
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
using BasicInterpolators: BicubicInterpolator

using Serialization
#using Pardiso
using Arpack
using LinearMaps
using ArnoldiMethod: partialschur, partialeigen, LR, LI, LM

using CairoMakie
using LaTeXStrings
CairoMakie.activate!()
using DelimitedFiles
using ColorSchemes
using ScatteredInterpolation: interpolate, 
                            evaluate, 
                            InverseMultiquadratic, 
                            Multiquadratic
using Statistics
using JLD2
using Dierckx #: Spline2D, evaluate
using MatrixMarket: mmwrite
using ModelingToolkit, NonlinearSolve
using IterativeSolvers
using LinearAlgebra
using JacobiDavidson, Plots


include("dmsuite.jl")
include("transforms.jl")
include("utils.jl")
include("setBCs.jl")
include("shift_invert.jl")
include("shift_invert_arnoldi.jl")
include("shift_invert_krylov.jl")

include("feast.jl")
using ..feastLinear

# include("FEASTSolver/src/FEASTSolver.jl")
# using Main.FEASTSolver

@with_kw mutable struct TwoDimGrid{Ny, Nz} 
    y = @SVector zeros(Float64, Ny)
    z = @SVector zeros(Float64, Nz)
end

@with_kw mutable struct ChebMarix{Ny, Nz} 
    𝒟ʸ::Array{Float64,  2}   = SparseMatrixCSC(Zeros(Ny, Ny))
    𝒟²ʸ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(Ny, Ny))
    𝒟⁴ʸ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(Ny, Ny))

    𝒟ᶻ::Array{Float64,  2}   = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟²ᶻ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟⁴ᶻ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(Nz, Nz))

    𝒟ᶻᴺ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟²ᶻᴺ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟⁴ᶻᴺ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))

    𝒟ᶻᴰ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟²ᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟⁴ᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))
end

@with_kw mutable struct Operator{N}
"""
    `subperscript with N' means Operator with Neumann boundary condition 
        after kronker product
    `subperscript with D' means Operator with Dirchilet boundary condition
        after kronker product
""" 

    𝒟ʸ::Array{Float64,  2}   = SparseMatrixCSC(Zeros(N, N))
    𝒟²ʸ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(N, N))
    𝒟⁴ʸ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(N, N))

    𝒟ᶻ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟²ᶻ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))

    𝒟ᶻᴺ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟²ᶻᴺ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟⁴ᶻᴺ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))

    𝒟ᶻᴰ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟ʸᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟²ᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟⁴ᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))

    𝒟ʸ²ᶻᴰ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟²ʸ²ᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))
end

@with_kw mutable struct MeanFlow{N} 
    B₀::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))
    U₀::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))

  ∇ʸU₀::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))
  ∇ᶻU₀::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))
  ∇ʸB₀::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))
  ∇ᶻB₀::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))

  ∇ʸʸU₀::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))
  ∇ᶻᶻU₀::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))
  ∇ʸᶻU₀::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))
end


function ImplementBCs_Dirchilet_on_D1_Order4!(diffMatrix, grid)
    N   = length(grid.z)
    del = grid.z[2] - grid.z[1] 
    diffMatrix.𝒟ᶻᴰ[1,:] .= 0.0;              diffMatrix.𝒟ᶻᴰ[1,1] = -(1/12)/del;
    diffMatrix.𝒟ᶻᴰ[1,2]  = (2/3)/del;        diffMatrix.𝒟ᶻᴰ[1,3] = -(1/12)/del;

    diffMatrix.𝒟ᶻᴰ[2,:] .= 0.0;              diffMatrix.𝒟ᶻᴰ[2,1] = -(2/3)/del;
    diffMatrix.𝒟ᶻᴰ[2,2]  = 0.0;              diffMatrix.𝒟ᶻᴰ[2,3] = (2/3)/del;
    diffMatrix.𝒟ᶻᴰ[2,4]  = -(1/12)/del;

    diffMatrix.𝒟ᶻᴰ[N,:]  .= 0;               diffMatrix.𝒟ᶻᴰ[N,N]   = (1/12)/del;
    diffMatrix.𝒟ᶻᴰ[N,N-1] = -(2/3)/del;      diffMatrix.𝒟ᶻᴰ[N,N-2] = (1/12)/del;

    diffMatrix.𝒟ᶻᴰ[N-1,:]  .= 0;             diffMatrix.𝒟ᶻᴰ[N-1,N]   = (2/3)/del;
    diffMatrix.𝒟ᶻᴰ[N-1,N-1] = 0;             diffMatrix.𝒟ᶻᴰ[N-1,N-2] = -(2/3)/del;
    diffMatrix.𝒟ᶻᴰ[N-1,N-3] = (1/12)/del; 
end

function ImplementBCs_Dirchilet_on_D2_Order4!(diffMatrix, grid)
    N   = length(grid.z)
    del = grid.z[2] - grid.z[1] 
    diffMatrix.𝒟²ᶻᴰ[1,:] .= 0;
    diffMatrix.𝒟²ᶻᴰ[1,1]  = -2/del^2;         diffMatrix.𝒟²ᶻᴰ[1,2] = 1/del^2;  

    diffMatrix.𝒟²ᶻᴰ[2,:] .= 0;                diffMatrix.𝒟²ᶻᴰ[2,1] = (4/3)/del^2; 
    diffMatrix.𝒟²ᶻᴰ[2,2]  = -(5/2)/del^2;     diffMatrix.𝒟²ᶻᴰ[2,3] = (4/3)/del^2;
    diffMatrix.𝒟²ᶻᴰ[2,4]  = -(1/12)/del^2;     

    diffMatrix.𝒟²ᶻᴰ[N,:]  .= 0;
    diffMatrix.𝒟²ᶻᴰ[N,N]   = -2/del^2;        diffMatrix.𝒟²ᶻᴰ[N,N-1] = 1/del^2;  

    diffMatrix.𝒟²ᶻᴰ[N-1,:]   .= 0;            diffMatrix.𝒟²ᶻᴰ[N-1,N]   = (4/3)/del^2;
    diffMatrix.𝒟²ᶻᴰ[N-1,N-1] = -(5/2)/del^2;  diffMatrix.𝒟²ᶻᴰ[N-1,N-2] = (4/3)/del^2;
    diffMatrix.𝒟²ᶻᴰ[N-1,N-3] = -(1/12)/del^2;
end

function ImplementBCs_Dirchilet_on_D4_Order4!(diffMatrix, grid)
    N   = length(grid.z)
    del = grid.z[2] - grid.z[1] 
    diffMatrix.𝒟⁴ᶻᴰ[1,:] .= 0;                  diffMatrix.𝒟⁴ᶻᴰ[1,1] = 5/del^4;
    diffMatrix.𝒟⁴ᶻᴰ[1,2]  = -4/del^4;           diffMatrix.𝒟⁴ᶻᴰ[1,3] = 1/del^4;
    
    diffMatrix.𝒟⁴ᶻᴰ[2,:] .= 0;                  diffMatrix.𝒟⁴ᶻᴰ[2,1] = -(38/6)/del^4;
    diffMatrix.𝒟⁴ᶻᴰ[2,2]  = (28/3)/del^4;       diffMatrix.𝒟⁴ᶻᴰ[2,3] = -(13/2)/del^4;
    diffMatrix.𝒟⁴ᶻᴰ[2,4]  = 2/del^4;            diffMatrix.𝒟⁴ᶻᴰ[2,5] = -(1/6)/del^4;
    
    diffMatrix.𝒟⁴ᶻᴰ[3,:] .= 0;                  diffMatrix.𝒟⁴ᶻᴰ[3,1] = 2/del^4;
    diffMatrix.𝒟⁴ᶻᴰ[3,2]  = -(13/2)/del^4;      diffMatrix.𝒟⁴ᶻᴰ[3,3] = (28/3)/del^4;
    diffMatrix.𝒟⁴ᶻᴰ[3,4]  = -(13/2)/del^4;      diffMatrix.𝒟⁴ᶻᴰ[3,5] = 2/del^4;
    diffMatrix.𝒟⁴ᶻᴰ[3,6]  = -(1/6)/del^4;
    
    diffMatrix.𝒟⁴ᶻᴰ[N,:]  .= 0;                 diffMatrix.𝒟⁴ᶻᴰ[N,N]   = 5/del^4;
    diffMatrix.𝒟⁴ᶻᴰ[N,N-1] = -4/del^4;          diffMatrix.𝒟⁴ᶻᴰ[N,N-2] = 1/del^4;
    
    diffMatrix.𝒟⁴ᶻᴰ[N-1,:]  .= 0;               diffMatrix.𝒟⁴ᶻᴰ[N-1,N]   = -(38/6)/del^4;
    diffMatrix.𝒟⁴ᶻᴰ[N-1,N-1] = (28/3)/del^4;    diffMatrix.𝒟⁴ᶻᴰ[N-1,N-2] = -(13/2)/del^4;
    diffMatrix.𝒟⁴ᶻᴰ[N-1,N-3] = 2/del^4;         diffMatrix.𝒟⁴ᶻᴰ[N-1,N-4] = -(1/6)/del^4;
    
    diffMatrix.𝒟⁴ᶻᴰ[N-2,:] .= 0;                diffMatrix.𝒟⁴ᶻᴰ[N-2,N]   = 2/del^4;
    diffMatrix.𝒟⁴ᶻᴰ[N-2,N-1]  = -(13/2)/del^4;  diffMatrix.𝒟⁴ᶻᴰ[N-2,N-2] = (28/3)/del^4;
    diffMatrix.𝒟⁴ᶻᴰ[N-2,N-3]  = -(13/2)/del^4;  diffMatrix.𝒟⁴ᶻᴰ[N-2,N-4] = 2/del^4;
    diffMatrix.𝒟⁴ᶻᴰ[N-2,N-5]  = -(1/6)/del^4;
end


function ImplementBCs_Neumann_on_D1_Order4!(diffMatrix, grid)
    N   = length(grid.z)
    del = grid.z[2] - grid.z[1] 
    diffMatrix.𝒟ᶻᴺ[1,:]    .= 0;              diffMatrix.𝒟ᶻᴺ[1,1] = -1/del;
    diffMatrix.𝒟ᶻᴺ[1,2]     = 1/del;         

    diffMatrix.𝒟ᶻᴺ[2,:]    .= 0;              diffMatrix.𝒟ᶻᴺ[2,1] = -(7/12)/del;
    diffMatrix.𝒟ᶻᴺ[2,2]     = 0;              diffMatrix.𝒟ᶻᴺ[2,3] = (2/3)/del;
    diffMatrix.𝒟ᶻᴺ[2,4]     = -(1/12)/del;

    diffMatrix.𝒟ᶻᴺ[N,:]    .= 0;              diffMatrix.𝒟ᶻᴺ[N,N] = 1/del;
    diffMatrix.𝒟ᶻᴺ[N,N-1]   = -1/del;      

    diffMatrix.𝒟ᶻᴺ[N-1,:]  .= 0;              diffMatrix.𝒟ᶻᴺ[N-1,N]   = (7/12)/del;
    diffMatrix.𝒟ᶻᴺ[N-1,N-1] = 0;              diffMatrix.𝒟ᶻᴺ[N-1,N-2] = -(2/3)/del;
    diffMatrix.𝒟ᶻᴺ[N-1,N-3] = (1/12)/del; 
end

function ImplementBCs_Neumann_on_D2_Order4!(diffMatrix, grid)
    N   = length(grid.z)
    del = grid.z[2] - grid.z[1] 
    diffMatrix.𝒟²ᶻᴺ[1,:] .= 0;                  diffMatrix.𝒟²ᶻᴺ[1,1] = -1/del^2;
    diffMatrix.𝒟²ᶻᴺ[1,2]  = 1/del^2;         

    diffMatrix.𝒟²ᶻᴺ[2,:] .= 0;                  diffMatrix.𝒟²ᶻᴺ[2,1] = (15/12)/del^2;
    diffMatrix.𝒟²ᶻᴺ[2,2]  = -(5/2)/del^2;       diffMatrix.𝒟²ᶻᴺ[2,3] = (4/3)/del^2;
    diffMatrix.𝒟²ᶻᴺ[2,4]  = -(1/12)/del^2;

    diffMatrix.𝒟²ᶻᴺ[N,:]  .= 0;                 diffMatrix.𝒟²ᶻᴺ[N,N] = -1/del^2;
    diffMatrix.𝒟²ᶻᴺ[N,N-1] = 1/del^2;      

    diffMatrix.𝒟²ᶻᴺ[N-1,:]  .= 0;               diffMatrix.𝒟²ᶻᴺ[N-1,N]   = (15/12)/del^2;
    diffMatrix.𝒟²ᶻᴺ[N-1,N-1] = -(5/2)/del^2;    diffMatrix.𝒟²ᶻᴺ[N-1,N-2] = (4/3)/del^2;
    diffMatrix.𝒟²ᶻᴺ[N-1,N-3] = -(1/12)/del^2;  
end

"""
Construct the derivative operator
"""
function Construct_DerivativeOperator!(diffMatrix, grid, params)
    N = params.Ny * params.Nz

    # ------------- setup differentiation matrices  -------------------
    # Fourier in y-direction: y ∈ [0, L)
    y1, diffMatrix.𝒟ʸ  = FourierDiff(params.Ny, 1)
    _,  diffMatrix.𝒟²ʸ = FourierDiff(params.Ny, 2)
    _,  diffMatrix.𝒟⁴ʸ = FourierDiff(params.Ny, 4)

    # 2nd order accurate finite difference method
    # y1, diffMatrix.𝒟ʸ  = FourierDiff_fdm(params.Ny, 1)
    # _,  diffMatrix.𝒟²ʸ = FourierDiff_fdm(params.Ny, 2)
    # _,  diffMatrix.𝒟⁴ʸ = FourierDiff_fdm(params.Ny, 4)

    # 4th order accurate finite difference method
    # y1, diffMatrix.𝒟ʸ  = FourierDiff_fdm_4th(params.Ny, 1)
    # _,  diffMatrix.𝒟²ʸ = FourierDiff_fdm_4th(params.Ny, 2)
    # _,  diffMatrix.𝒟⁴ʸ = FourierDiff_fdm_4th(params.Ny, 4)

    t1 = @. sin(y1)
    t2 = diffMatrix.𝒟ʸ * t1

    println(t1[1])
    println(t2[1])

    # Transform the domain and derivative operators from [0, 2π) → [0, L)
    grid.y         = params.L/2π  * y1
    diffMatrix.𝒟ʸ  = (2π/params.L)^1 * diffMatrix.𝒟ʸ
    diffMatrix.𝒟²ʸ = (2π/params.L)^2 * diffMatrix.𝒟²ʸ
    diffMatrix.𝒟⁴ʸ = (2π/params.L)^4 * diffMatrix.𝒟⁴ʸ

    #@assert maximum(grid.y) ≈ params.L && minimum(grid.y) ≈ 0.0

    if params.z_discret == "cheb"
        # Chebyshev in the z-direction
        # z, diffMatrix.𝒟ᶻ  = cheb(params.Nz-1)
        # grid.z = z
        # diffMatrix.𝒟²ᶻ = diffMatrix.𝒟ᶻ  * diffMatrix.𝒟ᶻ
        # diffMatrix.𝒟⁴ᶻ = diffMatrix.𝒟²ᶻ * diffMatrix.𝒟²ᶻ

        # @printf "size of Chebyshev matrix: %d × %d \n" size(diffMatrix.𝒟ᶻ)[1]  size(diffMatrix.𝒟ᶻ)[2]

        z1, D1z = chebdif(params.Nz, 1)
        _,  D2z = chebdif(params.Nz, 2)
        _,  D3z = chebdif(params.Nz, 3)
        _,  D4z = chebdif(params.Nz, 4)
        # Transform the domain and derivative operators from [-1, 1] → [0, H]
        grid.z, diffMatrix.𝒟ᶻ, diffMatrix.𝒟²ᶻ  = chebder_transform(z1,  D1z, 
                                                                        D2z, 
                                                                        zerotoL_transform, 
                                                                        params.H)
        _, _, diffMatrix.𝒟⁴ᶻ = chebder_transform_ho(z1, D1z, 
                                                        D2z, 
                                                        D3z, 
                                                        D4z, 
                                                        zerotoL_transform_ho, 
                                                        params.H)
        
        @assert maximum(grid.z) ≈ params.H && minimum(grid.z) ≈ 0.0

    elseif params.z_discret == "fdm"
        ## finite difference method in the z-direction
        grid.z = collect(range(0.0, stop=params.H, length=params.Nz))
        @assert std(diff(grid.z)) ≤ 1e-6
        diffMatrix.𝒟ᶻ  = ddz(  grid.z, order_accuracy=params.order_accuracy );
        diffMatrix.𝒟²ᶻ = ddz2( grid.z, order_accuracy=params.order_accuracy );
        diffMatrix.𝒟⁴ᶻ = ddz4( grid.z, order_accuracy=params.order_accuracy );
    else
        error("Invalid discretization type")
    end

    @testset "checking z-derivative differentiation matrix" begin
        tol = 1.0e-4
        t1 = diffMatrix.𝒟ᶻ * grid.z;
        @test maximum(t1) ≈ 1.0 atol=tol
        @test minimum(t1) ≈ 1.0 atol=tol
        t1 = diffMatrix.𝒟²ᶻ * (grid.z .^ 2);
        @test maximum(t1) ≈ factorial(2) atol=tol
        @test minimum(t1) ≈ factorial(2) atol=tol
        t1 = diffMatrix.𝒟⁴ᶻ * (grid.z .^ 4);
        @test maximum(t1) ≈ factorial(4) atol=tol
        @test minimum(t1) ≈ factorial(4) atol=tol
    end
    return nothing
end

function ImplementBCs_cheb!(Op, diffMatrix, params)
    Iʸ = sparse(Matrix(1.0I, params.Ny, params.Ny)) #Eye{Float64}(params.Ny)
    Iᶻ = sparse(Matrix(1.0I, params.Nz, params.Nz)) #Eye{Float64}(params.Nz)

    # Dirichilet boundary condition
    @. diffMatrix.𝒟ᶻᴰ  = diffMatrix.𝒟ᶻ 
    @. diffMatrix.𝒟²ᶻᴰ = diffMatrix.𝒟²ᶻ
    @. diffMatrix.𝒟⁴ᶻᴰ = diffMatrix.𝒟⁴ᶻ

    n = params.Nz
    for iter ∈ 1:n-1
        diffMatrix.𝒟⁴ᶻᴰ[1,iter+1] = (diffMatrix.𝒟⁴ᶻᴰ[1,iter+1] + 
                                -1.0 * diffMatrix.𝒟⁴ᶻᴰ[1,1] * diffMatrix.𝒟²ᶻᴰ[1,iter+1])

          diffMatrix.𝒟⁴ᶻᴰ[n,iter] = (diffMatrix.𝒟⁴ᶻᴰ[n,iter] + 
                                -1.0 * diffMatrix.𝒟⁴ᶻᴰ[n,n] * diffMatrix.𝒟²ᶻᴰ[n,iter])
    end

    diffMatrix.𝒟ᶻᴰ[1,1]  = 0.0
    diffMatrix.𝒟ᶻᴰ[n,n]  = 0.0

    diffMatrix.𝒟²ᶻᴰ[1,1] = 0.0
    diffMatrix.𝒟²ᶻᴰ[n,n] = 0.0   

    diffMatrix.𝒟⁴ᶻᴰ[1,1] = 0.0
    diffMatrix.𝒟⁴ᶻᴰ[n,n] = 0.0  

    # Neumann boundary condition
    @. diffMatrix.𝒟ᶻᴺ  = diffMatrix.𝒟ᶻ 
    @. diffMatrix.𝒟²ᶻᴺ = diffMatrix.𝒟²ᶻ
    for iter ∈ 1:n-1
        diffMatrix.𝒟²ᶻᴺ[1,iter+1] = (diffMatrix.𝒟²ᶻᴺ[1,iter+1] + 
                                -1.0 * diffMatrix.𝒟²ᶻᴺ[1,1] * diffMatrix.𝒟ᶻᴺ[1,iter+1]/diffMatrix.𝒟ᶻᴺ[1,1])

        diffMatrix.𝒟²ᶻᴺ[n,iter]   = (diffMatrix.𝒟²ᶻᴺ[n,iter] + 
                                -1.0 * diffMatrix.𝒟²ᶻᴺ[n,n] * diffMatrix.𝒟ᶻᴺ[n,iter]/diffMatrix.𝒟ᶻᴺ[n,n])
    end

    diffMatrix.𝒟²ᶻᴺ[1,1] = 0.0
    diffMatrix.𝒟²ᶻᴺ[n,n] = 0.0

    @. diffMatrix.𝒟ᶻᴺ[1,1:end] = 0.0
    @. diffMatrix.𝒟ᶻᴺ[n,1:end] = 0.0
    
    kron!( Op.𝒟ᶻᴰ  ,  Iʸ , diffMatrix.𝒟ᶻᴰ  )
    kron!( Op.𝒟²ᶻᴰ ,  Iʸ , diffMatrix.𝒟²ᶻᴰ )
    kron!( Op.𝒟⁴ᶻᴰ ,  Iʸ , diffMatrix.𝒟⁴ᶻᴰ )

    kron!( Op.𝒟ᶻᴺ  ,  Iʸ , diffMatrix.𝒟ᶻᴺ )
    kron!( Op.𝒟²ᶻᴺ ,  Iʸ , diffMatrix.𝒟²ᶻᴺ)

    kron!( Op.𝒟ʸ   ,  diffMatrix.𝒟ʸ  ,  Iᶻ ) 
    kron!( Op.𝒟²ʸ  ,  diffMatrix.𝒟²ʸ ,  Iᶻ )
    kron!( Op.𝒟⁴ʸ  ,  diffMatrix.𝒟⁴ʸ ,  Iᶻ ) 

    kron!( Op.𝒟ʸᶻᴰ   ,  diffMatrix.𝒟ʸ  ,  diffMatrix.𝒟ᶻᴰ  )
    kron!( Op.𝒟ʸ²ᶻᴰ  ,  diffMatrix.𝒟ʸ  ,  diffMatrix.𝒟²ᶻᴰ )
    kron!( Op.𝒟²ʸ²ᶻᴰ ,  diffMatrix.𝒟²ʸ ,  diffMatrix.𝒟²ᶻᴰ )

    return nothing
end


function ImplementBCs_fdm!(Op, diffMatrix, grid, params)
    # Dirchilet boundary condition
    @. diffMatrix.𝒟ᶻᴰ  = diffMatrix.𝒟ᶻ 
    @. diffMatrix.𝒟²ᶻᴰ = diffMatrix.𝒟²ᶻ
    @. diffMatrix.𝒟⁴ᶻᴰ = diffMatrix.𝒟⁴ᶻ
        
    # Neumann boundary condition
    @. diffMatrix.𝒟ᶻᴺ  = diffMatrix.𝒟ᶻ 
    @. diffMatrix.𝒟²ᶻᴺ = diffMatrix.𝒟²ᶻ

    if params.order_accuracy == 4
        ImplementBCs_Dirchilet_on_D1_Order4!(diffMatrix,  grid )
        ImplementBCs_Dirchilet_on_D2_Order4!(diffMatrix,  grid )
        ImplementBCs_Dirchilet_on_D4_Order4!(diffMatrix,  grid )
        ImplementBCs_Neumann_on_D1_Order4!(  diffMatrix,  grid ) 
        ImplementBCs_Neumann_on_D2_Order4!(  diffMatrix,  grid ) 
    else
        ImplementBCs_Dirchilet_on_D1_Order2!(diffMatrix,  grid )
        ImplementBCs_Dirchilet_on_D2_Order2!(diffMatrix,  grid )
        ImplementBCs_Dirchilet_on_D4_Order2!(diffMatrix,  grid )
        ImplementBCs_Neumann_on_D1_Order2!(  diffMatrix,  grid ) 
        ImplementBCs_Neumann_on_D2_Order2!(  diffMatrix,  grid ) 
    end

    Iʸ = sparse(Matrix(1.0I, params.Ny, params.Ny)) #Eye{Float64}(params.Ny)
    Iᶻ = sparse(Matrix(1.0I, params.Nz, params.Nz)) #Eye{Float64}(params.Nz)

    kron!( Op.𝒟ᶻᴰ  ,  Iʸ , diffMatrix.𝒟ᶻᴰ  )
    kron!( Op.𝒟²ᶻᴰ ,  Iʸ , diffMatrix.𝒟²ᶻᴰ )
    kron!( Op.𝒟⁴ᶻᴰ ,  Iʸ , diffMatrix.𝒟⁴ᶻᴰ )

    kron!( Op.𝒟ᶻᴺ  ,  Iʸ , diffMatrix.𝒟ᶻᴺ  )
    kron!( Op.𝒟²ᶻᴺ ,  Iʸ , diffMatrix.𝒟²ᶻᴺ )

    kron!( Op.𝒟ʸ   ,  diffMatrix.𝒟ʸ  ,  Iᶻ ) 
    kron!( Op.𝒟²ʸ  ,  diffMatrix.𝒟²ʸ ,  Iᶻ )
    kron!( Op.𝒟⁴ʸ  ,  diffMatrix.𝒟⁴ʸ ,  Iᶻ ) 

    kron!( Op.𝒟ʸᶻᴰ   ,  diffMatrix.𝒟ʸ  ,  diffMatrix.𝒟ᶻᴰ  )
    kron!( Op.𝒟ʸ²ᶻᴰ  ,  diffMatrix.𝒟ʸ  ,  diffMatrix.𝒟²ᶻᴰ )
    kron!( Op.𝒟²ʸ²ᶻᴰ ,  diffMatrix.𝒟²ʸ ,  diffMatrix.𝒟²ᶻᴰ )

    return nothing
end

function BasicState!(diffMatrix, mf, grid, params)
    @variables η ξ 
    @parameters β y₀ z₀

    # Define a nonlinear system
    eqs = [η + (0.5 - z₀) * ξ - y₀ ~ 0, ξ + 0.5*β/(cosh(β * η)*cosh(β * η)) ~ 0]
    @named ns = NonlinearSystem(eqs, [η, ξ], [β, y₀, z₀])
    ns = structural_simplify(ns)  # needed when running on Apple M1 and later version 

    y = grid.y 
    z = grid.z
    Y, Z = ndgrid(y, z)

    η₀ = zeros(length(y), length(z))
    ξ₀ = zeros(length(y), length(z))

    u0 = [η => 3.0, ξ => 2.0]
    for it in 1:length(y)
        for jt in 1:length(z)
            ps = [β  => params.β
                y₀ => y[it]
                z₀ => z[jt]]

            prob = NonlinearProblem(ns, u0, ps);
            sol = solve(prob, NewtonRaphson());

            #println(size(sol))
            
            η₀[it,jt] = sol[1]
            ξ₀[it,jt] = (y[it] - sol[1])/(0.5 - z[jt]) # needed this line on Apple M1 and later version 
            #ξ₀[it,jt] = sol[2]  # this works on linux not on Apple M1 or later
        end
    end

    U₀ = zeros(length(y), length(z))
    B₀ = zeros(length(y), length(z))

    @. U₀ = (0.5 - Z) * ξ₀;
    @. B₀ = -0.5tanh(params.β*η₀)

    ∂ʸB₀  = similar(B₀)
    ∂ᶻB₀  = similar(B₀)

    ∂ʸU₀  = similar(B₀)
    ∂ᶻU₀  = similar(B₀)

    ∂ʸʸU₀ = similar(B₀)
    ∂ʸᶻU₀ = similar(B₀)
    ∂ᶻᶻU₀ = similar(B₀)

    """
    Calculating necessary derivatives of the mean-flow quantities
    """
    ∂ʸB₀   = gradient(  B₀,  grid.y, dims=1)
    ∂ʸU₀   = gradient(  U₀,  grid.y, dims=1)
    ∂ʸʸU₀  = gradient2( U₀,  grid.y, dims=1)

    # `Thermal wind balance'
    @. ∂ᶻU₀  = -1.0 * ∂ʸB₀

    for iy ∈ 1:length(grid.y)
         #∂ᶻU₀[iy,:] = diffMatrix.𝒟ᶻ * U₀[iy,:]
         ∂ᶻB₀[iy,:] = diffMatrix.𝒟ᶻ * B₀[iy,:]
        ∂ᶻᶻU₀[iy,:] = diffMatrix.𝒟ᶻ * ∂ᶻU₀[iy,:]
    end

    # ∂ᶻB₀   = gradient( B₀,  grid.z, dims=2)
    # ∂ᶻᶻU₀  = gradient2(U₀,  grid.z, dims=2)

    ∂ʸᶻU₀ = gradient(∂ᶻU₀, grid.y, dims=1)

    @printf "min/max values of ∂ᶻU₀: %f %f \n" minimum(∂ᶻU₀) maximum(∂ᶻU₀)
    @printf "min/max values of ∂ʸU₀: %f %f \n" minimum(∂ʸU₀) maximum(∂ʸU₀)
    @printf "min/max values of ∂ᶻB₀: %f %f \n" minimum(∂ᶻB₀) maximum(∂ᶻB₀)
    @printf "min/max values of ∂ʸB₀: %f %f \n" minimum(∂ʸB₀) maximum(∂ʸB₀)

    @printf "min/max values of ∂ʸᶻU₀: %f %f \n" minimum(∂ʸᶻU₀) maximum(∂ʸᶻU₀)
    @printf "min/max values of ∂ᶻᶻU₀: %f %f \n" minimum(∂ᶻᶻU₀) maximum(∂ᶻᶻU₀)

    B₀    = transpose(B₀);       B₀ = B₀[:];
    U₀    = transpose(U₀);       U₀ = U₀[:];

    ∂ʸB₀  = transpose(∂ʸB₀);   ∂ʸB₀ = ∂ʸB₀[:];
    ∂ʸU₀  = transpose(∂ʸU₀);   ∂ʸU₀ = ∂ʸU₀[:];

    ∂ᶻB₀  = transpose(∂ᶻB₀);   ∂ᶻB₀ = ∂ᶻB₀[:];
    ∂ᶻU₀  = transpose(∂ᶻU₀);   ∂ᶻU₀ = ∂ᶻU₀[:];

    ∂ʸʸU₀ = transpose(∂ʸʸU₀); ∂ʸʸU₀ = ∂ʸʸU₀[:];
    ∂ᶻᶻU₀ = transpose(∂ᶻᶻU₀); ∂ᶻᶻU₀ = ∂ᶻᶻU₀[:];
    ∂ʸᶻU₀ = transpose(∂ʸᶻU₀); ∂ʸᶻU₀ = ∂ʸᶻU₀[:];

    mf.B₀[diagind(mf.B₀)] = B₀;
    mf.U₀[diagind(mf.U₀)] = U₀;

    mf.∇ʸU₀[diagind(mf.∇ʸU₀)]   = ∂ʸU₀;
    mf.∇ᶻU₀[diagind(mf.∇ᶻU₀)]   = ∂ᶻU₀;

    mf.∇ʸB₀[diagind(mf.∇ʸB₀)]   = ∂ʸB₀;
    mf.∇ᶻB₀[diagind(mf.∇ᶻB₀)]   = ∂ᶻB₀;

    mf.∇ʸʸU₀[diagind(mf.∇ʸʸU₀)] = ∂ʸʸU₀;
    mf.∇ᶻᶻU₀[diagind(mf.∇ᶻᶻU₀)] = ∂ᶻᶻU₀;
    mf.∇ʸᶻU₀[diagind(mf.∇ʸᶻU₀)] = ∂ʸᶻU₀;

    return nothing
end

function construct_matrices(Op, mf, params)
    N  = params.Ny * params.Nz
    I⁰ = sparse(Matrix(1.0I, N, N)) #Eye{Float64}(N)
    s₁ = size(I⁰, 1); s₂ = size(I⁰, 2)

    # allocating memory for the LHS and RHS matrices
    𝓛₁ = SparseMatrixCSC(Zeros{ComplexF64}(s₁, 3s₂))
    𝓛₂ = SparseMatrixCSC(Zeros{ComplexF64}(s₁, 3s₂))
    𝓛₃ = SparseMatrixCSC(Zeros{ComplexF64}(s₁, 3s₂))

    ℳ₁ = SparseMatrixCSC(Zeros{Float64}(s₁, 3s₂))
    ℳ₂ = SparseMatrixCSC(Zeros{Float64}(s₁, 3s₂))
    ℳ₃ = SparseMatrixCSC(Zeros{Float64}(s₁, 3s₂))

    @printf "Start constructing matrices \n"
    # -------------------- construct matrix  ------------------------
    # lhs of the matrix (size := 3 × 3)
    # eigenvectors: [uᶻ ωᶻ b]ᵀ
    """
        inverse of the horizontal Laplacian: 
        ∇ₕ² ≡ ∂xx + ∂yy 
        H = (∇ₕ²)⁻¹
        Two methods have been implemented here:
        Method 1: SVD 
        Method 2: QR decomposition 
        Note - Method 2 is probably the `best' option 
                if the matrix, ∇ₕ², is close singular.
    """
    ∇ₕ² = SparseMatrixCSC(Zeros(N, N))
    ∇ₕ² = (1.0 * Op.𝒟²ʸ - 1.0 * params.kₓ^2 * I⁰)

    # Method 1. SVD decmposition 
    # U, Σ, V = svd(∇ₕ²); 
    # H = sparse(V * inv(Diagonal(Σ)) * transpose(U))

    # Method 2. QR decomposition
    Qm, Rm = qr(∇ₕ²)
    invR   = inv(Rm) 
    Qm     = sparse(Qm) # by sparsing the matrix speeds up matrix-matrix multiplication 
    Qᵀ     = transpose(Qm)
    H      = (invR * Qᵀ)

    # difference in L2-norm should be small: ∇ₕ² * (∇ₕ²)⁻¹ - I⁰ ≈ 0 
    @assert norm(∇ₕ² * H - I⁰) ≤ 1.0e-6 "difference in L2-norm should be small"
    #@printf "||∇ₕ² * (∇ₕ²)⁻¹ - I||₂ =  %f \n" norm(∇ₕ² * H - I⁰) 

    D⁴  = (1.0 * Op.𝒟⁴ʸ 
        + 1.0/params.ε^4 * Op.𝒟⁴ᶻᴰ 
        + 1.0params.kₓ^4 * I⁰ 
        - 2.0params.kₓ^2 * Op.𝒟²ʸ 
        - 2.0/params.ε^2 * params.kₓ^2 * Op.𝒟²ᶻᴰ
        + 2.0/params.ε^2 * Op.𝒟²ʸ²ᶻᴰ)
        
    D²  = (1.0/params.ε^2 * Op.𝒟²ᶻᴰ + 1.0 * ∇ₕ²)
    Dₙ² = (1.0/params.ε^2 * Op.𝒟²ᶻᴺ + 1.0 * ∇ₕ²)

    #* 1. uᶻ equation (bcs: uᶻ = ∂ᶻᶻuᶻ = 0 @ z = 0, 1)
    𝓛₁[:,    1:1s₂] = (-1.0params.E * D⁴
                    + 1.0im * params.kₓ * mf.U₀ * D²
                    + 1.0im * params.kₓ * (mf.∇ʸʸU₀ - 1.0/params.ε^2 * mf.∇ᶻᶻU₀) * I⁰
                    + 2.0im * params.kₓ * mf.∇ʸU₀ * Op.𝒟ʸ
                    + 2.0im/params.ε^2 * params.kₓ * mf.∇ʸU₀ * H * Op.𝒟ʸ²ᶻᴰ
                    + 2.0im/params.ε^2 * params.kₓ * mf.∇ʸᶻU₀ * H * Op.𝒟ʸᶻᴰ) * params.ε^2 

    𝓛₁[:,1s₂+1:2s₂] = (1.0/params.ε^2 * Op.𝒟ᶻᴺ 
                    + 2.0/params.ε^2 * params.kₓ^2 * mf.∇ʸU₀ * H * Op.𝒟ᶻᴺ
                    + 2.0/params.ε^2 * params.kₓ^2 * mf.∇ʸᶻU₀ * H * I⁰) * params.ε^2 
                    
    𝓛₁[:,2s₂+1:3s₂] = -1.0 * ∇ₕ²

    #* 2. ωᶻ equation (bcs: ∂ᶻωᶻ = 0 @ z = 0, 1)
    𝓛₂[:,    1:1s₂] = (1.0 * mf.∇ʸU₀ * Op.𝒟ᶻᴰ
                    - 1.0 * mf.∇ᶻU₀ * Op.𝒟ʸ
                    - 1.0 * mf.∇ʸᶻU₀ * I⁰
                    - 1.0 * Op.𝒟ᶻᴰ 
                    + 1.0 * mf.∇ʸʸU₀ * H * Op.𝒟ʸᶻᴰ)

    𝓛₂[:,1s₂+1:2s₂] = (1.0im * params.kₓ * mf.U₀ * I⁰
                    - 1.0im * params.kₓ * mf.∇ʸʸU₀ * H * I⁰
                    - 1.0params.E * Dₙ²)

    #𝓛₂[:,2s₂+1:3s₂] = 0.0 * I⁰        

    #* 3. b equation (bcs: b = 0 @ z = 0, 1)
    𝓛₃[:,    1:1s₂] = (1.0 * mf.∇ᶻB₀ * I⁰
                    - 1.0 * mf.∇ʸB₀ * H * Op.𝒟ʸᶻᴰ) 

    𝓛₃[:,1s₂+1:2s₂] = 1.0im * params.kₓ * mf.∇ʸB₀ * H * I⁰

    𝓛₃[:,2s₂+1:3s₂] = (-1.0params.E * Dₙ² 
                    + 1.0im * params.kₓ * mf.U₀ * I⁰) 

    𝓛 = ([𝓛₁; 𝓛₂; 𝓛₃]);
##############
    # [uz, wz, b] ~ [uz, wz, b] exp(σt), growth rate = real(σ)
    cnst = -1.0 #1.0im #* params.kₓ
    ℳ₁[:,    1:1s₂] = 1.0cnst * params.ε^2 * D²;
    ℳ₂[:,1s₂+1:2s₂] = 1.0cnst * I⁰;
    ℳ₃[:,2s₂+1:3s₂] = 1.0cnst * I⁰;
    ℳ = ([ℳ₁; ℳ₂; ℳ₃])
    
    #@. 𝓛 *= 1.0/params.kₓ 
    return 𝓛, ℳ
end

"""
Parameters:
"""
@with_kw mutable struct Params{T<:Real} @deftype T
    L::T        = 4.0          # horizontal domain size
    H::T        = 1.0          # vertical domain size
    ε::T        = 0.1          # front strength Γ ≡ M²/f² = λ/H = 1/ε → ε = 1/Γ
    β::T        = 2.0          # steepness of the initial buoyancy profile
    kₓ::T       = 0.0          # x-wavenumber
    E::T        = 1.0e-8      # Ekman number 
    Ny::Int64   = 120           # no. of y-grid points
    Nz::Int64   = 36           # no. of z-grid points
    order_accuracy::Int = 4
    z_discret::String = "cheb"   # option: "cheb", "fdm"
    #method::String   = "feast"
    #method::String    = "shift_invert"
    method::String    = "krylov"
    #method::String   = "arnoldi"
    #method::String   = "JacobiDavidson"
end

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

function EigSolver(Op, mf, params, emid, ra, x₀, σ, λ₀, it)
    printstyled("($it) kₓ: $(params.kₓ) \n"; color=:blue)

    𝓛, ℳ = construct_matrices(Op, mf, params)
    
    N = params.Ny * params.Nz 
    MatrixSize = 3N
    @assert size(𝓛, 1)  == MatrixSize && 
            size(𝓛, 2)  == MatrixSize &&
            size(ℳ, 1)  == MatrixSize &&
            size(ℳ, 2)  == MatrixSize "matrix size does not match!"

    # mmwrite("beta_2/systemA_" * string(trunc(Int32, it)) * ".mtx", 𝓛)
    # mmwrite("beta_2/systemB_" * string(trunc(Int32, it)) * ".mtx", ℳ)

    # file = matopen("Amat.mat", "w"); write(file, "A", 𝓛); close(file);
    # file = matopen("Bmat.mat", "w"); write(file, "B", ℳ); close(file);

    if params.method == "feast"
        nc    = 25          # number of contour points
        ε     = 1.0e-5      # tolerance
        maxit = 100         # maximum FEAST iterations
        printstyled("Eigensolver using FEAST ...\n"; color=:red)
        λₛ, Χ = feast_linear(𝓛, ℳ, x₀, nc, emid, ra, ra, ε, ra, 1e6+1e6im, maxit)

        contour    = circular_contour_trapezoidal(emid, ra, 10)
        #λₛ, Χ, res = gen_feast!(x₀, 𝓛, ℳ, contour, iter=maxit, debug=true, ϵ=ε)

    elseif params.method == "shift_invert"
        printstyled("Eigensolver using Arpack eigs with shift and invert method ...\n"; 
                    color=:red)

        λₛ = EigSolver_shift_invert( 𝓛, ℳ, σ₀=σ)
        @printf "found eigenvalue (at first): %f + im %f \n" λₛ[1].re λₛ[1].im

        λₛ = EigSolver_shift_invert_arpack_checking(𝓛, ℳ, σ₀=λₛ[1], α=0.08)
        λₛ = EigSolver_shift_invert_arpack_checking(𝓛, ℳ, σ₀=λₛ[1], α=0.04)
        λₛ = EigSolver_shift_invert_arpack_checking(𝓛, ℳ, σ₀=λₛ[1], α=0.02)
        #λₛ = EigSolver_shift_invert_arpack_checking(𝓛, ℳ, σ₀=λₛ[1], α=0.01)

        println(λₛ)
        print_evals(λₛ, length(λₛ))

    elseif params.method == "krylov"
        printstyled("KrylovKit Method ... \n"; color=:red)

        λₛ = EigSolver_shift_invert_krylov( 𝓛, ℳ, σ₀=σ)
        @printf "found eigenvalue (at first): %f + im %f \n" λₛ[1].re λₛ[1].im

        # λₛ = EigSolver_shift_invert_krylov_checking(𝓛, ℳ, σ₀=λₛ[1],   α=0.08)
        # #λₛ = EigSolver_shift_invert_krylov_checking(𝓛, ℳ, σ₀=λₛ[1],   α=0.02)
        # λₛ = EigSolver_shift_invert_krylov_checking(𝓛, ℳ, σ₀=λₛ[1],   α=1e-2)
        # # λₛ = EigSolver_shift_invert_krylov_checking(𝓛, ℳ, σ₀=λₛ[1],   α=1e-3)
        # # λₛ = EigSolver_shift_invert_krylov_checking(𝓛, ℳ, σ₀=λₛ[1],   α=1e-4)

        println(λₛ)
        print_evals(λₛ, length(λₛ))

    elseif params.method == "arnoldi"
        printstyled("Arnoldi: based on Implicitly Restarted Arnoldi Method ... \n"; 
                        color=:red)

        λₛ = EigSolver_shift_invert_arnoldi( 𝓛, ℳ, σ=σ)

        λₛ = EigSolver_shift_invert_arnoldi_checking(𝓛, ℳ, σ₀=λₛ[1].re, α=0.08)
        @printf "found eigenvalue (α=0.08): %f + im %f \n" λₛ[1].re λₛ[1].im

        λₛ = EigSolver_shift_invert_arnoldi_checking(𝓛, ℳ, σ₀=λₛ[1].re, α=0.04)
        @printf "found eigenvalue (α=0.04): %f + im %f \n" λₛ[1].re λₛ[1].im

        λₛ = EigSolver_shift_invert_arnoldi_checking(𝓛, ℳ, σ₀=λₛ[1].re, α=0.02)
        @printf "found eigenvalue (α=0.02): %f + im %f \n" λₛ[1].re λₛ[1].im

        println(λₛ)
        print_evals(λₛ, length(λₛ))
    else
        printstyled("Jacobi Davidson method... \n"; color=:red)
        target = Near(σ + 0.0im)
        pschur, residuals = jdqz(
                                𝓛, ℳ,
                                solver = GMRES(size(𝓛, 1), iterations = 7),
                                target = target,
                                pairs  = 1,
                                subspace_dimensions = 10:15,
                                max_iter = 300,
                                verbosity = 1)
      
        # The eigenvalues found by Jacobi-Davidson
        λₛ = pschur.alphas ./ pschur.betas
    end
    # ======================================================================
    @assert length(λₛ) > 0 "No eigenvalue(s) found!"

    # Post Process egenvalues
    #λₛ, Χ = remove_evals(λₛ, Χ, 0.0, 10.0, "M") # `R`: real part of λₛ.
    #λₛ, Χ = sort_evals(λₛ, Χ, "R")   
    
    #λₛ = sort_evals_(λₛ, "R")

    #= 
        this removes any further spurious eigenvalues based on norm 
        if you don't need it, just `comment' it!
    =#
    # while norm(𝓛 * Χ[:,1] - λₛ[1]/cnst * ℳ * Χ[:,1]) > 8e-2 # || imag(λₛ[1]) > 0
    #     @printf "norm (inside while): %f \n" norm(𝓛 * Χ[:,1] - λₛ[1]/cnst * ℳ * Χ[:,1]) 
    #     λₛ, Χ = remove_spurious(λₛ, Χ)
    # end
   
    #@printf "norm: %f \n" norm(𝓛 * Χ[:,1] - λₛ[1] * ℳ * Χ[:,1])
    
    #print_evals(λₛ, length(λₛ))
    @printf "largest growth rate : %1.4e%+1.4eim\n" real(λₛ[1]) imag(λₛ[1])

    𝓛 = nothing
    ℳ = nothing

    #return nothing #
    return λₛ[1] #, Χ[:,1]
end

function search_complexregion(Δkₓ, radi, serach_region₋₁, serach_region₋₂)
    ∂λ_∂kₓ = (serach_region₋₁ - serach_region₋₂) / Δkₓ
    serach_region = serach_region₋₁ + ∂λ_∂kₓ * Δkₓ
    return serach_region, max(max(radi, ∂λ_∂kₓ.re * Δkₓ), serach_region.re) 
    #max(radi, 0.75serach_region.re)
end

function solve_Ou1984()
    params      = Params{Float64}(kₓ=0.5)
    grid        = TwoDimGrid{params.Ny,  params.Nz}()
    diffMatrix  = ChebMarix{ params.Ny,  params.Nz}()
    Op          = Operator{params.Ny * params.Nz}()
    mf          = MeanFlow{params.Ny * params.Nz}()
    Construct_DerivativeOperator!(diffMatrix, grid, params)
    if params.z_discret == "cheb"
        ImplementBCs_cheb!(Op, diffMatrix, params)
    elseif params.z_discret == "fdm"
        ImplementBCs_fdm!( Op, diffMatrix, grid, params)
    else
        error("Invalid discretization type!")
    end

    @. grid.y += -0.5params.L;
    BasicState!(diffMatrix, mf, grid, params)
    N = params.Ny * params.Nz
    MatSize = Int(3N)

    @printf "β: %f \n" params.β
    @printf "ε: %f \n" params.ε
    @printf "E: %1.1e \n" params.E
    @printf "min/max of U: %f %f \n" minimum(mf.U₀ ) maximum(mf.U₀ )
    @printf "min/max of y: %f %f \n" minimum(grid.y) maximum(grid.y)
    @printf "no of y and z grid points: %i %i \n" params.Ny params.Nz
    
    kₓ  = range(0.01, stop=6.0, length=150) |> collect
    #kₓ  = range(3.54, stop=6.0, length=50) |> collect
    Δkₓ = kₓ[2] - kₓ[1]

    # file = jldopen("eigenvals_beta2.0_ep0.1_50120_E1e-8.jld2", "r");
	# kₓ   = file["kₓ"];   
	# λ₂   = file["λₛ"];
	# close(file)
    # Δkₓ = kₓ[2] - kₓ[1]

    @printf "total number of kₓ: %d \n" length(kₓ)
    λₛ  = zeros(ComplexF64, length(kₓ))
    
    m₀   = 30 #40 #100          #subspace dimension  
    ra   = 0.002 
    ra₀  = ra
    emid = complex(ra, ra)
    x₀   = sprand(ComplexF64, MatSize, m₀, 0.2) 
    for it in 1:length(kₓ)
        params.kₓ = kₓ[it] 
        
        if it == 1
            @time λₛ[it] = EigSolver(Op, mf, params, emid, 1ra, x₀, ra, λₛ, it)
            #@time λₛ[it], Χ = EigSolver(Op, mf, params, emid, ra, x₀, λ₂[it].re, it)
        else
            ra   = 0.04
            emid = complex(ra, -0.98ra)
            ra₀ = ra
            x₀ = sprand(ComplexF64, MatSize, m₀, 0.2) 
            # if it > 2 && abs(λₛ[it-1].re) ≥ 1ra 
            #     emid, ra = search_complexregion(Δkₓ, ra₀, λₛ[it-1],  λₛ[it-2])
            #     ra   = max(0.75emid.re, ra₀) 
            #     emid = complex(emid.re, -0.98ra)
            # else 
            #     if abs(λₛ[it-1].re) ≥ 1ra 
            #         emid = complex(λₛ[it-1].re, λₛ[it-1].im)
            #     else
            #         emid = complex(ra, -0.98ra) 
            #     end
            # end
            # println("$emid ($ra)")
            if it == 2
                @time λₛ[it] = EigSolver(Op, mf, params, emid, ra, x₀, λₛ[it-1].re, λₛ, it)
                #@time λₛ[it], Χ = EigSolver(Op, mf, params, emid, ra, x₀, λ₂[it].re, it)
            else
                @time λₛ[it] = EigSolver(Op, mf, params, emid, ra, x₀, λₛ[it-1].re, λₛ, it)
                #@time λₛ[it], Χ = EigSolver(Op, mf, params, emid, ra, x₀, λ₂[it].re, it)
            end
        end
        # println("==================================================================")
        # Ny::Int = params.Ny
        # Nz::Int = params.Nz 
        # jldsave("nw_eigenfun_beta1.0_ep0.1" * "_" * string(Nz) * string(Ny) * ".jld2";  
        #                                     y=grid.y, z=grid.z, 
        #                                     kₓ=params.kₓ, λₛ=λₛ[1], 
        #                                     X=Χ, U=diag(mf.U₀), B=diag(mf.B₀));
    end

    β  = params.β
    ε  = params.ε
    Ny::Int = params.Ny
    Nz::Int = params.Nz 
    filename = "nw_run/eigenvals_beta" * string(β) * "_ep" * string(ε) * "_" * string(Nz) * string(Ny) * ".jld2"
    jldsave(filename; kₓ=kₓ, λₛ=λₛ)
end

solve_Ou1984()

