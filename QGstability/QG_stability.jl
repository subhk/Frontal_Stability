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

include("FEASTSolver/src/FEASTSolver.jl")
using Main.FEASTSolver

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

    𝒟ᶻ::Array{Float64,  2}   = SparseMatrixCSC(Zeros(N, N))
    𝒟²ᶻ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(N, N))

    𝒟ᶻᴺ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟²ᶻᴺ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟⁴ᶻᴺ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))

    𝒟ᶻᴰ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟ʸᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟²ᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟⁴ᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))
end

@with_kw mutable struct MeanFlow{N} 
    B₀::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))
    U₀::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))

  ∇ʸU₀::Array{Float64, 2}   = SparseMatrixCSC(Zeros(N, N))
  ∇ᶻU₀::Array{Float64, 2}   = SparseMatrixCSC(Zeros(N, N))
  
  ∇ᶻB₀⁻¹::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))
  ∇ᶻB₀⁻²::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))

  ∇ʸQ₀::Array{Float64, 2}   = SparseMatrixCSC(Zeros(N, N))

  ∇ᶻᶻB₀::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))
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
        
        @printf "size of Chebyshev matrix: %d × %d \n" size(diffMatrix.𝒟ᶻ)[1]  size(diffMatrix.𝒟ᶻ)[2]

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
    diffMatrix.𝒟ᶻᴰ  = deepcopy( diffMatrix.𝒟ᶻ  ) 
    diffMatrix.𝒟²ᶻᴰ = deepcopy( diffMatrix.𝒟²ᶻ )
    diffMatrix.𝒟⁴ᶻᴰ = deepcopy( diffMatrix.𝒟⁴ᶻ )

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
    diffMatrix.𝒟ᶻᴺ  = deepcopy( diffMatrix.𝒟ᶻ  )
    diffMatrix.𝒟²ᶻᴺ = deepcopy( diffMatrix.𝒟²ᶻ )
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

    kron!( Op.𝒟ᶻᴺ  ,  Iʸ , diffMatrix.𝒟ᶻᴺ  )
    kron!( Op.𝒟²ᶻᴺ ,  Iʸ , diffMatrix.𝒟²ᶻᴺ )

    kron!( Op.𝒟ʸ   ,  diffMatrix.𝒟ʸ  ,  Iᶻ ) 
    kron!( Op.𝒟²ʸ  ,  diffMatrix.𝒟²ʸ ,  Iᶻ )

    kron!( Op.𝒟ᶻ   ,  Iʸ , diffMatrix.𝒟ᶻ   )
    kron!( Op.𝒟²ᶻ  ,  Iʸ , diffMatrix.𝒟²ᶻ  )

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
    ∂ᶻᶻU₀ = similar(B₀)
    ∂ᶻᶻB₀ = similar(B₀)

    """
    Calculating necessary derivatives of the mean-flow quantities
    """
    ∂ʸB₀   = gradient(  B₀,  grid.y, dims=1)
    ∂ʸU₀   = gradient(  U₀,  grid.y, dims=1)
    ∂ʸʸU₀  = gradient2( U₀,  grid.y, dims=1)

    # `Thermal wind balance'
    @. ∂ᶻU₀  = -1.0 * ∂ʸB₀

    for iy ∈ 1:length(grid.y)
         ∂ᶻB₀[iy,:] = diffMatrix.𝒟ᶻ * B₀[iy,:]
        ∂ᶻᶻU₀[iy,:] = diffMatrix.𝒟ᶻ * ∂ᶻU₀[iy,:]
        ∂ᶻᶻB₀[iy,:] = diffMatrix.𝒟ᶻ * ∂ᶻB₀[iy,:]
    end

    max_  = maximum(∂ᶻB₀)
    @assert max_ > 0.0
    @. ∂ᶻB₀ += 0.0001max_
    cnst  = @. 1.0/∂ᶻB₀ 
    cnst2 = @. cnst * cnst

    max_  = maximum(∂ᶻᶻB₀)
    @assert max_ > 0.0
    @. ∂ᶻᶻB₀ += 0.0001max_

    ∂ʸQ₀ = @. -1.0 * ∂ʸʸU₀ - (1.0 * ∂ᶻB₀ * ∂ᶻᶻU₀ - 1.0 * ∂ᶻU₀ * ∂ᶻᶻB₀) * cnst2 

    @printf "min/max values of ∂ᶻU₀: %f %f \n" minimum(∂ᶻU₀) maximum(∂ᶻU₀)
    @printf "min/max values of ∂ʸU₀: %f %f \n" minimum(∂ʸU₀) maximum(∂ʸU₀)
    @printf "min/max values of ∂ᶻB₀: %f %f \n" minimum(∂ᶻB₀) maximum(∂ᶻB₀)
    @printf "min/max values of ∂ʸB₀: %f %f \n" minimum(∂ʸB₀) maximum(∂ʸB₀)

    @printf "min/max values of ∂ᶻᶻU₀: %f %f \n" minimum(∂ᶻᶻU₀) maximum(∂ᶻᶻU₀)
    @printf "min/max values of ∂ᶻᶻB₀: %f %f \n" minimum(∂ᶻᶻB₀) maximum(∂ᶻᶻB₀)

    @printf "min/max values of ∂ʸQ₀: %f %f \n" minimum(∂ʸQ₀) maximum(∂ʸQ₀)

    ∂ᶻB₀⁻¹ = @. 1.0/∂ᶻB₀ 
    ∂ᶻB₀⁻² = @. 1.0/(∂ᶻB₀ * ∂ᶻB₀) 

    B₀    = transpose(B₀);       B₀ = B₀[:];
    U₀    = transpose(U₀);       U₀ = U₀[:];

    ∂ʸB₀  = transpose(∂ʸB₀);   ∂ʸB₀ = ∂ʸB₀[:];
    ∂ʸU₀  = transpose(∂ʸU₀);   ∂ʸU₀ = ∂ʸU₀[:];
   
    ∂ʸQ₀  = transpose(∂ʸQ₀);   ∂ʸQ₀ = ∂ʸQ₀[:];

    ∂ᶻB₀  = transpose(∂ᶻB₀);   ∂ᶻB₀ = ∂ᶻB₀[:];
    ∂ᶻU₀  = transpose(∂ᶻU₀);   ∂ᶻU₀ = ∂ᶻU₀[:];

    ∂ᶻB₀⁻¹ = transpose(∂ᶻB₀⁻¹); ∂ᶻB₀⁻¹ = ∂ᶻB₀⁻¹[:];
    ∂ᶻB₀⁻² = transpose(∂ᶻB₀⁻²); ∂ᶻB₀⁻² = ∂ᶻB₀⁻²[:];

    ∂ᶻᶻB₀  = transpose(∂ᶻᶻB₀); ∂ᶻᶻB₀   = ∂ᶻᶻB₀[:];


    mf.B₀[diagind(mf.B₀)] = B₀;
    mf.U₀[diagind(mf.U₀)] = U₀;

    mf.∇ʸU₀[diagind(mf.∇ʸU₀)]   = ∂ʸU₀;
    mf.∇ᶻU₀[diagind(mf.∇ᶻU₀)]   = ∂ᶻU₀;

    #mf.∇ʸB₀[diagind(mf.∇ʸB₀)]   = ∂ʸB₀;
    #mf.∇ᶻB₀[diagind(mf.∇ᶻB₀)]   = ∂ᶻB₀;

    mf.∇ʸQ₀[diagind(mf.∇ʸQ₀)]   = ∂ʸQ₀;

    mf.∇ᶻB₀⁻¹[diagind(mf.∇ᶻB₀⁻¹)] = ∂ᶻB₀⁻¹
    mf.∇ᶻB₀⁻²[diagind(mf.∇ᶻB₀⁻²)] = ∂ᶻB₀⁻²

    mf.∇ᶻᶻB₀[diagind(mf.∇ᶻᶻB₀)] = ∂ᶻᶻB₀;

    return nothing
end

function construct_matrices(Op, mf, params)
    N  = params.Ny * params.Nz
    I⁰ = sparse(Matrix(1.0I, N, N)) #Eye{Float64}(N)
    s₁ = size(I⁰, 1); s₂ = size(I⁰, 2)

    # allocating memory for the LHS and RHS matrices
    𝓛 = SparseMatrixCSC(Zeros{ComplexF64}(s₁, s₂))
    ℳ = SparseMatrixCSC(Zeros{ Float64  }(s₁, s₂))

    B = SparseMatrixCSC(Zeros{ComplexF64}(s₁, s₂));
    C = SparseMatrixCSC(Zeros{ Float64  }(s₁, s₂));

    @printf "Start constructing matrices \n"
    # -------------------- construct matrix  ------------------------
    # lhs of the matrix (size := 2 × 2)
    # eigenvectors: [ψ q]ᵀ
    ∇ₕ² = SparseMatrixCSC(Zeros{Float64}(N, N))
    ∇ₕ² = (1.0 * Op.𝒟²ʸ - 1.0 * params.kₓ^2 * I⁰)

    # definition of perturbation PV, q = D₂³ᵈ{ψ}
    D₂³ᵈ = (1.0 * ∇ₕ²
            + 1.0  * mf.∇ᶻB₀⁻¹  * Op.𝒟²ᶻ
            - 1.0  * mf.∇ᶻᶻB₀ * mf.∇ᶻB₀⁻² * Op.𝒟ᶻ)

    #* 1. ψ equation
    𝓛[:,1:1s₂] = (1.0im * params.kₓ * mf.U₀ * D₂³ᵈ
                + 1.0im * params.kₓ * mf.∇ʸQ₀ * I⁰
                - 1.0 * params.E * D₂³ᵈ)

##############
    # [ψ] = Re {[ψ♯] exp(σt)}, growth rate = real(σ)
    ℳ[:,1:1s₂] = -1.0 * D₂³ᵈ;

    ###
    # stuff required for implementing boundary conditions 
    ###
    _, zi  = ndgrid(1:1:params.Ny, 1:1:params.Nz)
    zi     = transpose(zi);
    zi     = zi[:];
    bcᶻᵇ   = findall( x -> (x==1),         zi );
    bcᶻᵗ   = findall( x -> (x==params.Nz), zi );

    ###
    # Implementing boundary condition for 𝓛 matrix in the z-direction: 
    ###
    #fill!(B, 0.0); #B = sparse(B); 
    B[:,1:1s₂] = 1.0im * params.kₓ * mf.U₀ * Op.𝒟ᶻ - 1.0im * params.kₓ * mf.∇ᶻU₀ * I⁰; 
    # Bottom boundary condition @ z=0  
    𝓛[bcᶻᵇ, :] = B[bcᶻᵇ, :]
    # Top boundary condition @ z = 1
    𝓛[bcᶻᵗ, :] = B[bcᶻᵗ, :]

    ###
    # Implementing boundary condition for ℳ matrix in the z-direction: 
    ###
    #fill!(C, 0.0); #C = sparse(C); 
    C[:,1:1s₂] = -1.0 * Op.𝒟ᶻ; 
    # Bottom boundary condition @ z=0  
    ℳ[bcᶻᵇ, :] = C[bcᶻᵇ, :]
    # Top boundary condition @ z = 1
    ℳ[bcᶻᵗ, :] = C[bcᶻᵗ, :]

    return 𝓛, ℳ
end

"""
Parameters:
"""
@with_kw mutable struct Params{T<:Real} @deftype T
    L::T        = 12.0          # horizontal domain size
    H::T        = 1.0          # vertical domain size
    #ε::T        = 0.1          # front strength Γ ≡ M²/f² = λ/H = 1/ε → ε = 1/Γ
    β::T        = 0.1          # steepness of the initial buoyancy profile
    kₓ::T       = 0.0          # x-wavenumber
    E::T        = 1.0e-16      # Ekman number 
    Ny::Int64   = 96           # no. of y-grid points
    Nz::Int64   = 20           # no. of z-grid points
    order_accuracy::Int = 4
    z_discret::String = "cheb"   # option: "cheb", "fdm"
    #method::String    = "feast"
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
    MatrixSize = 1N
    @assert size(𝓛, 1)  == MatrixSize && 
            size(𝓛, 2)  == MatrixSize &&
            size(ℳ, 1)  == MatrixSize &&
            size(ℳ, 2)  == MatrixSize "matrix size does not match!"

    if params.method == "feast"
        nc    = 15          # number of contour points
        ε     = 1.0e-5      # tolerance
        maxit = 100         # maximum FEAST iterations
        printstyled("Eigensolver using FEAST ...\n"; color=:red)
        #λₛ, Χ = feast_linear(𝓛, ℳ, x₀, nc, emid, ra, ra, ε, ra, 1e6+1e6im, maxit)

        contour    = circular_contour_trapezoidal(emid, ra, 10)
        λₛ, Χ, res = gen_feast!(x₀, 𝓛, ℳ, contour, iter=maxit, debug=true, ϵ=ε)

    elseif params.method == "shift_invert"
        printstyled("Eigensolver using Arpack eigs with shift and invert method ...\n"; 
                    color=:red)
        # if it > 5
        #     if λ₀[it-1].re > λ₀[it-2].re
        #         λₛ, Χ = EigSolver_shift_invert( 𝓛, ℳ, σ₀=σ₀)
        #         @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
        #     else
        #         λₛ, Χ = EigSolver_shift_invert_2( 𝓛, ℳ, σ₀=σ₀)
        #         @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
        #     end
        # else
        #     λₛ, Χ = EigSolver_shift_invert( 𝓛, ℳ, σ₀=σ₀)
        #     @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
        # end
        
        #λₛ, Χ = EigSolver_shift_invert1(𝓛, ℳ, σ₀=λₛ[1].re)

        λₛ = EigSolver_shift_invert( 𝓛, ℳ, σ₀=σ)
        @printf "found eigenvalue (at first): %f + im %f \n" λₛ[1].re λₛ[1].im

        λₛ = EigSolver_shift_invert_arpack_checking(𝓛, ℳ, σ₀=λₛ[1], α=0.08)
        λₛ = EigSolver_shift_invert_arpack_checking(𝓛, ℳ, σ₀=λₛ[1], α=0.04)
        λₛ = EigSolver_shift_invert_arpack_checking(𝓛, ℳ, σ₀=λₛ[1], α=0.02)
        λₛ = EigSolver_shift_invert_arpack_checking(𝓛, ℳ, σ₀=λₛ[1], α=0.01)

        println(λₛ)
        print_evals(λₛ, length(λₛ))

    elseif params.method == "krylov"
        printstyled("KrylovKit Method ... \n"; color=:red)

        λₛ = EigSolver_shift_invert_krylov( 𝓛, ℳ, σ₀=σ)
        @printf "found eigenvalue (at first): %f + im %f \n" λₛ[1].re λₛ[1].im

        #λₛ = EigSolver_shift_invert_krylov_checking(𝓛, ℳ, σ₀=λₛ[1],   α=0.08)
        #λₛ = EigSolver_shift_invert_krylov_checking(𝓛, ℳ, σ₀=λₛ[1],   α=0.02)
        #λₛ = EigSolver_shift_invert_krylov_checking(𝓛, ℳ, σ₀=λₛ[1],   α=0.01)
        # # λₛ = EigSolver_shift_invert_krylov_checking(𝓛, ℳ, σ₀=λₛ[1],   α=1e-3)
        # # λₛ = EigSolver_shift_invert_krylov_checking(𝓛, ℳ, σ₀=λₛ[1],   α=1e-4)

        println(λₛ)
        print_evals(λₛ, length(λₛ))

    elseif params.method == "arnoldi"
        printstyled("Arnoldi: based on Implicitly Restarted Arnoldi Method ... \n"; 
                        color=:red)
        # if it > 5
        #     if λ₀[it-1].re > λ₀[it-2].re
        #         λₛ = EigSolver_shift_invert_arnoldi( 𝓛, ℳ, σ₀=σ₀)
        #         @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
        #     else
        #         λₛ = EigSolver_shift_invert_arnoldi( 𝓛, ℳ, σ₀=σ₀)
        #         @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
        #     end
        # else
        #     λₛ = EigSolver_shift_invert_arnoldi( 𝓛, ℳ, σ₀=σ₀)
        #     @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
        # end
        
        #λₛ = EigSolver_shift_invert_arnoldi1(𝓛, ℳ, σ₀=λₛ[1].re)

        λₛ = EigSolver_shift_invert_arnoldi( 𝓛, ℳ, σ₀=σ)

        λₛ = EigSolver_shift_invert_arnoldi_checking(𝓛, ℳ, σ₀=λₛ[1].re, α=0.08)
        @printf "found eigenvalue (α=0.08): %f + im %f \n" λₛ[1].re λₛ[1].im

        λₛ = EigSolver_shift_invert_arnoldi_checking(𝓛, ℳ, σ₀=λₛ[1].re, α=0.04)
        @printf "found eigenvalue (α=0.04): %f + im %f \n" λₛ[1].re λₛ[1].im

        λₛ = EigSolver_shift_invert_arnoldi_checking(𝓛, ℳ, σ₀=λₛ[1].re, α=0.02)
        @printf "found eigenvalue (α=0.02): %f + im %f \n" λₛ[1].re λₛ[1].im

        λₛ = EigSolver_shift_invert_arnoldi_checking(𝓛, ℳ, σ₀=λₛ[1].re, α=0.01)
        @printf "found eigenvalue (α=0.01): %f + im %f \n" λₛ[1].re λₛ[1].im

        λₛ = EigSolver_shift_invert_arnoldi_checking(𝓛, ℳ, σ₀=λₛ[1].re, α=0.005)
        @printf "found eigenvalue (α=0.005): %f + im %f \n" λₛ[1].re λₛ[1].im

        λₛ = EigSolver_shift_invert_arnoldi_checking(𝓛, ℳ, σ₀=λₛ[1].re, α=0.002)
        @printf "found eigenvalue (α=0.002): %f + im %f \n" λₛ[1].re λₛ[1].im

        println(λₛ)
        print_evals(λₛ, length(λₛ))
    else
        printstyled("Jacobi Davidson method... \n"; color=:red)
        target = Near(σ₀ + 0.0im)
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
    #elseif params.z_discret == "fdm"
    #    ImplementBCs_fdm!( Op, diffMatrix, grid, params)
    else
        error("Invalid discretization type!")
    end

    @. grid.y += -0.5params.L;
    BasicState!(diffMatrix, mf, grid, params)
    N = params.Ny * params.Nz
    MatSize = Int(1N)

    @printf "β: %f \n" params.β
    #@printf "ε: %f \n" params.ε
    @printf "E: %1.1e \n" params.E
    @printf "min/max of U: %f %f \n" minimum(mf.U₀ ) maximum(mf.U₀ )
    @printf "min/max of y: %f %f \n" minimum(grid.y) maximum(grid.y)
    @printf "no of y and z grid points: %i %i \n" params.Ny params.Nz
    
    #kₓ  = range(0.01, stop=8.0, length=200) |> collect

    kₓ  = range(0.01, stop=50.0, length=1000) |> collect
    #kₓ  = range(80.0,  stop=100.0, length=400) |> collect
    Δkₓ = kₓ[2] - kₓ[1]

    # file = jldopen("eigenvals_beta2.0_ep0.1_50120_E1e-8.jld2", "r");
	# kₓ   = file["kₓ"];   
	# λ₂   = file["λₛ"];
	# close(file)
    # Δkₓ = kₓ[2] - kₓ[1]

    @printf "total number of kₓ: %d \n" length(kₓ)
    λₛ  = zeros(ComplexF64, length(kₓ))
    
    m₀   = 20 #40 #100          #subspace dimension  
    ra   = 1.6e-4 #0.0000001 
    ra₀  = ra
    emid = complex(ra, 1ra)
    if params.method == "feast"; println("$emid ($ra)"); end
    x₀   = sprand(ComplexF64, MatSize, m₀, 0.2) 
    for it in 1:length(kₓ)
        params.kₓ = kₓ[it] 
        
        if it == 1
            @time λₛ[it] = EigSolver(Op, mf, params, emid, 1ra, x₀, ra, λₛ, it)
            #@time λₛ[it], Χ = EigSolver(Op, mf, params, emid, ra, x₀, λ₂[it].re, it)
        else
            ra   = 0.005
            emid = complex(ra, 0.5ra)
            ra₀ = ra
            x₀ = sprand(ComplexF64, MatSize, m₀, 0.2) 
            # if it > 2 && abs(λₛ[it-1].re) ≥ 1ra 
            #     emid, ra = search_complexregion(Δkₓ, ra₀, λₛ[it-1],  λₛ[it-2])
            #     ra   = max(0.75emid.re, ra₀) 
            #     emid = complex(emid.re, 0.5ra)
            # else 
            #     if abs(λₛ[it-1].re) ≥ 1ra 
            #         ra  = λₛ[it-1].re
            #         emid = complex(λₛ[it-1].re, 0.5ra)
            #     else
            #         emid = complex(ra, 0.5ra) 
            #     end
            # end
            if params.method == "feast"; println("$emid ($ra)"); end
            if it == 2
                @time λₛ[it] = EigSolver(Op, mf, params, emid, 0.5ra, x₀, λₛ[it-1].re, λₛ, it)
                #@time λₛ[it], Χ = EigSolver(Op, mf, params, emid, ra, x₀, λ₂[it].re, it)
            else
                @time λₛ[it] = EigSolver(Op, mf, params, emid, 0.5ra, x₀, λₛ[it-1].re, λₛ, it)
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
    #ε  = params.ε
    Ny::Int = params.Ny
    Nz::Int = params.Nz 
    filename = "eigenvals_beta" * string(β) * "_" * string(Nz) * string(Ny) * ".jld2"
    jldsave(filename; kₓ=kₓ, λₛ=λₛ)
end

solve_Ou1984()

