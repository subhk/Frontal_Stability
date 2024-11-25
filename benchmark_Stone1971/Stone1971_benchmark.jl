#=
Baroclinic instability of a 2D front: Eady model (Stone 1971 JFM)
=#
using LazyGrids
using BlockArrays
using Printf
using SparseArrays
using SparseMatrixDicts
using SpecialFunctions
using FillArrays
using Parameters
using Test
using BenchmarkTools
using BasicInterpolators: BicubicInterpolator
using Interpolations
using Kronecker: ⊗

using Serialization
using Pardiso
using Arpack
using LinearMaps
using ArnoldiMethod
using Dierckx
using LinearAlgebra
using JacobiDavidson
using JLD
#using SparseMatricesCSR
using MatrixMarket
using SparseMatricesCOO

using CairoMakie
using LaTeXStrings
CairoMakie.activate!()
using DelimitedFiles
using ColorSchemes
using MAT
using IterativeSolvers
using Statistics

include("dmsuite.jl")
include("transforms.jl")
include("utils.jl")
include("setBCs.jl")

include("feast.jl")
using ..feastLinear

# include("FEASTSolver/src/FEASTSolver.jl")
# using Main.FEASTSolver


function Interp3D(xn, yn, zn, An, grid)
    itp = interpolate((xn,yn,zn), An, Gridded(Linear()))
    A₀ = zeros(Float64, length(grid.r), length(grid.z))
    A₀ = [itp(xᵢ,yᵢ,zᵢ) for xᵢ in grid.x, yᵢ in grid.y, zᵢ in grid.z]
    A₀ = transpose(A₀)
    A₀ = A₀[:]
    return A₀
end

#### plotting the eigenfunction
function Interp2D(yn, zn, An, yint, zint)
    itp = BicubicInterpolator(yn, zn, transpose(An))
    A₀ = zeros(Float64, length(yint), length(zint))
    A₀ = [itp(yᵢ, zᵢ) for yᵢ ∈ yint, zᵢ ∈ zint]
    return A₀
end

# function Interp2D(yn, zn, An, yint, zint)
#     itp = Spline2D(yn, zn, transpose(An); kx=3, ky=3, s=0.0)
#     A₀ = [evaluate(spl, yᵢ, zᵢ) for yᵢ ∈ yint, zᵢ ∈ zint]
#     return A₀
# end

function ContourPlot(y, z, uᶻ, ωᶻ, U, filename, it)
    uᶻ = reshape( uᶻ, (length(z), length(y)) )
    ωᶻ = reshape( ωᶻ, (length(z), length(y)) )
    U  = reshape( U , (length(z), length(y)) )

    levels  = LinRange(minimum(U), maximum(U), 10) 
    levels₋ = levels[findall( x -> (x ≤ 0.0), levels )]
    levels₊ = levels[findall( x -> (x > 0.0), levels )]
    
    y_interp = collect(LinRange(minimum(y), maximum(y), 1000))
    z_interp = collect(LinRange(minimum(z), maximum(z), 100) )
    
    uᶻ_interp = Interp2D(y, z, uᶻ, y_interp, z_interp)
    ωᶻ_interp = Interp2D(y, z, ωᶻ, y_interp, z_interp)
    U_interp  = Interp2D(y, z, U,  y_interp, z_interp)
    
    fig = Figure(fontsize=30, size = (1800, 500), font="Times")
    
    ax1 = Axis(fig[1, 1], xlabel=L"$y$", xlabelsize=30,
                        ylabel=L"$z$", ylabelsize=30)
    
    max_val = maximum(abs.(uᶻ))
    levels = range(-max_val, max_val, length=20)
    co = contourf!(y_interp, z_interp, uᶻ_interp, colormap=cgrad(:balance, rev=false),
        levels=levels, extendlow = :auto, extendhigh = :auto )

    contour!(y_interp, z_interp, U_interp, levels=levels₋, 
                        linestyle=:dash,  color=:grey, linewidth=3) 
    contour!(y_interp, z_interp, U_interp, levels=levels₊, 
                        linestyle=:solid, color=:grey, linewidth=3) 
    
    tightlimits!(ax1)
    xlims!(minimum(y), maximum(y))
    ylims!(minimum(z), maximum(z))

    cbar = Colorbar(fig[1, 2], co)
        
    ax2 = Axis(fig[1, 3], xlabel=L"$y$", xlabelsize=30,
                        ylabel=L"$z$", ylabelsize=30)

    max_val = maximum(abs.(ωᶻ))
    levels = range(-max_val, max_val, length=20)
    co = contourf!(y_interp, z_interp, ωᶻ_interp, colormap=cgrad(:balance, rev=false),
        levels=levels, extendlow = :auto, extendhigh = :auto )
    
    contour!(y_interp, z_interp, U_interp, levels=levels₋, 
                        linestyle=:dash,  color=:grey, linewidth=3) 
    contour!(y_interp, z_interp, U_interp, levels=levels₊, 
                        linestyle=:solid, color=:grey, linewidth=3) 

    tightlimits!(ax2)
    cbar = Colorbar(fig[1, 4], co)
    
    xlims!(minimum(y), maximum(y))
    ylims!(minimum(z), maximum(z))
    
    fig
    file = filename * string(it) * ".png"
    save(file, fig, px_per_unit=2)
end

@with_kw mutable struct TwoDimGrid{Ny, Nz} 
    y::Array{Float64} = zeros(Float64, Ny)
    z::Array{Float64} = zeros(Float64, Nz)
end

@with_kw mutable struct ChebMarix{Ny, Nz} 
    𝒟ʸ::Array{Float64,  2}   = SparseMatrixCSC(Zeros(Ny, Ny))
    𝒟²ʸ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(Ny, Ny))
    𝒟⁴ʸ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(Ny, Ny))

    𝒟ᶻ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟²ᶻ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟³ᶻ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟⁴ᶻ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))

    𝒟ᶻᴺ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟²ᶻᴺ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟⁴ᶻᴺ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))

    𝒟ᶻᴰ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(Nz, Nz))
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

    𝒟ᶻ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(N, N))
    𝒟²ᶻ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))

    𝒟ᶻᴺ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(N, N))
    𝒟²ᶻᴺ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟⁴ᶻᴺ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))

    𝒟ᶻᴰ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(N, N))
    𝒟ʸᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟²ᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟⁴ᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))

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

function ImplementBCs_Dirchilet_on_D1!(diffMatrix, grid; order_accuracy)
    N   = length(grid.z)
    del = grid.z[2] - grid.z[1]
    if order_accuracy == 4
        diffMatrix.𝒟ᶻᴰ[1,:] .= 0.0;              diffMatrix.𝒟ᶻᴰ[1,1] = -(1/12)/del;
        diffMatrix.𝒟ᶻᴰ[1,2]  = (2/3)/del;        diffMatrix.𝒟ᶻᴰ[1,3] = -(1/12)/del;

        diffMatrix.𝒟ᶻᴰ[2,:] .= 0.0;              diffMatrix.𝒟ᶻᴰ[2,1] = -(2/3)/del;
        diffMatrix.𝒟ᶻᴰ[2,2]  = 0.0;              diffMatrix.𝒟ᶻᴰ[2,3] = (2/3)/del;
        diffMatrix.𝒟ᶻᴰ[2,4]  = -(1/12)/del;

        diffMatrix.𝒟ᶻᴰ[N,:]    .= -1.0 .* diffMatrix.𝒟ᶻᴰ[1,:];               
        diffMatrix.𝒟ᶻᴰ[N-1,:]  .= -1.0 .* diffMatrix.𝒟ᶻᴰ[2,:];   
    else
        diffMatrix.𝒟ᶻᴰ[1,:] .= 0;                      
        diffMatrix.𝒟ᶻᴰ[1,2]  = 0.5/del;         
        diffMatrix.𝒟ᶻᴰ[N,:] .= -1.0 .* diffMatrix.𝒟ᶻᴰ[1,:];  
    end
end

function ImplementBCs_Dirchilet_on_D2!(diffMatrix, grid; order_accuracy)
    N   = length(grid.z)
    del = grid.z[2] - grid.z[1] 
    if order_accuracy == 4
        diffMatrix.𝒟²ᶻᴰ[1,:] .= 0;
        diffMatrix.𝒟²ᶻᴰ[1,1]  = -2/del^2;         diffMatrix.𝒟²ᶻᴰ[1,2] = 1/del^2;  

        diffMatrix.𝒟²ᶻᴰ[2,:] .= 0;                diffMatrix.𝒟²ᶻᴰ[2,1] = (4/3)/del^2; 
        diffMatrix.𝒟²ᶻᴰ[2,2]  = -(5/2)/del^2;     diffMatrix.𝒟²ᶻᴰ[2,3] = (4/3)/del^2;
        diffMatrix.𝒟²ᶻᴰ[2,4]  = -(1/12)/del^2;     

        diffMatrix.𝒟²ᶻᴰ[N,:]   .= 1.0 .* diffMatrix.𝒟²ᶻᴰ[1,:];
        diffMatrix.𝒟²ᶻᴰ[N-1,:] .= 1.0 .* diffMatrix.𝒟²ᶻᴰ[2,:];
    else
        diffMatrix.𝒟²ᶻᴰ[1,:] .= 0;
        diffMatrix.𝒟²ᶻᴰ[1,1]  = -2.0/del^2;         
        diffMatrix.𝒟²ᶻᴰ[1,2]  = 1.0/del^2;  
        diffMatrix.𝒟²ᶻᴰ[N,:] .= 1.0 .* diffMatrix.𝒟²ᶻᴰ[1,:];        
    end
end

function ImplementBCs_Dirchilet_on_D4!(diffMatrix, grid; order_accuracy)
    N   = length(grid.z)
    del = grid.z[2] - grid.z[1] 
    if order_accuracy == 4
        diffMatrix.𝒟⁴ᶻᴰ[1,:] .= 0;                  diffMatrix.𝒟⁴ᶻᴰ[1,1] = 5/del^4;
        diffMatrix.𝒟⁴ᶻᴰ[1,2]  = -4/del^4;           diffMatrix.𝒟⁴ᶻᴰ[1,3] = 1/del^4;
        
        diffMatrix.𝒟⁴ᶻᴰ[2,:] .= 0;                  diffMatrix.𝒟⁴ᶻᴰ[2,1] = -(38/6)/del^4;
        diffMatrix.𝒟⁴ᶻᴰ[2,2]  = (28/3)/del^4;       diffMatrix.𝒟⁴ᶻᴰ[2,3] = -(13/2)/del^4;
        diffMatrix.𝒟⁴ᶻᴰ[2,4]  = 2/del^4;            diffMatrix.𝒟⁴ᶻᴰ[2,5] = -(1/6)/del^4;
        
        diffMatrix.𝒟⁴ᶻᴰ[3,:] .= 0;                  diffMatrix.𝒟⁴ᶻᴰ[3,1] = 2/del^4;
        diffMatrix.𝒟⁴ᶻᴰ[3,2]  = -(13/2)/del^4;      diffMatrix.𝒟⁴ᶻᴰ[3,3] = (28/3)/del^4;
        diffMatrix.𝒟⁴ᶻᴰ[3,4]  = -(13/2)/del^4;      diffMatrix.𝒟⁴ᶻᴰ[3,5] = 2/del^4;
        diffMatrix.𝒟⁴ᶻᴰ[3,6]  = -(1/6)/del^4;
        
        diffMatrix.𝒟⁴ᶻᴰ[N,:]    .= 1.0 .* diffMatrix.𝒟⁴ᶻᴰ[1,:];
        diffMatrix.𝒟⁴ᶻᴰ[N-1,:]  .= 1.0 .* diffMatrix.𝒟⁴ᶻᴰ[2,:];
        diffMatrix.𝒟⁴ᶻᴰ[N-2,:]  .= 1.0 .* diffMatrix.𝒟⁴ᶻᴰ[3,:];
    else
        diffMatrix.𝒟⁴ᶻᴰ[1,:] .= 0;                  diffMatrix.𝒟⁴ᶻᴰ[1,1] = 5.0/del^4;
        diffMatrix.𝒟⁴ᶻᴰ[1,2]  = -4.0/del^4;         diffMatrix.𝒟⁴ᶻᴰ[1,3] = 1.0/del^4;
 
        diffMatrix.𝒟⁴ᶻᴰ[2,:]   .= 0;                diffMatrix.𝒟⁴ᶻᴰ[2,1] = -4.0/del^4;
        diffMatrix.𝒟⁴ᶻᴰ[2,2]    = 6.0/del^4;        diffMatrix.𝒟⁴ᶻᴰ[2,3] = -4.0/del^4;
        diffMatrix.𝒟⁴ᶻᴰ[2,4]    = 1.0/del^4;     
        diffMatrix.𝒟⁴ᶻᴰ[N,  :] .= 1.0 .* diffMatrix.𝒟⁴ᶻᴰ[1,:];
        diffMatrix.𝒟⁴ᶻᴰ[N-1,:] .= 1.0 .* diffMatrix.𝒟⁴ᶻᴰ[2,:];  
    end
end


function ImplementBCs_Neumann_on_D1!(diffMatrix, grid; order_accuracy)
    N   = length(grid.z)
    del = grid.z[2] - grid.z[1] 
    if order_accuracy == 4
        diffMatrix.𝒟ᶻᴺ[1,:]   .= 0;              diffMatrix.𝒟ᶻᴺ[1,1] = -1/del;
        diffMatrix.𝒟ᶻᴺ[1,2]    = 1/del;         

        diffMatrix.𝒟ᶻᴺ[2,:]   .= 0;              diffMatrix.𝒟ᶻᴺ[2,1] = -(7/12)/del;
        diffMatrix.𝒟ᶻᴺ[2,2]    = 0;              diffMatrix.𝒟ᶻᴺ[2,3] = (2/3)/del;
        diffMatrix.𝒟ᶻᴺ[2,4]    = -(1/12)/del;

        diffMatrix.𝒟ᶻᴺ[N,:]   .= -1.0 .* diffMatrix.𝒟ᶻᴺ[1,:];              
        diffMatrix.𝒟ᶻᴺ[N-1,:] .= -1.0 .* diffMatrix.𝒟ᶻᴺ[2,:];
    else
        diffMatrix.𝒟ᶻᴺ[1,:]  .= 0;              
        diffMatrix.𝒟ᶻᴺ[1,1]   = -0.5/del;
        diffMatrix.𝒟ᶻᴺ[1,2]   = 0.5/del;         
        diffMatrix.𝒟ᶻᴺ[N,:]  .= -1.0 .* diffMatrix.𝒟ᶻᴺ[1,:];           
    end
end

function ImplementBCs_Neumann_on_D2!(diffMatrix, grid; order_accuracy)
    N   = length(grid.z)
    del = grid.z[2] - grid.z[1]
    if order_accuracy == 4 
        diffMatrix.𝒟²ᶻᴺ[1,:] .= 0;                  diffMatrix.𝒟²ᶻᴺ[1,1] = -1/del^2;
        diffMatrix.𝒟²ᶻᴺ[1,2]  = 1/del^2;         

        diffMatrix.𝒟²ᶻᴺ[2,:] .= 0;                  diffMatrix.𝒟²ᶻᴺ[2,1] = (15/12)/del^2;
        diffMatrix.𝒟²ᶻᴺ[2,2]  = -(5/2)/del^2;       diffMatrix.𝒟²ᶻᴺ[2,3] = (4/3)/del^2;
        diffMatrix.𝒟²ᶻᴺ[2,4]  = -(1/12)/del^2;

        diffMatrix.𝒟²ᶻᴺ[N,:]   .= 1.0 .* diffMatrix.𝒟²ᶻᴺ[1,:];                 
        diffMatrix.𝒟²ᶻᴺ[N-1,:] .= 1.0 .* diffMatrix.𝒟²ᶻᴺ[2,:]; 
    else
        diffMatrix.𝒟²ᶻᴺ[1,:]   .= 0;                  
        diffMatrix.𝒟²ᶻᴺ[1,1]    = -1.0/del^2;
        diffMatrix.𝒟²ᶻᴺ[1,2]    = 1.0/del^2;
        diffMatrix.𝒟²ᶻᴺ[N,:]   .= 1.0 .* diffMatrix.𝒟²ᶻᴺ[1,:];         
    end
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

    y1, diffMatrix.𝒟ʸ  = FourierDiff_fdm(params.Ny, 1)
    _,  diffMatrix.𝒟²ʸ = FourierDiff_fdm(params.Ny, 2)
    _,  diffMatrix.𝒟⁴ʸ = FourierDiff_fdm(params.Ny, 4)

    # Transform the domain and derivative operators from [0, 2π) → [0, L)
    grid.y         = params.L/2π  .* y1
    diffMatrix.𝒟ʸ  = (2π/params.L)^1 .* diffMatrix.𝒟ʸ
    diffMatrix.𝒟²ʸ = (2π/params.L)^2 .* diffMatrix.𝒟²ʸ
    diffMatrix.𝒟⁴ʸ = (2π/params.L)^4 .* diffMatrix.𝒟⁴ʸ

    # Chebyshev in z-direction
    grid.z, diffMatrix.𝒟ᶻ  = cheb(params.Nz-1)
    grid.z = grid.z[:,1]
    diffMatrix.𝒟²ᶻ = diffMatrix.𝒟ᶻ  * diffMatrix.𝒟ᶻ
    diffMatrix.𝒟⁴ᶻ = diffMatrix.𝒟²ᶻ * diffMatrix.𝒟²ᶻ

    grid.z = collect(range(0.0, stop=params.H, length=params.Nz));
    @assert std(diff(grid.z)) ≤ 1e-6
    diffMatrix.𝒟ᶻ  = ddz(  grid.z, order_accuracy=4 );
    diffMatrix.𝒟²ᶻ = ddz2( grid.z, order_accuracy=4 );
    diffMatrix.𝒟⁴ᶻ = ddz4( grid.z, order_accuracy=4 );

    return nothing
end

function ImplementBCs_chebyshev!(Op, diffMatrix, params)

    Iʸ = Eye{Float64}(params.Ny)
    Iᶻ = Eye{Float64}(params.Nz)

    # Dirichilet boundary condition
    @. diffMatrix.𝒟ᶻᴰ  = diffMatrix.𝒟ᶻ
    @. diffMatrix.𝒟²ᶻᴰ = diffMatrix.𝒟²ᶻ
    @. diffMatrix.𝒟⁴ᶻᴰ = diffMatrix.𝒟⁴ᶻ

    n = params.Nz
    for iter ∈ 1:n-1
        diffMatrix.𝒟⁴ᶻᴰ[1,iter+1] = diffMatrix.𝒟⁴ᶻᴰ[1,iter+1] + 
                                (-1.0 * diffMatrix.𝒟⁴ᶻᴰ[1,1] * diffMatrix.𝒟²ᶻᴰ[1,iter+1])

          diffMatrix.𝒟⁴ᶻᴰ[n,iter] = diffMatrix.𝒟⁴ᶻᴰ[n,iter] + 
                                (-1.0 * diffMatrix.𝒟⁴ᶻᴰ[n,n] * diffMatrix.𝒟²ᶻᴰ[n,iter])
    end

    diffMatrix.𝒟ᶻᴰ[1,1] = 0.0
    diffMatrix.𝒟ᶻᴰ[n,n] = 0.0

    diffMatrix.𝒟²ᶻᴰ[1,1] = 0.0
    diffMatrix.𝒟²ᶻᴰ[n,n] = 0.0   

    diffMatrix.𝒟⁴ᶻᴰ[1,1] = 0.0
    diffMatrix.𝒟⁴ᶻᴰ[n,n] = 0.0  

    # Neumann boundary condition
    @. diffMatrix.𝒟ᶻᴺ  = diffMatrix.𝒟ᶻ
    @. diffMatrix.𝒟²ᶻᴺ = diffMatrix.𝒟²ᶻ
    
    for iter ∈ 1:n-1
        diffMatrix.𝒟²ᶻᴺ[1,iter+1] = (diffMatrix.𝒟²ᶻᴺ[1,iter+1] + 
                                (-1.0 * diffMatrix.𝒟²ᶻᴺ[1,1] * diffMatrix.𝒟ᶻᴺ[1,iter+1]/diffMatrix.𝒟ᶻᴺ[1,1]))

          diffMatrix.𝒟²ᶻᴺ[n,iter] = (diffMatrix.𝒟²ᶻᴺ[n,iter] + 
                                (-1.0 * diffMatrix.𝒟²ᶻᴺ[n,n] * diffMatrix.𝒟ᶻᴺ[n,iter]/diffMatrix.𝒟ᶻᴺ[n,n]))
    end

    diffMatrix.𝒟²ᶻᴺ[1,1] = 0.0
    diffMatrix.𝒟²ᶻᴺ[n,n] = 0.0

    for iter ∈ 1:n
        diffMatrix.𝒟ᶻᴺ[1,iter] = 0.0
        diffMatrix.𝒟ᶻᴺ[n,iter] = 0.0
    end
    
    kron!( Op.𝒟ᶻᴰ  ,  Iʸ , diffMatrix.𝒟ᶻᴰ  )
    kron!( Op.𝒟²ᶻᴰ ,  Iʸ , diffMatrix.𝒟²ᶻᴰ )
    kron!( Op.𝒟⁴ᶻᴰ ,  Iʸ , diffMatrix.𝒟⁴ᶻᴰ )

    kron!( Op.𝒟ᶻᴺ  ,  Iʸ , diffMatrix.𝒟ᶻᴺ )
    kron!( Op.𝒟²ᶻᴺ ,  Iʸ , diffMatrix.𝒟²ᶻᴺ)

    kron!( Op.𝒟ʸ   ,  diffMatrix.𝒟ʸ  ,  Iᶻ ) 
    kron!( Op.𝒟²ʸ  ,  diffMatrix.𝒟²ʸ ,  Iᶻ )
    kron!( Op.𝒟⁴ʸ  ,  diffMatrix.𝒟⁴ʸ ,  Iᶻ ) 

    kron!( Op.𝒟ʸᶻᴰ   ,  diffMatrix.𝒟ʸ  ,  diffMatrix.𝒟ᶻᴰ  )
    kron!( Op.𝒟²ʸ²ᶻᴰ ,  diffMatrix.𝒟²ʸ ,  diffMatrix.𝒟²ᶻᴰ )

    return nothing
end


function ImplementBCs_fdm!(Op, diffMatrix, grid, params)
    # Dirichilet boundary condition
    @. diffMatrix.𝒟ᶻᴰ  = diffMatrix.𝒟ᶻ
    @. diffMatrix.𝒟²ᶻᴰ = diffMatrix.𝒟²ᶻ
    @. diffMatrix.𝒟⁴ᶻᴰ = diffMatrix.𝒟⁴ᶻ
        
    # Neumann boundary condition
    @. diffMatrix.𝒟ᶻᴺ  = diffMatrix.𝒟ᶻ
    @. diffMatrix.𝒟²ᶻᴺ = diffMatrix.𝒟²ᶻ

    ImplementBCs_Dirchilet_on_D1!(diffMatrix, grid, order_accuracy=4)
    ImplementBCs_Dirchilet_on_D2!(diffMatrix, grid, order_accuracy=4)
    ImplementBCs_Dirchilet_on_D4!(diffMatrix, grid, order_accuracy=4)

    ImplementBCs_Neumann_on_D1!(diffMatrix, grid, order_accuracy=4)
    ImplementBCs_Neumann_on_D2!(diffMatrix, grid, order_accuracy=4)
    
    Iʸ = Eye{Float64}(params.Ny)
    Iᶻ = Eye{Float64}(params.Nz)

    kron!( Op.𝒟ᶻᴰ  ,  Iʸ , diffMatrix.𝒟ᶻᴰ  )
    kron!( Op.𝒟²ᶻᴰ ,  Iʸ , diffMatrix.𝒟²ᶻᴰ )
    kron!( Op.𝒟⁴ᶻᴰ ,  Iʸ , diffMatrix.𝒟⁴ᶻᴰ )

    kron!( Op.𝒟ᶻᴺ  ,  Iʸ , diffMatrix.𝒟ᶻᴺ )
    kron!( Op.𝒟²ᶻᴺ ,  Iʸ , diffMatrix.𝒟²ᶻᴺ)

    kron!( Op.𝒟ʸ   ,  diffMatrix.𝒟ʸ  ,  Iᶻ ) 
    kron!( Op.𝒟²ʸ  ,  diffMatrix.𝒟²ʸ ,  Iᶻ )
    kron!( Op.𝒟⁴ʸ  ,  diffMatrix.𝒟⁴ʸ ,  Iᶻ ) 

    kron!( Op.𝒟ʸᶻᴰ   ,  diffMatrix.𝒟ʸ  ,  diffMatrix.𝒟ᶻᴰ  )
    kron!( Op.𝒟²ʸ²ᶻᴰ ,  diffMatrix.𝒟²ʸ ,  diffMatrix.𝒟²ᶻᴰ )

    return nothing
end


function BasicState!(mf, grid, params)
    Y, Z = ndgrid(grid.y, grid.z)
    Y    = transpose(Y)
    Z    = transpose(Z)
    @printf "size of Y: %s \n" size(Y)

    # imposed buoyancy profile
    B₀      = @. Z - 1.0/params.Ri * Y  
    ∂ʸB₀ = - 1.0/params.Ri .* ones(size(Y))  
    ∂ᶻB₀ = 1.0 .* ones(size(Y))  

    # along-front profile (using thermal wind balance)
    U₀      = @. 1.0 * Z #- 0.5
    ∂ᶻU₀    = ones(size(Y))  

    @printf "min/max values of ∂ᶻU₀: %f %f\n" minimum(∂ᶻU₀) maximum(∂ᶻU₀)
    @printf "min/max values of ∂ᶻB₀: %f %f\n" minimum(∂ᶻB₀) maximum(∂ᶻB₀)
    @printf "min/max values of ∂ʸB₀: %f %f\n" minimum(∂ʸB₀) maximum(∂ʸB₀)

      B₀  = B₀[:];
      U₀  = U₀[:];
    ∂ʸB₀  = ∂ʸB₀[:]; 
    ∂ᶻB₀  = ∂ᶻB₀[:]; 
    ∂ᶻU₀  = ∂ᶻU₀[:];

    mf.B₀[diagind(mf.B₀)] = B₀
    mf.U₀[diagind(mf.U₀)] = U₀

    mf.∇ᶻU₀[diagind(mf.∇ᶻU₀)] = ∂ᶻU₀
    mf.∇ʸB₀[diagind(mf.∇ʸB₀)] = ∂ʸB₀
    mf.∇ᶻB₀[diagind(mf.∇ᶻB₀)] = ∂ᶻB₀

    return nothing
end


function construct_matrices(params)
    grid        = TwoDimGrid{params.Ny,  params.Nz}()
    diffMatrix  = ChebMarix{ params.Ny,  params.Nz}()
    Op          = Operator{params.Ny * params.Nz}()
    mf          = MeanFlow{params.Ny * params.Nz}()
    Construct_DerivativeOperator!(diffMatrix, grid, params)
    #ImplementBCs_chebyshev!(Op, diffMatrix, params)
    ImplementBCs_fdm!(Op, diffMatrix, grid, params)
    BasicState!(mf, grid, params)

    N  = params.Ny * params.Nz
    I⁰ = Eye{Float64}(N)
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
    Note - Method 2 is probably the `best' option if the matrix, ∇ₕ², is close to singular.
    """
    ∇ₕ² = SparseMatrixCSC(Zeros{Float64}(N, N))
    ∇ₕ² = 1.0 * Op.𝒟²ʸ - 1.0 * params.kₓ^2 * I⁰

    # Method 1. SVD decmposition 
    # U, Σ, V = svd(∇ₕ²); 
    # H = sparse(V * inv(Diagonal(Σ)) * transpose(U))

    # Method 2. QR decomposition
    Qm, Rm = qr(∇ₕ²)
    invR   = inv(Rm) 
    Qm     = sparse(Qm) # by sparsing the matrix speeds up matrix-matrix multiplication 
    Qᵀ     = transpose(Qm)
    H      = sparse(invR * Qᵀ)

    # difference in L2-norm should be small: ∇ₕ² * (∇ₕ²)⁻¹ - I⁰ ≈ 0 
    @printf "||∇ₕ² * (∇ₕ²)⁻¹ - I||₂ =  %f \n" norm(∇ₕ² * H - I⁰) 

    D⁴ = SparseMatrixCSC(Zeros{Float64}(N, N))
    D⁴ = sparse(1.0 * params.δ^4 * Op.𝒟⁴ʸ 
                + 1.0 * Op.𝒟⁴ᶻᴰ 
                + 1.0 * params.δ^4 * params.kₓ^4 * I⁰
                - 2.0 * params.δ^4 * params.kₓ^2 * Op.𝒟²ʸ 
                - 2.0 * params.δ^2 * params.kₓ^2 * Op.𝒟²ᶻᴰ
                + 2.0 * params.δ^2 * Op.𝒟²ʸ²ᶻᴰ)
        
    D²  = sparse(1.0 * Op.𝒟²ᶻᴰ + 1.0 * params.δ^2 * ∇ₕ²)
    Dₙ² = sparse(1.0 * Op.𝒟²ᶻᴺ + 1.0 * params.δ^2 * ∇ₕ²)

    #* 1. uᶻ equation (bcs: uᶻ = ∂ᶻᶻuᶻ = 0 @ z = 0, 1)
    𝓛₁[:,    1:1s₂] = (-1.0 * params.E * D⁴ 
                    + 1.0 * im * params.kₓ * mf.U₀ * D²)
    𝓛₁[:,1s₂+1:2s₂] = 1.0 * Op.𝒟ᶻᴺ 
    𝓛₁[:,2s₂+1:3s₂] = -1.0 * params.Ri * ∇ₕ²

    #* 2. ωᶻ equation (bcs: ∂ᶻωᶻ = 0 @ z = 0, 1)
    𝓛₂[:,    1:1s₂] = - 1.0 * mf.∇ᶻU₀ * Op.𝒟ʸ - 1.0 * Op.𝒟ᶻᴰ
    𝓛₂[:,1s₂+1:2s₂] = (1.0 * im * params.kₓ * mf.U₀ * I⁰
                    - 1.0 * params.E * Dₙ²)
    𝓛₂[:,2s₂+1:3s₂] = 0.0 * I⁰        

    #* 3. b equation (bcs: b = 0 @ z = 0, 1)
    𝓛₃[:,    1:1s₂] = (1.0 * mf.∇ᶻB₀ * I⁰
                    - 1.0 * mf.∇ʸB₀ * H * Op.𝒟ʸᶻᴰ) 
    𝓛₃[:,1s₂+1:2s₂] = 1.0 * im * params.kₓ * mf.∇ʸB₀ * H * I⁰
    𝓛₃[:,2s₂+1:3s₂] = (-1.0 * params.E * D² 
                    + 1.0 * im * params.kₓ * mf.U₀ * I⁰) 

    𝓛 = sparse([𝓛₁; 𝓛₂; 𝓛₃]);

##############

    # [uz, wz, b] ~ [uz, wz, b] exp(σt), growth rate = imag(σ) * k
    cnst = -1.0
    ℳ₁[:,    1:1s₂] = 1.0cnst * D²
    ℳ₂[:,1s₂+1:2s₂] = 1.0cnst * I⁰ 
    ℳ₃[:,2s₂+1:3s₂] = 1.0cnst * I⁰

    ℳ = sparse([ℳ₁; ℳ₂; ℳ₃])

    #@. 𝓛 *= 1.0/params.kₓ

    @printf "Done constructing matrices \n"
    return mf, grid, 𝓛, ℳ
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
    a = ShiftAndInvert(factorize(A), B, Vector{eltype(A)}(undef, size(A,1)))
    LinearMap{eltype(A)}(a, size(A,1), ismutating=true)
end

function construct_linear_map(H, S; num_thread=40)
    ps = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.COMPLEX_NONSYM)
    pardisoinit(ps)
    fix_iparm!(ps, :N)
    H_pardiso = get_matrix(ps, H, :N)
    b = rand(ComplexF64, size(H, 1))
    set_phase!(ps, Pardiso.ANALYSIS)
    #set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)
    set_nprocs!(ps, num_thread) 
    pardiso(ps, H_pardiso, b)
    set_phase!(ps, Pardiso.NUM_FACT)
    pardiso(ps, H_pardiso, b)
    return (LinearMap{eltype(H)}(
            (y, x) -> begin
                set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
                pardiso(ps, y, H_pardiso, S * x)
            end,
            size(H, 1);
            ismutating=true), ps)
end

"""
Parameters:
"""
@with_kw mutable struct Params{T<:Real} @deftype T
    L::T        = 2π        # horizontal domain size
    H::T        = 1.0       # vertical   domain size
    Ri::T       = 2.0       # Richardson number
    δ::T        = 10.0       # parameter denotes nonhydrostaty, 0:hydrostaty
    kₓ::T       = 0.0       # x-wavenumber
    E::T        = 1.0e-8    # Ekman number 
    Ny::Int64   = 20        # no. of y-grid points
    Nz::Int64   = 20        # no. of grid points in z
end


# function search_complexregion(serach_region, radi, serach_region_prev)
#     if radi > real(serach_region_prev) && radi > abs(imag(serach_region_prev))
#         radi  = radi 
#         serach_region = complex(radi, 
#                                 imag(serach_region))
#     elseif radi > real(serach_region_prev) && radi < abs(imag(serach_region_prev))
#         radi  = radi 
#         sign_ = sign(imag(serach_region))
#         serach_region = complex(radi, 
#                                 imag(serach_region) + 1im*sign_*radi)
#     elseif radi < real(serach_region_prev) && radi < abs(imag(serach_region_prev))
#         radi  = radi 
#         sign_ = sign(imag(serach_region))
#         serach_region = complex(radi, 
#                                 imag(serach_region) + 1im*sign_*radi)
#     else
#         serach_region = serach_region_prev
#         radi  = radi
#     end
#     return serach_region, radi
# end


function search_complexregion(Δkₓ, radi₀, serach_region₋₁, serach_region₋₂)
    ∂λ_∂kₓ = (serach_region₋₁ - serach_region₋₂) / Δkₓ
    serach_region = serach_region₋₁ + ∂λ_∂kₓ * Δkₓ
    return serach_region, radi₀ #min(radi₀, abs(real(serach_region)))
end

function solve_SI2d(kₓ, Ny, Nz, m₀, x₀, emid, ra, ctr)
    params = Params{Float64}(kₓ=kₓ, Ny=Ny, Nz=Nz)
    printstyled("kₓ: $(kₓ) \n"; color=:yellow)
    @printf "Ekman number: %1.1e \n" params.E

    mf, grid, 𝓛, ℳ = construct_matrices(params)
    N = params.Ny * params.Nz 
    MatSize = 3N
    @assert size(𝓛, 1) == MatSize && 
            size(𝓛, 2) == MatSize &&
            size(ℳ, 1) == MatSize &&
            size(ℳ, 2) == MatSize "matrix size does not match!"

    # MatrixMarket.mmwrite("save_matrix/d_10/systemA_" * string(floor(Int8, ctr)) * ".mtx", 𝓛)
    # MatrixMarket.mmwrite("save_matrix/d_10/systemB_" * string(floor(Int8, ctr)) * ".mtx", ℳ)

    #FEAST parameters
    emid  = emid        #contour center
    ra    = ra          #contour radius 1
    rb    = ra          #contour radius 2
    nc    = 800         #number of contour points
    m₀    = m₀          #subspace dimension
    x₀    = x₀          #sprand(ComplexF32, MatSize, m₀, 0.1)   
    ε     = 1.0e-6      # tolerance
    #=shift the contour center and radii in case eigenvalues 
    are not found with the above specified contour, i.e.,
    emid = emid ± εᵣ, ra = εᵣ, rb = εᵣ 
    =#
    maxit = 50                 #maximum FEAST iterations
    λₛ, Χ = feast_linear(𝓛, ℳ, x₀, nc, emid, ra, rb, ε, ra, 1e6+1e6im, maxit)

    contour    = circular_contour_trapezoidal(emid, ra, 200)
    #λₛ, Χ, res = gen_feast!(x₀, 𝓛, ℳ, contour, iter=maxit, debug=true, ϵ=ε)

    ## =======

    cnst = 1.0 #-1.0im * params.kₓ 
    @. λₛ *= cnst

    @assert length(λₛ) > 0 "No eigenvalue(s) found!"
    @printf "\n"

    # Post Process egenvalues
    λₛ, Χ = remove_evals(λₛ, Χ, 0.0, 10.0, "M") # `R`: real part of λₛ.
    λₛ, Χ = sort_evals(λₛ, Χ, "R")              # `lm': largest magnitude.

    #= 
    this removes any further spurious eigenvalues based on norm 
    if you don't need it, just `comment' it!
    =#
    while norm(𝓛 * Χ[:,1] - λₛ[1]/cnst * ℳ * Χ[:,1]) > ε # || imag(λₛ[1]) > 0
        #@printf "norm: %f \n" norm(𝓛 * Χ[:,1] - λₛ[1]/cnst * ℳ * Χ[:,1]) 
        λₛ, Χ = remove_spurious(λₛ, Χ)
    end
    
    print_evals(λₛ, length(λₛ))
    @printf "largest growth rate : %1.4e%+1.4eim\n"  real(λₛ[1]) imag(λₛ[1])

    𝓛 = nothing
    ℳ = nothing

    return mf.U₀, mf.B₀, grid.y, grid.z, λₛ[1], Χ 
end


# calculate growthrate over a range of wavenumber
function RegimeDiag(kₓ; Ny, Nz)
    λₛ = zeros(ComplexF64, length(kₓ))

    N = Ny*Nz
    MatSize = 3N 
    m₀ = 30               #subspace dimension

    # search contour domain for FEAST algorithm
    # here I've used a circle contour (can be elliptic by changing one of radii)
    emid = complex(0.01, -0.01) #contour center 
    ra  = 0.01 #contour radius 1
    ra₀ = ra

    ctr = 1
    Δkₓ = kₓ[2] - kₓ[1]

    Χ = sprand(ComplexF64, MatSize, m₀, 0.1) 
    for it ∈ 1:length(kₓ)
        if it==1
            #eigenvector initial guess for FEAST algorithm
            @time U, B, y, z, λₛ[it], Χ = solve_SI2d(kₓ[it], Ny, Nz, m₀, Χ, emid, ra, ctr)
        else
            cnst = 1.0 #-1.0im * kₓ[it-1]
            #eigenvector initial guess for FEAST algorithm
            #x₀ = sprand(ComplexF64, MatSize, m₀, 0.1) 
            if it > 2
                emid, ra  = search_complexregion( Δkₓ, 0.01, λₛ[it-1]/cnst,  λₛ[it-2]/cnst)
            end
            println(ra)
            @time U, B, y, z, λₛ[it], Χ = solve_SI2d(kₓ[it], Ny, Nz, m₀, sparse(Χ), emid, ra, ctr)
        end
        ContourPlot(y, z, real(Χ[1:1N]), real(Χ[1N+1:2N]), diag(U), "Ston1971_", it)
        ctr += 1
        ## initial guess from previouse eigenvectors; for it=1, it is a random 
        #x₀ = sparse(Χ) + 0.1sprand(ComplexF64, MatSize, size(Χ,2), 0.1)
        @printf("=================================================================== \n")
    end

    if length(kₓ) == 1
        return U, B, y, z, λₛ[it], Χ
    else
        return λₛ
    end

end
 
# x-wavenumber range
#kₓ = range(1e-3, 1.75, length=50) #δ = 0

kₓ = range(1e-2, 2.0, length=60) #δ = 10
λₛ = RegimeDiag(kₓ, Ny=30, Nz=30)

save("GrowthRate_delta10_ek1e-9.jld", "kₓ", kₓ, "λₛ", λₛ)