#include("mybicgstab.jl")
using SparseArrays
using Printf
using LinearAlgebra

#one-sided FEAST for linear eigenvalue problems
function feast_linear(A::SparseMatrixCSC, 
                    B::SparseMatrixCSC, 
                    x₀::SparseMatrixCSC, 
                    nc::Int64, 
                    emid::ComplexF64, 
                    ra::Float64, 
                    rb::Float64, 
                    ε::Float64, 
                    εᵣ::Float64,
                    λref::ComplexF64, 
                    maxit::Int64; 
                    log=false, 
                    insideEps=1.0e-2, 
                    set_nin=false)

    n, m₀ = size(x₀)

    if size(A, 1) != size(A, 2)
        error("Incorrect dimensions of A, must be square")
    elseif size(A,1) != n
        error("Incorrect dimensions of x₀, must match A")
    end

    #integrand(z, x)=\(z*B-A, B*x)

    # function integrand(z, x) #this function is slower compare to the top!!!
    #     res = zeros(ComplexF64, n, m₀)
    #     den = zeros(ComplexF64, n, m₀)
    #     num = zeros(ComplexF64, n, n)

    #     LinearAlgebra.mul!(den, B, x)
    #     LinearAlgebra.mul!(num, z, B)
    #     num = num - A
    #     LinearAlgebra.ldiv!(res, qr(num), den)
    #     return res
    # end

    # function rrsolve(Q)
    #     Aq = Q'*A*Q
    #     Bq = Q'*B*Q

    #     le, xe = eigen(Aq,Bq)
    #     return le, Q*xe
    # end

    # function rrsolve(Q)
    #     """
    #     Aq, Bq ∈ ℂ(m₀×m₀)
    #     """

    #     Λ  = zeros(ComplexF64, m₀)
    #     R  = similar(x₀, ComplexF64)
    #     Aq = zeros(ComplexF64, m₀, m₀)
    #     Bq = zeros(ComplexF64, m₀, m₀)
    #     #Xq = similar(x₀, ComplexF64)

    #     LinearAlgebra.mul!(R,  A,  Q)
    #     LinearAlgebra.mul!(Aq, Q', R)     # Aq = Q' * A * Q
    #     LinearAlgebra.mul!(R,  B,  Q)
    #     LinearAlgebra.mul!(Bq, Q', R)     # Bq = Q' * B * Q

    #     F  = eigen!(Aq, Bq)
    #     Λ  = F.values
    #     #Xq = F.vectors
    #     LinearAlgebra.mul!(R, Q, F.vectors)

    #     return Λ, R #Q*Xq 
    # end

    # Tf(λ, x) = A*x - B*x*diagm(0 => λ) 

    #return feast_core(Tf, integrand, rrsolve, x₀, nc, emid, ra, rb, ε, maxit; log=log, insideEps=insideEps, set_nin=set_nin)

    return feast_core(A, B, x₀, nc, emid, ra, rb, ε, εᵣ, λref, maxit; log=log, insideEps=insideEps, set_nin=set_nin)
end


#two-sided FEAST for linear eigenvalue problems
function feastNS_linear(A::SparseMatrixCSC,  
                        B::SparseMatrixCSC,  
                        x₀::SparseMatrixCSC, 
                        y₀::SparseMatrixCSC,  
                        nc::Int, 
                        emid::ComplexF64, 
                        ra::Float64, 
                        rb::Float64, 
                        eps::Float64, 
                        maxit::Int; 
                        log=false,
                        insideEps=1e-2)
    
    n, m₀ = size(x₀)

    integrand(z,x,lest,data,resvecs)=\(z*B-A,B*x)
    hintegrand(z,y,lest,hdata,hresvecs)=\((z*B-A)',B'*y)

    function rrsolvens(Q,R)
        Aq=R'*A*Q
        Bq=R'*B*Q
        (ye,le,xe)=nseig(Aq,Bq)
        return (R*ye,le,Q*xe)
    end

    Tf(l,x)=(B*x*spdiagm(0 => l)-A*x)
    hTf(l,y)=B'*y*spdiagm(0 => l)'-A'*y

    #use SVD to B-biorthogonalize subspaces
    function biortho(Q,R)
        Bq=R'*B*Q
        (u,s,v)=svd(Bq)
        Y=R*u*diagm( 0 => 1 ./sqrt.(s))
        X=Q*v*diagm( 0 => 1 ./sqrt.(s))
        return (X,Y)
    end

    return feastNS_core(Tf, hTf, integrand, hintegrand, rrsolvens, biortho, x₀, y₀, nc, emid, ra, rb, eps, maxit; log=log, insideEps=insideEps)

end


#generalized inexact FEAST algorithm with gmres
function ifeast_linear(A::SparseMatrixCSC,
                    B::SparseMatrixCSC,
                    x₀::SparseMatrixCSC,
                    alpha::Float64,
                    isMaxit::Int,
                    nc::Int,
                    emid::ComplexF64,
                    ra::Float64,
                    rb::Float64,
                    eps::Float64,
                    maxit::Int;
                    log=false,
                    insideEps=1e-2, 
                    verbose=1)
    n, m₀ = size(x₀)

    function integrand(z,x,lest,data,resvecs)
        nc=data[:shiftIndex][z]
        M=(z*B-A)
        #b=(B*x*diagm(lest)-A*x)
        #normb=maximum(sqrt.(sum(abs.(b).^2,1)))

        int=zeros(ComplexF64,n,m0)
        maxits=0
        rhs=convert(Array{ComplexF64,2},resvecs)

        for i in 1:m₀
            #(int[:,i],history)=bicgstabl(M,resvecs[:,i],1,max_mv_products=isMaxit,tol=alpha,initial_zero=true,log=true)
            #(int[:,i],history)=idrs(M,rhs[:,i];maxiter=isMaxit,tol=alpha,log=true)
            (int[:,i],history)=gmres(M,resvecs[:,i],restart=isMaxit,tol=alpha,initially_zero=true,maxiter=isMaxit,log=true)
            #(int[:,i],history)=minres(M,resvecs[:,i],tol=alpha,initially_zero=true,maxiter=isMaxit,log=true)
            nlinits=size(history[:resnorm],1)
            data[:linIts][i,nc,data[:iterations]]=nlinits
            #data[:linResiduals][i,nc,data[:iterations]]=history[:resnorm][nlinits]
            data[:linResiduals][i,nc,data[:iterations]]=norm(resvecs[:,i]-M*int[:,i])/norm(resvecs[:,i])

            if(nlinits>maxits)
                maxits=nlinits
            end
        end
        #int=\(M,resvecs)
        #println("      linits=$maxits")
        #int=zbicgstabBlock(M,resvecs,zeros(n,m0),isMaxit,alpha)
        return (x-int)*spdiagm(0 => 1 ./(z.-lest))
    end

    function rrsolve(Q)
        Aq=Q'*A*Q
        Bq=Q'*B*Q
        le,xe=eigen(Aq,Bq)
        return (le,Q*xe)
    end

    Tf(l,x)=(B*x*spdiagm(0 => l)-A*x)

    return feast_core(Tf, integrand, rrsolve, x₀, nc, emid, ra, rb, eps, maxit; log=log, insideEps=insideEps, verbose=verbose)

end

#generalized ifeast with bicgstab
function ifeast_linearBicgstab(A::SparseMatrixCSC,
                            B::SparseMatrixCSC,
                            x₀::SparseMatrixCSC,
                            alpha::Float64,
                            isMaxit::Int,
                            nc::Int,
                            emid::ComplexF64,
                            ra::Float64,
                            rb::Float64,
                            eps::Float64,
                            maxit::Int;
                            log=false,
                            insideEps=1e-2,
                            verbose=1)

    n, m₀ = size(x₀)

    function integrand(z,x,lest,data,resvecs)
        nc=data[:shiftIndex][z]
        M1=(z*B-A)
        P=speye(n)
        M=P*M1*P
        rhs=P*resvecs
        #b=(B*x*diagm(lest)-A*x)
        #normb=maximum(sqrt.(sum(abs.(b).^2,1)))

        int = zeros(ComplexF64, n, m₀)

        for i in 1:m₀
            int[:,i], history = bicgstabl(M, resvecs[:,i], 1, max_mv_products=isMaxit, tol=alpha, initial_zero=true, log=true)
            nlinits = size(history[:resnorm],1)
            data[:linIts][i,nc,data[:iterations]] = nlinits
            data[:linResiduals][i,nc,data[:iterations]] = history[:resnorm][nlinits]
        end


        return (x-int)*spdiagm(0 => 1 ./(z.-lest))
    end

    function rrsolve(Q)
        Aq = Q'*A*Q
        Bq = Q'*B*Q
        le, xe = eigen(Aq, Bq)
        return le, Q*xe
    end

    Tf(l,x)=(B*x*spdiagm(0 => l)-A*x)

    return feast_core(Tf, integrand, rrsolve, x₀, nc, emid, ra, rb, eps, maxit; log=log, insideEps=insideEps, verbose=verbose)

end


#inexact two-sided ifeast with my own bicgstab implementation
function ifeastNS_linear(A::SparseMatrixCSC,
                        B::SparseMatrixCSC,
                        x0::SparseMatrixCSC,
                        y0::SparseMatrixCSC,
                        alpha::Float64,
                        isMaxit::Int,
                        nc::Int,
                        emid::ComplexF64,
                        ra::Float64,
                        rb::Float64,
                        eps::Float64,
                        maxit::Int; 
                        log=false,
                        insideEps=1e-2)
    (n,m0)=size(x0)

    #integrand(z,x,lest,data,resvecs)=\(z*B-A,B*x)
    #hintegrand(z,y,lest,hdata,hresvecs)=\((z*B-A)',B'*y)
    function integrand(z,x,lest,data,resvecs)
        M=z*B-A
        rhs=B*x*spdiagm(0 => lest)-A*x
        int0=zeros(ComplexF64,n,m0)
        int=zeros(ComplexF64,n,m0)
        int=zbicgstabBlock(M,resvecs,int0,isMaxit,alpha)
        for i in 1:m0
            #(int[:,i],history)=gmres(M,rhs[:,i],restart=isMaxit,tol=alpha,initially_zero=true,maxiter=isMaxit,log=true)
            #(int[:,i],history)=bicgstabl(M,rhs[:,i],2,max_mv_products=isMaxit,tol=alpha,initial_zero=true,log=true)
        end

        #int=\(M,resvecs)
        return (x-int)*spdiagm(0 => 1 ./(z.-(lest)))
    end

    function hintegrand(z,y,lest,hdata,hresvecs)
        M=(z*B-A)'
        rhs=B'*y*spdiagm(0 => lest)'-A'*y
        int0=zeros(ComplexF64,n,m0)
        int=zeros(ComplexF64,n,m0)
        int=zbicgstabBlock(M,hresvecs,int0,isMaxit,alpha)
        for i in 1:m0
            #(int[:,i],history)=gmres(M,rhs[:,i],restart=isMaxit,tol=alpha,initially_zero=true,maxiter=isMaxit,log=true)

            #(int[:,i],history)=bicgstabl(M,rhs[:,i],2,max_mv_products=isMaxit,tol=alpha,initial_zero=true,log=true)
        end
        #int=\(M,hresvecs)
        return (y-int)*spdiagm(0 => 1 ./(z'.-conj.(lest)))
    end

    function rrsolvens(Q,R)
        Aq=R'*A*Q
        Bq=R'*B*Q
        (ye,le,xe)=nseig(Aq,Bq)
        return (R*ye,le,Q*xe)
    end

    Tf(l,x)=(B*x*spdiagm(0 => l)-A*x)
    hTf(l,y)=B'*y*spdiagm(0 => l)'-A'*y

    #use SVD to B-biorthogonalize subspaces
    function biortho(Q,R)
        Bq=R'*B*Q
        (u,s,v)=svd(Bq)
        Y=R*u*diagm( 0 => 1 ./sqrt.(s))
        X=Q*v*diagm( 0 => 1 ./sqrt.(s))
        return (X,Y)
    end

    return feastNS_core(Tf, hTf, integrand, hintegrand, rrsolvens, biortho, x0, y0, nc, emid, ra, rb, eps, maxit; log=log, insideEps=insideEps)

end
