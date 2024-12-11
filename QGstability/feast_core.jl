using SparseArrays
using Printf
using LinearAlgebra
using SparseMatrixDicts
using LazyGrids
using FillArrays
using LowRankApprox
using Pardiso
using MUMPS
using MPI
using ComplexRegions
using Random

function eigs_in_subspace(Λ, x, emid, ra, rb)
    @assert ra ≈ rb
    R = Circle(emid, ra)

    neigs = length(Λ) 
    store = []   
    for it ∈ 1:neigs
        if isinside(Λ[it], R)
            push!(store, it)
        end
    end
    Λ = Λ[store]
    x = x[:,store]
    return Λ, x
end

function MKLPardiso_MatrixSolver!(Qk, num, den, num_thread=40)
    n, m = size(num)
    ps = MKLPardisoSolver()
    set_nprocs!(ps, num_thread)
    set_matrixtype!(ps, Pardiso.COMPLEX_NONSYM)

    # Initialize the default settings with the current matrix type
    pardisoinit(ps)

    # Remember that we pass in a CSC matrix to Pardiso, so need
    # to set the transpose iparm option.
    fix_iparm!(ps, :N)

    # Get the correct matrix to be sent into the pardiso function.
    # :N for normal matrix, :T for transpose, :C for conjugate
    A_pardiso = get_matrix(ps, num, :N)

    # Analyze the matrix and compute a symbolic factorization.
    set_phase!(ps, Pardiso.ANALYSIS)
    set_perm!(ps, randperm(n))
    pardiso(ps, A_pardiso, den)

    # Compute the numeric factorization.
    set_phase!(ps, Pardiso.NUM_FACT)
    pardiso(ps, A_pardiso, den)

    # Compute the solutions X using the symbolic factorization.
    set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)

    #Qk = similar(den) # Solution is stored in X
    pardiso(ps, Qk, A_pardiso, den)

    # Free the PARDISO data structures.
    set_phase!(ps, Pardiso.RELEASE_ALL)
    pardiso(ps)

    return nothing
end

#get dictionary for storing log information
function getLog(m₀, nc, maxit)
    data = Dict{Symbol, Any}()
    data[:residuals]        = zeros(Float64, maxit, m₀)
    data[:feastResidual]    = zeros(Float64, maxit)
    data[:eigenvalues]      = zeros(ComplexF64, maxit, m₀)
    data[:ninside]          = zeros(Int16, maxit)
    data[:iterations]       = 0
    data[:insideIndices]    = Dict{Int16, Any}()
    data[:shiftIndex]       = Dict{ComplexF64, Int64}()

    data[:linResiduals]     = zeros(Float64, m₀, nc,maxit)
    data[:linIts]           = zeros(Int16, m₀, nc, maxit)

    data[:hlinResiduals]    = zeros(Float64, m₀, nc, maxit)
    data[:hlinIts]          = zeros(Int16, m₀, nc, maxit)

    return data
end


########################
# CORE FEAST FUNCTIONS #
########################

#These functions are one- and two-sided implementations of the FEAST algorithm. 
#They are essentially outlines that encompass all possible variations of the FEAST algorithm.

#They are used by passing functions to them to perform the essential operations of the FEAST algorithm, 
#such as evaluating the eigenvector residual and calculating the contour integrand.

#core one-sided FEAST algorithm; this function is used for all variations of FEAST
#=
    Tf: eigenvector residual function. E.g. for linear problems Tf(l,x) = B*x*diagm(l)-A*x
    integrand: function that returns integral integrand evaluated at point z 
    in complex plane, applied to a vector x. E.g. for linear FEAST integrand(z,x,l)=\(z*B-A,x)
    rrsolve: function for returning eigenvalue and eigenvector approximations from rayleigh ritz. 
    E.g. for linear FEAST rrsolve(Q)=eig(Q'*A*Q,Q'*B*Q)
    x0: initial guess for eigenvectors
    nc: number of quadrature points for numerical integration
    emid: center of integration ellipse
    ra,rb: radii of integration ellipse
    ε: convergence tolerance
    maxit: maximum number of FEAST iterations
    insideEps: residual threshold for determining whether or not an eigenvalue inside the contour is spurious
    log: if true, returns dictionary object with logged convergence information
    verbose(0-1): how much information to print out
    set_nin: if equal to an integer, sets the number of eigenvalues inside the contour to 'set_nin'; 
    used for diagnostic purposes
=#
#function feast_core(Tf, integrand, rrsolve, x0, nc, emid, ra, rb, ε::Float64, maxit; log=false, insideEps=1e-2, verbose=1, set_nin=false)


function feast_core(A, 
                    B, 
                    x₀, 
                    nc, 
                    emid, 
                    ra, 
                    rb, 
                    ε, 
                    εᵣ,
                    λref,
                    maxit; 
                    log=false, 
                    insideEps=1e-2, 
                    verbose=1, 
                    set_nin=false)

    T::Type = ComplexF64
    #get shapes of arrays:
	n, m₀ = size(x₀)
    #storing convergence info:
    data = getLog(m₀, nc, maxit)
    #get quadrature points from trapezoidal rule:
    gk, wk = trapezoidal(nc)
    ε₀ = ε
    """
    offset angle of first quadrature point; makes 
    sure it isn't on real axis for hermitian problems
    """
    offset   = π/nc * 1.0

    #save indices for contour points
    @inbounds for k in 1:nc
        #curve parametrization angle:
        θ = gk[k]*π + π + offset
        #quadrature point:
        z = emid + 0.5(ra+rb)*exp(im*θ) + 0.5(ra-rb)*exp(-1.0*im*θ)
        data[:shiftIndex][z] = k
    end

    #Initialize FEAST subspace:
    Q  = zeros(T, n, m₀)
    copyto!(Q, x₀)
    Qk = zeros(T, n, m₀)

    #initialize eigenvalue estimates:
	Λ       = zeros(T, m₀)
    R       = zeros(T, n, m₀)
    Aq      = zeros(T, m₀, m₀)
    Bq      = zeros(T, m₀, m₀)

    res     = zeros(T, n, m₀)
    #den     = zeros(T, n, m₀)
    den     = SparseMatrixCSC(Zeros{T}(n, m₀)) #this speedup the calculation for MUMPS
    num     = SparseMatrixCSC(Zeros{T}(n, n))

    res = 1.0 #initial residual
    #initialize iteration number:
	it = 0

    MPI.Init()
    icntl = get_icntl(verbose=false)
    mumps = Mumps{T}(mumps_unsymmetric, icntl, default_cntl64)

    check = true
    ninside = 0

    Iter = 6
	@inbounds while res > ε && it < maxit
		it = it + 1 #update number of iterations

        data[:iterations] = it

        #Rayleigh-Ritz
        mul!(R,  A,  Q)
        mul!(Aq, Q', R)     # Aq = Q' * A * Q
        mul!(R,  B,  Q)
        mul!(Bq, Q', R)     # Bq = Q' * B * Q

        F  = eigen!(Aq, Bq)
        Λ  = F.values
        Xq = F.vectors
        mul!(R, Q, Xq) ### Recover eigenvectors from Ritz vectors ( x = Q * Xq )

        #sort everything by real part of eigenvalue
        p   = sortperm(real(Λ))
        Λ   = Λ[p]
        x   = R[:,p]

        #store convergence data
        #indicate which eigenvalues are inside the contour
        data[:insideIndices][it] = getInsideIndex(Λ, emid, ra, rb)
        #store all the eigenvalues
        data[:eigenvalues][it,:] = Λ

        #calculate eigenvector residuals
        resvecs = A*x - B*x*diagm(0 => Λ)
        for i in 1:m₀
            data[:residuals][it,i] = norm(resvecs[:,i]) / norm(x[:,i])
        end

        #find the largest residual inside the contour
        res, ninside = getMaxResInside(Λ, x, resvecs, emid, ra, rb; inEps=insideEps, set_nin=set_nin)

        data[:feastResidual][it] = res
        data[:ninside][it]       = ninside

        verbose>0 && println("$it  res=$res  minres=$(minimum(data[:residuals][it,:]))  nin=$ninside")

        println("$emid ($ra)")

        if res < ε
            if log
	            return eigs_in_subspace(Λ, x, emid, ra, rb), data#, emid, ra, rb
	        else
	            return eigs_in_subspace(Λ, x, emid, ra, rb) #, emid #, ra, rb
	        end
        end

        #apply contour integral to get FEAST subspace:
	    fill!(Q, 0.0)
        LinearAlgebra.mul!(den, B, x)

	    for k in 1:nc  
            #integration curve is an ellipse centered at emid, with radii ra and rb
	        #curve parametrization angle:
	        θ = gk[k]*π + π + offset

	        #quadrature point:
	        z    = emid + 0.5(ra+rb) * exp(im*θ) + 0.5(ra-rb) * exp(-1.0*im*θ)
            num  = z*B - A

            # @info("Solving Ax=b")
            # MKLPardiso_MatrixSolver!(Qk, num, den)

            associate_matrix!(mumps, num)
            associate_rhs!(mumps, den)
            factorize!(mumps)
            MUMPS.solve!(mumps)
            Qk = get_solution(mumps)
            MUMPS.set_job!(mumps, 1)
        
            #Qk   = num\den
            #@info("Done solving Ax=b")

            #add integrand contribution to quadrature:
            z = wk[k] * ( 0.5(ra+rb)*exp(im*θ) - 0.5(ra-rb)*exp(-1.0*im*θ) )
	        Q = Q + z * Qk
        end    
        #Orthonormalize FEAST subspace to avoid spurious eigenpairs:
        #0.63
        # Qq, Rq = qr(Q)
        # Q[:] = Qq

        #0.7
        Q = Matrix( qr(Q).Q )
	end

    finalize(mumps)
    MPI.Finalize()

    if log
        return eigs_in_subspace(Λ, x, emid, ra, rb), data# , emid, ra, rb
    else
        return eigs_in_subspace(Λ, x, emid, ra, rb) #, ra, rb
    end

end


#core two-sided FEAST algorithm
#Difference from symmetric FEAST: have to solve for left and right eigenvectors simultaneously and biorthogonalize them
function feastNS_core(Tf,hTf,integrand,hintegrand,rrsolvens,biortho,x0,y0,nc,emid,ra,rb,eps,maxit; insideEps=1e-2, log=false)
    # Tf: eigenvector residual function. E.g. for linear problems Tf(l,x) = B*x*diagm(l)-A*x
    # hTf: hermitian conjugate transpose of residual function
    # integrand: function that returns integral integrand evaluated at point z in complex plane, applied to a vector x. E.g. for linear FEAST integrand(z,x,l)=\(z*B-A,x)
    # rrsolvens: function for returning eigenvalue and eigenvector approximations from rayleigh ritz. E.g. for linear FEAST rrsolve(Q)=eig(Q'*A*Q,Q'*B*Q)
    # x0: initial guess for right eigenvectors
    # y0: initial guess for left eigenvectors
    # nc: number of quadrature points for numerical integration
    # emid: center of integration ellipse
    # ra,rb: radii of integration ellipse
    # eps: convergence tolerance
    # maxit: maximum number of FEAST iterations
    # insideEps: residual threshold for determining whether or not an eigenvalue inside the contour is spurious
    # log: if true, returns dictionary object with logged convergence information

    #get shapes of arrays:
	n, m0 = size(x0)

	#storing convergence info:
    data=getLog(m0,nc,maxit)

    #start with initial guess
    x=copy(x0)
    y=copy(y0)

    #initialize eigenvalue estimates:
	lest=zeros(ComplexF64,m0,1)

	#initialize iteration number:
	it=0

    #get quadrature points from trapezoidal rule:
    gk, wk = trapezoidal(nc)
    """
    offset angle of first quadrature point; 
    makes sure it isn't on real axis for hermitian problems
    """
    offset = pi/π 
    #save indices for contour points
    for k in 1:nc
        #curve parametrization angle:
        θ = gk[k]*π + π + offset

        #quadrature point:
        z = emid + 0.5(ra+rb)*exp(im*θ) + 0.5(ra-rb)*exp(-1.0*im*θ)
        data[:shiftIndex][z] = k
    end


    #Initialize FEAST subspaces:
    Q = copy(x)
    R = copy(y)

    res=1.0 #initial residual
	while res>eps && it<maxit
		it = it+1 #update number of iterations
		data[:iterations] = it

        #Biorthogonalize Q and R subspaces
        Q, R = biortho(Q, R)

        #rayleigh ritz
        y, lest, x = rrsolvens(Q, R)

        #calculate eigenvector residuals
        resvecs  = Tf(lest,  x)
        hresvecs = hTf(lest, y)

        #store convergence data
        #indicate which eigenvalues are inside the contour
        data[:insideIndices][it] = getInsideIndex(lest, emid, ra, rb)
        #store all the eigenvalues
        data[:eigenvalues][it,:] = lest

        #calculate eigenvector residuals
        resvecs = Tf(lest,x)
        for i in 1:m0
            data[:residuals][it,i] = norm(resvecs[:,i])/norm(x[:,i])
        end

        #find the largest residual inside the contour
        res,ninside=getMaxResInside(lest,x,resvecs,emid,ra,rb; inEps=insideEps)
        data[:feastResidual][it]=res
        data[:ninside][it]=ninside

        println("  $it  res=$res  minres=$(minimum(data[:residuals][it,:]))  nin=$ninside")
        if(res<eps)
            if(log)
                    return (y,lest,x,data)
                else
                    return (y,lest,x)
            end
        end

        #apply contour integral to get FEAST subspace:
	    Q = zero(x)
	    R = zero(y)
	    for k in 1:nc
	        #integration curve is an ellipse centered at emid, with radii ra and rb

	        #curve parametrization angle:
	        theta=gk[k]*pi+pi+offset

	        #quadrature point:
	        z=emid+((ra+rb)/2)*exp(im*theta)+((ra-rb)/2)*exp(-1.0*im*theta)

            #integrand evaluated at quadrature point z
            Qk=integrand(z,x,lest,data,resvecs)
            Rk=hintegrand(z,y,lest,data,hresvecs)

            #add integrand contribution to quadrature:
	        Q=Q+wk[k]*(((ra+rb)/2)*exp(im*theta)-((ra-rb)/2)*exp(-1.0*im*theta))*Qk
	        R=R+(wk[k]*(((ra+rb)/2)*exp(im*theta)-((ra-rb)/2)*exp(-1.0*im*theta)))'*Rk
        end

	end
        if(log)
            return (y,lest,x,data)
        else
            return (y,lest,x)
        end

end
