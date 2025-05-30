function estConvRate(residuals)
    #set up and solve least squares problem to estimate slope of convergence line
    n=size(residuals,1)
    A=zeros(n,2)
    A[:,1]=ones(n)
    A[:,2]=collect(0:n-1)

    #solve least squares Ax=residuals
    x=\(A,log.(residuals)/log(10))
    return x[2]
end

function linquad(M,D,K)

    #A=[-D -K; eye(size(M,1)) zeros(size(M))]
    #B=[M zeros(size(M)); zeros(size(M)) eye(size(M,1))]

    A=[-D -K; -K zeros(size(M))]
    B=[M zeros(size(M)); zeros(size(M)) -K]

    #A=[-K zeros(size(M)); zeros(size(M)) M]
    #B=[D M; M zeros(size(M))]

    return (A,B)
end

#get left and right eigenvectors of generalized, nonsymmetric eigenvalue problem
function nseig(A,B)
    if(typeof(A)!=Array{ComplexF64,2})
        A=convert(Array{ComplexF64,2},A)
    end

    if(typeof(B)!=Array{ComplexF64,2})
        B=convert(Array{ComplexF64,2},B)
    end

    alpha,beta,vl,vr=LinearAlgebra.LAPACK.ggev!('V','V',A,B)

    return vl,alpha./beta,vr
end

#trapezoidal quadrature rule
function trapezoidal(nc)
    points  = zeros(nc)
    weights = zeros(nc)

    a = -1.0
    b = 1.0
    Δx = (b-a)/(nc)
    @inbounds for i in 1:nc
        points[i]  = a + Δx*(i-1)
        weights[i] = (b-a)/(nc+1)
    end

    return points, weights
end

function getRealSpurious(lest,resnorms,emid,ra,rb; inEps=1e-2)
#get distances of estimated eigenvalues from middle of contour:
	#have to rescale the approximate eigenvalues to adjust for the fact that we are using an ellipse and not a circle
	#more complicated contours will require a different approach
	lestdif=abs.(real(lest.-emid)/abs.(ra)+im*imag(lest.-emid)/abs.(rb))
	m0=size(lest,1)
	ninside=0

    p=sortperm(resnorms)

	s=Array{Int64,1}() #indices of eigenvalues inside contour
	spurious=Array{Int64,1}() #indices of spurious eigenvalues inside contour
	other=Array{Int64,1}() #stuff outside contour


	@inbounds for k in 1:m0
	    if(lestdif[k]<=1.0 && resnorms[k]<inEps)
	        ninside=ninside+1
	        push!(s,k)
	    elseif(lestdif[k]<=1.0 && !(resnorms[k]<inEps))
	        push!(spurious,k)
	    elseif (!(lestdif[k]<=1.0))
	        push!(other,k)
	    end
	end

    return s, spurious, other
end

#find maximum residual inside contour
function getMaxResInside(lest,x,res,emid,ra,rb;inEps=1e-2, getIndices=false, getSpurious=false, set_nin=false)
#get distances of estimated eigenvalues from middle of contour:
	#have to rescale the approximate eigenvalues to adjust for the fact that we are using an ellipse and not a circle
	#more complicated contours will require a different approach
	lestdif=abs.(real(lest.-emid)/abs.(ra)+im*imag(lest.-emid)/abs.(rb))
	(n,m0)=size(x)
	ninside=0

	resnorms=zeros(m0)
    for i in 1:m0
        resnorms[i]=norm(res[:,i])/norm(x[:,i])
    end

    p=sortperm(resnorms)

	s=Array{Int64,1}() #indices of eigenvalues inside contour
	spurious=Array{Int64,1}() #indices of spurious eigenvalues inside contour

	@inbounds for k in 1:m0
	    if(lestdif[k]<=1.0 && resnorms[k]<inEps)
	        ninside=ninside+1
	        push!(s,k)
	    end
	    if(lestdif[k]<=1.0 && !(resnorms[k]<inEps))
	        push!(spurious,k)
	    end
	end


    maxres=0.0
	if (set_nin!=false)
	    maxres=resnorms[set_nin]
    else
        if(ninside==0)
            maxres=maximum(resnorms)
        else
            @inbounds for k in 1:m0
                if(lestdif[k]<=1.0 && resnorms[k]<inEps) #using 1.0 as the radius because we rescaled everything when forming lestdif
                   tmp=resnorms[k]#norm(res[:,k])/norm(x[:,k])
                   if(tmp>maxres)
                        maxres=tmp
                   end
                end
            end
        end
    end

    #println("50th residual=$(resnorms[p[50]])")

    if(getIndices & !(getSpurious))
        return maxres,ninside,s
    elseif (getIndices & (getSpurious))
        return maxres,ninside,s,spurious
    else
        return maxres,ninside
    end
end


function getInsideIndex(lest,emid,ra,rb)

    s=Array{Int64,1}()
    m0=size(lest,1)

    lestdif=abs.(real(lest.-emid)/abs.(ra)+im*imag(lest.-emid)/abs.(rb))

    @inbounds for k in 1:m0
        if(lestdif[k]<=1.0) #using 1.0 as the radius because we rescaled everything when forming lestdif
           push!(s,k)
        end
    end

    return s
end


function getLock(lest,x,res,emid,ra,rb,eps)
#get distances of estimated eigenvalues from middle of contour:
	#have to rescale the approximate eigenvalues to adjust for the fact that we are using an ellipse and not a circle
	#more complicated contours will require a different approach
	lestdif=abs.(real(lest.-emid)/abs.(ra)+im*imag(lest.-emid)/abs.(rb))
	(n,m0)=size(x)
	ninside=0
	@inbounds for k in 1:m0
	    if(lestdif[k]<=1.0)
	        ninside=ninside+1
	    end
	end

    resnorms=zeros(m0)
    @inbounds for i in 1:m0
        resnorms[i]=norm(res[:,i])/norm(x[:,i])
    end

    p=sortperm(resnorms)

    locked=Array{Int64,1}()

    maxres=0.0
    @inbounds for k in 1:m0
        if(lestdif[k]<=1.0 && resnorms[k]<=eps) #using 1.0 as the radius because we rescaled everything when forming lestdif
            push!(locked,k)
        end
    end

    #println("50th residual=$(resnorms[p[50]])")

    return locked
end


function getNotLock(lest,x,res,emid,ra,rb,eps)

    resnorms=zeros(m0)
    @inbounds for i in 1:m0
        resnorms[i]=norm(res[:,i])/norm(x[:,i])
    end

    notlocked=Array{Int64,1}()

    maxres=0.0
    @inbounds for k in 1:m0
        if(resnorms[k]>eps) #using 1.0 as the radius because we rescaled everything when forming lestdif
            push!(notlocked,k)
        end
    end

    #println("50th residual=$(resnorms[p[50]])")

    return notlocked
end

#solve reduced quadratic nonlinear problem with linearization
#select the m0 solutions that are closest to the center of the contour
function Quadrrsolve(Q,M,D,K,emid,ra,rb)
        (n,m0)=size(Q)
        #rayleigh ritz
        Mq=Matrix(Q'*(M*Q))
        Dq=Matrix(Q'*(D*Q))
        Kq=Matrix(Q'*(K*Q))

        #solve projected problem
        (Aq,Bq) = linquad(Mq,Dq,Kq)
        lest,xest=eigen(Matrix(Aq),Matrix(Bq))

        #get eigenvectors from linearization
        xall=Q*(xest[m0+1:2*m0,:])

	    #get distances of estimated eigenvalues from middle of contour:
	    #have to rescale the approximate eigenvalues to adjust for the fact that we are using an ellipse and not a circle
	    #more complicated contours will require a different approach
	    lestdif=abs.(real(lest.-emid)/abs.(ra)+im*imag(lest.-emid)/abs.(rb))

        #sort eigenpairs based on those distances:
        p=sortperm(lestdif)

        #take only the eigenpairs closest to the contour center:
        x=xall[:,p[1:m0]]
        lest=lest[p[1:m0]]
        return (lest,x)
end

#Beyn's method for nonlinear problems
#TinvApp = function for applying T^-1 to matrices
#Tapp=T(z)
function beynLoop(Tf,V,ncmax,emid,ra,rb,eps)
    (n,m0)=size(V)

    l=zeros(ComplexF64,m0)
    x=zeros(ComplexF64,n,m0)

    @inbounds for nc in 1:ncmax
        (l,x)=beyn(Tf,V,nc,emid,ra,rb)
        R=zeros(ComplexF64,n,m0)
        for i in 1:m0
            R[:,i]=Tf(l[i])*x[:,i]
        end
        res,ninside=getMaxResInside(l,x,R,emid,ra,rb)
        println("---Beyn cp $nc  $res")
        if(res<=eps)
            return (l,x)
        end
    end

    return (l,x)
end

function beyn(Tf,V::Array{ComplexF64,2},nc,emid,ra,rb; verbose=false)
    (n,m0)=size(V)

    (gk,wk)=trapezoidal(nc) #contour integral quadrature rule
    offset=pi/nc #offset angle of first quadrature point; makes sure it isn't on real axis for hermitian problems

    A0=zero(V)
    A1=zero(V)
    @inbounds for k in 1:nc
	#for k in 1:nc
	    if(verbose)
	        println("Beyn system $k of $nc")
	    end
	    theta=gk[k]*pi+pi+offset
	    z=emid+((ra+rb)/2)*exp(im*theta)+((ra-rb)/2)*exp(-1.0*im*theta)

        Qk=zeros(ComplexF64,n,m0)
        Tfz=Tf(z)

        Qk=\(Tfz,V)

	    A0=A0+wk[k]*(((ra+rb)/2)*exp(im*theta)-((ra-rb)/2)*exp(-1.0*im*theta))*Qk
	    A1=A1+wk[k]*(((ra+rb)/2)*exp(im*theta)-((ra-rb)/2)*exp(-1.0*im*theta))*z*Qk
    end

    (v0,s0,w0) = svd(A0)

    Av=v0'*A1*w0*diagm(0 => 1 ./ s0)

    l,xv=eigen(Av)

    return (l,v0*xv)
end


function ibeyn(Tf,V::Array{ComplexF64,2},alpha,isMaxit,nc,emid,ra,rb; verbose=false)
    (n,m0)=size(V)

    (gk,wk)=trapezoidal(nc) #contour integral quadrature rule
    offset=pi/nc #offset angle of first quadrature point; makes sure it isn't on real axis for hermitian problems

    A0=zeros(V)
    A1=zeros(V)
    @inbounds for k in 1:nc
	    if(verbose)
	        println("Beyn system $k of $nc")
	    end
	    theta=gk[k]*pi+pi+offset
	    z=emid+((ra+rb)/2)*exp(im*theta)+((ra-rb)/2)*exp(-1.0*im*theta)

        Qk=zeros(ComplexF64,n,m0)
        Tfz=Tf(z)

        #if(typeof(Tfz)==SparseMatrixCSC{Complex{Float64},Int64})
        #    Qk=\(Tfz,V)
        #else
        #    Qk=\(Tfz,V)
        #end

        for i in 1:m0
            #Qk[:,i]=bicgstabl(Tfz,V[:,i],1,max_mv_products=isMaxit,tol=alpha,initial_zero=true,log=false)
            #Qk[:,i]=gmres(Tfz,V[:,i],restart=isMaxit,tol=alpha,initially_zero=true,maxiter=isMaxit,log=false)
            Qk[:,i]=minres(Tfz,V[:,i],tol=alpha,initially_zero=true,maxiter=isMaxit,log=false)
        end

	    A0=A0+wk[k]*(((ra+rb)/2)*exp(im*theta)-((ra-rb)/2)*exp(-1.0*im*theta))*Qk
	    A1=A1+wk[k]*(((ra+rb)/2)*exp(im*theta)-((ra-rb)/2)*exp(-1.0*im*theta))*z*Qk
    end

    (v0,s0,w0) = svd(A0)

    Av=v0'*A1*w0*diagm(1 ./ s0)

    (l,xv)=eigen(Av)

    return (l,v0*xv)
end



function beynscan(emin,emax,ndiv,Tf,nc,n,m0; verbose=false)

    eigs=Array{ComplexF64,1}()
    resvals=Array{ComplexF64,1}()

    edges=linspace(emin,emax,ndiv+1)
    x0=rand(ComplexF64,n,m0)

    @inbounds for i in 1:ndiv
        if(verbose)
            println("Div $i of $ndiv")
        end
        emid=(edges[i]+edges[i+1])/2.0
        ra=(edges[i]-edges[i+1])/2.0
        rb=ra

        (lest,xest)=beyn(Tf,x0,nc,emid,ra,rb)

        res=zeros(x0)
        for j in 1:m0
            res[:,j]=Tf(lest[j])*xest[:,j]
        end
        (maxres,ninside,insideIndices)=getMaxResInside(lest,xest,res,emid,ra,rb;getIndices=true)

        for j in 1:ninside
            push!(eigs,lest[insideIndices[j]])
            tmp=norm(res[:,insideIndices[j]])/norm(xest[:,insideIndices[j]])
            push!(resvals,tmp)
            if(verbose)
                println("   $(lest[insideIndices[j]])   $tmp")
            end
        end
    end

    return eigs,resvals
end




function trapezoidalCont(nc)
    offset=pi/nc
    a=offset
    b=2*pi+offset

    dx=(b-a)/nc

    weights=zeros(nc)
    points=zeros(nc)

    weights[1]=dx
    points[1]=a
    @inbounds for i in 2:nc
        points[i]=points[i-1]+dx
        weights[i]=dx
    end

    return (points,weights)
end
