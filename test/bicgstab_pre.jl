using LinearAlgebra
using SparseArrays
using CUDA
using Krylov
function bicgstab_gpu(A, b; tol=1e-6, maxit=20, M1=nothing, M2=nothing, x0=nothing)
    m, n = size(A)

    maxit = max(maxit, 0)

    n2b = norm(b)
    x = CUDA.zeros(Float32,n)

    # if M1 != nothing && size(M1) != (m, m)
    #     error("Preconditioner M1 must be the same size as A")
    # end

    # if M2 != nothing && size(M2) != (m, m)
    #     error("Preconditioner M2 must be the same size as A")
    # end

    flag = 1
    xmin = x
    imin = 0
    tolb = tol * n2b
    r = b - mul!(CUDA.zeros(Float32, m), A, x)
    normr = norm(r)
    normr_act = copy(normr)

    if normr <= tolb
        return x, 0, normr / n2b, 0, [normr]
    end

    rt = copy(r)
    resvec = CUDA.zeros(Float32,2 * maxit + 1)
    CUDA.@sync begin
        CUDA.fill!(view(resvec, 1:1), normr)
    end
    normrmin = copy(normr)
    rho = 1
    omega = 1
    stag = 0
    moresteps = 0
    maxmsteps = min(floor(Int, n / 50), 10, n - maxit)
    maxstagsteps = 3
    alpha = 1.0  # 初始化 alpha 为标量
    p = copy(r)  # 初始化 p
    v = CUDA.zeros(Float32,n)  # 初始化 v
    ii = 0  # 初始化 ii

    for ii in 1:maxit
        rho1 = rho
        rho = dot(rt, r)
        if rho == 0.0 || isinf(rho)
            flag = 4
            resvec = resvec[1:2*ii-1]
            break
        end

        if ii == 1
            p = r
        else
            beta = (rho / rho1) * (alpha / omega)
            if beta == 0 || !isfinite(beta)
                flag = 4
                break
            end
            p = r + beta * (p - omega * v)
        end
            ph = p
        v = A * ph
        rtv = dot(rt, v)
        if rtv == 0 || isinf(rtv)
            flag = 4
            resvec = resvec[1:2*ii-1]
            break
        end

        alpha = rho / rtv
        if isinf(alpha)
            flag = 4
            resvec = resvec[1:2*ii-1]
            break
        end

        if abs(alpha) * norm(ph) < eps() * norm(x)
            stag += 1
        else
            stag = 0
        end

        xhalf = x + alpha * ph
        s = r - alpha * v
        normr = norm(s)
        normr_act = normr
        resvec[2*ii:2*ii] = normr

        if normr <= tolb || stag >= maxstagsteps || moresteps > 0
            s = b - A * xhalf
            normr_act = norm(s)
            resvec[2*ii] = normr_act
            if normr_act <= tolb
                x = xhalf
                flag = 0
                iter = ii - 0.5
                resvec = resvec[1:2*ii]
                break
            else
                if stag >= maxstagsteps && moresteps == 0
                    stag = 0
                end
                moresteps += 1
                if moresteps >= maxmsteps
                    @warn "Tolerance is too small"
                    flag = 3
                    x = xhalf
                    resvec = resvec[1:2*ii]
                    break
                end
            end
        end

        if stag >= maxstagsteps
            flag = 3
            resvec = resvec[1:2*ii]
            break
        end

        if normr_act < normrmin
            normrmin = normr_act
            xmin = xhalf
            imin = ii - 0.5
        end
            sh = s
        if M2 != nothing
            sh = M2 \ sh
            if !all(isfinite(sh))
                flag = 2
                resvec = resvec[1:2*ii]
                break
            end
        end

        t = A * sh
        tt = dot(t, t)
        if tt == 0 || isinf(tt)
            flag = 4
            resvec = resvec[1:2*ii]
            break
        end

        omega = dot(t, s) / tt
        if isinf(omega)
            flag = 4
            resvec = resvec[1:2*ii]
            break
        end

        if abs(omega) * norm(sh) < eps() * norm(xhalf)
            stag += 1
        else
            stag = 0
        end

        x = xhalf + omega * sh
        r = s - omega * t
        normr = norm(r)
        normr_act = normr
        resvec[2*ii+1:2*ii+1] = normr

        if normr <= tolb || stag >= maxstagsteps || moresteps > 0
            r = b - A * x
            normr_act = norm(r)
            resvec[2*ii+1] = normr_act
            if normr_act <= tolb
                flag = 0
                iter = ii
                resvec = resvec[1:2*ii+1]
                break
            else
                if stag >= maxstagsteps && moresteps == 0
                    stag = 0
                end
                moresteps += 1
                if moresteps >= maxmsteps
                    @warn "Tolerance is too small"
                    flag = 3
                    resvec = resvec[1:2*ii+1]
                    break
                end
            end
        end

        if normr_act < normrmin
            normrmin = normr_act
            xmin = x
            imin = ii
        end

        if stag >= maxstagsteps
            flag = 3
            resvec = resvec[1:2*ii+1]
            break
        end
    end

    if isempty(ii)
        ii = 0
    end

    if flag == 0
        relres = normr_act / n2b
    else
        r = b - A * xmin
        if norm(r) <= normr_act
            x = xmin
            iter = imin
            relres = norm(r) / n2b
        else
            iter = ii
            relres = normr_act / n2b
        end
    end

    return x, flag, relres, iter, resvec
end

# 示例矩阵和向量
n = 100
A = sprand(n, n, 0.1) + I
A=CuArray(A)
b = rand(n)
b=CuArray(b)
# 使用自定义的 BiCGSTAB 函数求解
@time x, flag, relres, iter, resvec = bicgstab_gpu(A, b, tol=1e-8, maxit=1000)

# 打印结果
# println("Solution: ", x)
# println("Flag: ", flag)
# println("Relative Residual: ", relres)
# println("Iterations: ", iter)
# println("Residual Vector: ", resvec)

# # 使用 Krylov 库的 bicgstab 函数求解
@time x_krylov, stats = Krylov.bicgstab(A, b, atol=1e-8, itmax=1000)
# # 打印结果
# mis = x_krylov - x_gpu_cpu
# plot(mis, title="Difference between Krylov and GPU BiCGSTAB", xlabel="Index", ylabel="Difference")

#================================================================================#
using LinearAlgebra
using SparseArrays
using CUDA
using Krylov

using CUDA
using LinearAlgebra
#A kernel function test for BiCGSTAB with preconditioner
# The preconditioner is a simple LU preconditioner
# The matrix is a 2D Laplacian matrix
# The right hand side is a random vector
# The initial guess is a zero vector
# The solution is the result of BiCGSTAB
# The error is the difference between the solution and the true solution
# The error is expected to be small
function preconditionAWed_BICGSTAB(b_gpu,A_gpu,tol)
    x0 = CUDA.zeros(Float32,length(b_gpu))
    r0=b_gpu-mul!(CUDA.zeros(Float32,length(b_gpu)),A_gpu,x0)
    rt0=copy(r0)
    rho0=1.0
    alpha=1.0
    omega0=1.0
    r1=copy(r0)
    u0=0.0
    p0=0.0
    while norm(r1)>tol
       rho=dot(rt0,r0)
       beta=(rho/rho0)*(alpha/omega0)
       p==r0+beta*(p0-omega0*u0)
    end
end
A_gpu = CUDA.rand(100,100)
b_gpu = CUDA.rand(100)
tol=1e-6
@time x=preconditionAWed_BICGSTAB(b_gpu,A_gpu,tol)