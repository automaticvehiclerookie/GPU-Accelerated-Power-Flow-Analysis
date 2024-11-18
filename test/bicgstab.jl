using CUDA
using SparseArrays
using LinearAlgebra
using Krylov
using BenchmarkTools
using JLD2
using KrylovPreconditioners
using CUDA.CUSPARSE
_get_type(J::CuSparseMatrixCSR) = CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
_is_csr(J::CuSparseMatrixCSR) = true
# Function： Using BiCGStab on CPU to solve the sparse matrix linear equation
function solve_sparse_cpu(A, b)
    P⁻¹ = BlockJacobiPreconditioner(A)
    x=Krylov.bicgstab(A, b,M=P⁻¹)
    return x
end

# Function： Using BiCGStab on GPU to solve the sparse matrix linear equation
function solve_sparse_gpu(A, b,n,m,device=CUDABackend())
    x = similar(b); 
    # r = similar(b)
    nblocks = 2
    if _is_csr(A)
        scaling_csr!(A, b, device)
    end
    precond = BlockJacobiPreconditioner(A, nblocks, device)
    update!(precond, A)
    S = _get_type(A)
    linear_solver = Krylov.BicgstabSolver(n, m, S)
    Krylov.bicgstab!(
        linear_solver, A, b;
        N=precond,
        atol=1e-10,
        rtol=1e-10,
        verbose=0,
        history=true,
    )
    # n_iters = linear_solver.stats.niter
    copyto!(x, linear_solver.x)
    return x  
end

@load "C:/Users/13733/Desktop/DistributionPowerFlow/J.jld2" J
@load "C:/Users/13733/Desktop/DistributionPowerFlow/F.jld2" F
J_cpu=J
F_cpu=F

matrix_sizes = size(J,1)

for n in matrix_sizes
    println("The scale of matrix A: $n x $n")

    A = sparse(J)  
    b = -F
    A_cpu=sparse(J_cpu)
    b_cpu=-F_cpu
    device=CUDABackend()
    AT=CuArray
    SMT=CuSparseMatrixCSR
    n=length(b)
    m=length(b)
    x♯=rand(n)
    A = A |> SMT
    b = b |> AT
    x♯ = x♯ |> AT        
    #CPU time consumption
    cpu_time = @belapsed solve_sparse_cpu($A_cpu, $b_cpu)
    println("CPU time consumption $cpu_time seconds")

    # GPU time consumption
    gpu_time = @belapsed solve_sparse_gpu($A, $b,$n,$m)
    println("GPU time consumption: $gpu_time seconds")

    println("speed up (CPU / GPU): $(cpu_time / gpu_time)")

    println("=====================================")
end
