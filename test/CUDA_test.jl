using SparseArrays, Krylov, LinearOperators
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using JLD2
using LinearAlgebra
using IncompleteLU
using SuiteSparse

@load "C:/Users/13733/Desktop/DistributionPowerFlow/J.jld2" J
@load "C:/Users/13733/Desktop/DistributionPowerFlow/F.jld2" F
A_cpu = sprand(ComplexF64, 10000, 10000, 0.1) + I * 1e-8  # 添加小的扰动到对角线
b_cpu = rand(ComplexF64, 10000)
# A_cpu=sparse(J)
# b_cpu=F
  # Optional -- Compute a permutation vector p such that A[:,p] has no zero diagonal
  p = zfd(A_cpu)
  p .+= 1
  A_cpu = A_cpu[:,p]

  # Transfer the linear system from the CPU to the GPU
  A_gpu = CuSparseMatrixCSR(A_cpu)  # A_gpu = CuSparseMatrixCSC(A_cpu)
  b_gpu = CuVector(b_cpu)

  # ILU(0) decomposition LU ≈ A for CuSparseMatrixCSC or CuSparseMatrixCSR matrices
  P = ilu02(A_gpu)

  # Additional vector required for solving triangular systems
  n = length(b_gpu)
  T = eltype(b_gpu)
  z = CUDA.zeros(T, n)

  # Solve Py = x
  function ldiv_ilu0!(P::CuSparseMatrixCSR, x, y, z)
    ldiv!(z, UnitLowerTriangular(P), x)  # Forward substitution with L
    ldiv!(y, UpperTriangular(P), z)      # Backward substitution with U
    return y
  end

  function ldiv_ilu0!(P::CuSparseMatrixCSC, x, y, z)
    ldiv!(z, LowerTriangular(P), x)      # Forward substitution with L
    ldiv!(y, UnitUpperTriangular(P), z)  # Backward substitution with U
    return y
  end

  # Operator that model P⁻¹
  symmetric = hermitian = false
  opM = LinearOperator(T, n, n, symmetric, hermitian, (y, x) -> ldiv_ilu0!(P, x, y, z))

  # Solve a non-Hermitian system with an ILU(0) preconditioner on GPU
  @time x̄, stats = bicgstab(A_gpu, -b_gpu, M=opM)

  # Recover the solution of Ax = b with the solution of A[:,p]x̄ = b
  invp = invperm(p)
  dx = x̄[invp]



# Function for ILU(0) on the CPU using IncompleteLU
using ILUZero, Krylov
Pᵣ = ilu0(A_cpu)
@time x, stats = bicgstab(A_cpu, -b_cpu, N=Pᵣ, ldiv=true)  # right preconditioning