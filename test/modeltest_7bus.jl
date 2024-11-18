using SparseArrays, Random, Test
using LinearAlgebra, Krylov, KrylovPreconditioners
using CUDA.CUSPARSE
using CUDA
using JLD2
using ILUZero
_get_type(J::CuSparseMatrixCSR) = CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
_is_csr(J::CuSparseMatrixCSR) = true
@load "C:/Users/13733/Desktop/DistributionPowerFlow/J.jld2" J
@load "C:/Users/13733/Desktop/DistributionPowerFlow/F.jld2" F
@load "C:/Users/13733/Desktop/DistributionPowerFlow/J1888.jld2" J1888
# J=J9241
# F=F9241
    FC =ComplexF64
    V= CuVector{FC}
    M=CuSparseMatrixCSR{FC}
    n = length(F9241)
    R = real(FC)
    A_cpu = J9241
    A_cpu = sparse(A_cpu)
    b_cpu = -F9241
  
    A_gpu = M(A_cpu)
    b_gpu = V(b_cpu)
    P = kp_ilu0(A_gpu)
  
    @time x_gpu, stats = gmres(A_gpu, b_gpu, N=P, ldiv=true)

    Pᵣ = ilu0(A_cpu)
    @time x, stats = gmres(A_cpu, b_cpu, N=Pᵣ, ldiv=true)  # right preconditioning