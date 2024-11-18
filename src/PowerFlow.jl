"""
    Define the power flow module with different functions
"""

module PowerFlow
    using Printf
    using SparseArrays
    using LinearAlgebra
    using PrettyTables
    using AMD
    using SuiteSparse
    using IterativeSolvers
    using IncompleteLU
    using KrylovKit
    using Krylov
    using LinearOperators
    using CUDA.CUSPARSE
    using CUDA
    using CUDA.CUSOLVER
    using CUDA.CUBLAS
    using AlgebraicMultigrid
    using JuMP
    using Ipopt
    using KrylovPreconditioners
    # using KrylovPreconditioners
    # using different packages based on the operating system
    if Sys.iswindows()
        using CUDA
    # else
    #     using Metal
    end
    include("idx.jl")
    include("bustypes.jl")
    include("ext2int.jl")
    include("makeYbus.jl")
    include("newtonpf.jl")
    include("makeSbus.jl")
    include("makeSdzip.jl")
    include("mplinsolve.jl")
    include("total_load.jl")
    include("pfsoln.jl")
    include("dSbus_dV.jl")
    include("runpf.jl")
    include("settings.jl")
    include("rundcpf.jl")
    include("makeBdc.jl")
    include("dcpf.jl")
    include("gpu_gmres.jl")
    include("run_hybridpf.jl")
    include("hybrid_bustypes.jl")
    include("makehYbus.jl")
    include("makehBdc.jl")
    include("runmodel.jl")
    include("hybridnewtonpf.jl")
    include("kernel_index!.jl")
    include("makeSbus_gpu.jl")
    include("makeSdzip_gpu.jl")
    include("Parallel_execution_jacobi.jl")
    include("dividecuarray.jl")
    include("kernel_populate_csr!.jl")
    include("allocate_memory.jl")
    # include("data_structure.jl")
    # export idx_bus, idx_brch, idx_gen, bustypes, makeYbus, newtonpf, makeSbus, makeSdzip, mplinsolve, total_load, pfsoln, dSbus_dV, MPC
end

export PowerFlow