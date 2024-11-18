using CUDA
using CUDA.CUSPARSE
using CUDA.CUSOLVER
using SparseArrays
using LinearAlgebra
using JLD2
using SuiteSparse
using BenchmarkTools
@load "C:/Users/13733/Desktop/DistributionPowerFlow/J9241" J9241
@load "C:/Users/13733/Desktop/DistributionPowerFlow/F9241" F9241
A=sparse(J9241)
A=CuSparseMatrixCSR(A)
b=-F9241
b=CuArray(b)
gpu_time = @belapsed begin
    n=length(b)
    handle = CUDA.CUSOLVER.cusolverSpCreate()

    # Step 4: Create matrix descriptor
    descrA = Ref{Ptr{Nothing}}()  # Initialize a reference to hold the descriptor
    ccall((:cusparseCreateMatDescr, CUDA.libcusparse), CUDA.CUSPARSE.cusparseStatus_t, (Ref{Ptr{Nothing}},), descrA)
    ccall((:cusparseSetMatIndexBase, CUDA.libcusparse), CUDA.CUSPARSE.cusparseStatus_t, (Ptr{Nothing}, Cint), descrA[], CUDA.CUSPARSE.CUSPARSE_INDEX_BASE_ONE)
    ccall((:cusparseSetMatType, CUDA.libcusparse), CUDA.CUSPARSE.cusparseStatus_t, (Ptr{Nothing}, Cint), descrA[], CUDA.CUSPARSE.CUSPARSE_MATRIX_TYPE_GENERAL)

    # Step 5: Placeholder for solution x on GPU (must be Float32)
    x = CUDA.fill(Float64(0.0), n)
    info = Ref{Cint}()
    tol = Float64(1e-6)
    CUDA.CUSOLVER.cusolverSpDcsrlsvqr(
        handle,
        n,
        nnz(A),
        descrA[],              # Matrix descriptor (dereference the pointer)
        A.nzVal,                 # Non-zero values (CSR format)
        A.rowPtr,                # Row pointers (CSR format)
        A.colVal,                # Column indices (CSR format)
        b,                   # Right-hand side vector b
        tol,                   # Tolerance for convergence
        1,                     # Reordering enabled for stability
        x,                   # Solution vector on the GPU
        info                   # Output information about the solution status
    )
    CUDA.CUSOLVER.cusolverSpDestroy(handle)
    ccall((:cusparseDestroyMatDescr, CUDA.libcusparse), CUDA.CUSPARSE.cusparseStatus_t, (Ptr{Nothing},), descrA[])

end
J=sparse(J9241)
cpu_time = @belapsed begin
    a = qr(J)
    x = a\ (-F9241)
end
println("GPU time: $gpu_time")
println("CPU time: $cpu_time")
    