using CUDA
using SparseArrays
using LinearAlgebra
using JLD2

# Step 1: Load the matrix J and vector F from files
@load "C:/Users/13733/Desktop/DistributionPowerFlow/J.jld2" J
@load "C:/Users/13733/Desktop/DistributionPowerFlow/F.jld2" F
@load "C:/Users/13733/Desktop/DistributionPowerFlow/d_col.jld2" d_col
@load "C:/Users/13733/Desktop/DistributionPowerFlow/d_val.jld2" d_val
@load "C:/Users/13733/Desktop/DistributionPowerFlow/d_row_ptrs.jld2" d_row_ptrs
@load "C:/Users/13733/Desktop/DistributionPowerFlow/nzz.jld2" nzz

# Convert matrix J and vector F to Float32 and move to GPU
J = sparse(Float32.(J))   # Sparse matrix J
F = Float32.(F)           # Right-hand side vector F
d_J = CUDA.CUSPARSE.CuSparseMatrixCSR(J)  # Convert J to CSR format and move to GPU
d_F = CuArray(F)                  # Move F to GPU
n = length(F)
d_x = CUDA.fill(Float32(0.0), n)  # Solution vector on GPU
d_col = CuArray(d_col)
d_col = Int32.(d_col)
d_val = CuArray(Float32.(d_val))  # Ensure values are in Float32
d_row_ptrs = CuArray(d_row_ptrs)
d_row_ptrs .+= 1
d_row_ptrs = Int32.(d_row_ptrs)

# Step 2: Create cuSOLVER handle and matrix descriptor
handle = CUDA.CUSOLVER.cusolverSpCreate()
descrA = Ref{Ptr{Nothing}}()  # Matrix descriptor for cuSparse
ccall((:cusparseCreateMatDescr, CUDA.libcusparse), CUDA.CUSPARSE.cusparseStatus_t, (Ref{Ptr{Nothing}},), descrA)
ccall((:cusparseSetMatIndexBase, CUDA.libcusparse), CUDA.CUSPARSE.cusparseStatus_t, (Ptr{Nothing}, Cint), descrA[], CUDA.CUSPARSE.CUSPARSE_INDEX_BASE_ONE)
ccall((:cusparseSetMatType, CUDA.libcusparse), CUDA.CUSPARSE.cusparseStatus_t, (Ptr{Nothing}, Cint), descrA[], CUDA.CUSPARSE.CUSPARSE_MATRIX_TYPE_GENERAL)

# Step 3: LU factorization using cusolverSp with single precision
info = Ref{Cint}()
tol = 1e-6  # Tolerance

@time CUDA.CUSOLVER.cusolverSpScsrlsvqr(
    handle,
    n,
    nzz,
    descrA[],          # Matrix descriptor
    d_val,             # Non-zero values (CSR) in Float32
    d_row_ptrs,        # Row pointer (CSR)
    d_col,             # Column indices (CSR)
    d_F,               # Right-hand side vector
    tol,               # Tolerance
    1,                 # Reordering enabled
    d_x,               # Solution vector on GPU
    info               # Info status
)

# Step 4: Check for errors in LU factorization
if info[] != 0
    println("LU factorization failed with error code: ", info[])

else
    println("LU factorization succeeded")
end

# Step 5: Transfer the solution back to the CPU
x_host = Array(d_x)

# Display the solution
println("Solution x = ", x_host)

# Step 6: Clean up cuSolver and cuSparse resources
CUDA.CUSOLVER.cusolverSpDestroy(handle)
ccall((:cusparseDestroyMatDescr, CUDA.libcusparse), CUDA.CUSPARSE.cusparseStatus_t, (Ptr{Nothing},), descrA[])
