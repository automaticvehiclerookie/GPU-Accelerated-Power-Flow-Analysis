
using CUDA
using Krylov, LinearOperators, IncompleteLU, HarwellRutherfordBoeing
using LinearAlgebra, Printf, SuiteSparseMatrixCollection, SparseArrays

#load the database of the SuiteSparseMatrixCollection
ssmc = ssmc_db(verbose=false)
#load the matrix from the SuiteSparseMatrixCollection
matrix = ssmc_matrices(ssmc, "HB", "sherman1")
#fetch the matrix from the SuiteSparseMatrixCollection
#the provided format is RB
path = fetch_ssmc(matrix, format="RB")
#number of rows of the matrix
n = matrix.nrows[1]
#load the matrix from the file
A = RutherfordBoeingData(joinpath(path[1], "$(matrix.name[1]).rb")).data
#right-hand side
b = A * ones(n)
#Solve Ax = b with CUDA 
# Convert A and b to CuSparseMatrixCSR and CuArray, respectively
A_gpu = CuMatrix(A)
b_gpu = CuVector(b)
# Solve Ax = b using the GMRES method
@time x_gpu = Krylov.gmres(A_gpu, b_gpu)
x_gpu=x_gpu[1]
r= b_gpu - A_gpu * x_gpu
@printf("[GMRES]Residual norm: %8.1e\n", norm(r))
# Solve Ax = b using the BiCGSTAB method
@time x_gpu = Krylov.bicgstab(A_gpu, b_gpu)
x_gpu=x_gpu[1]
r= b_gpu - A_gpu * x_gpu
@printf("[BICGSTAB]Residual norm: %8.1e\n", norm(r))
# Solve Ax = b using the LSQR method
@time x_gpu = Krylov.lsqr(A_gpu, b_gpu)
x_gpu=x_gpu[1]
r= b_gpu - A_gpu * x_gpu
@printf("[LSQR]Residual norm: %8.1e\n", norm(r))
# Solve Ax = b using the LSLQ method
@time x_gpu = Krylov.lslq(A_gpu, b_gpu)
x_gpu=x_gpu[1]
r= b_gpu - A_gpu * x_gpu
@printf("[LSLQ]Residual norm: %8.1e\n", norm(r))
# Solve Ax = b using the MINRES method
@time x_gpu = Krylov.minres(A_gpu, b_gpu)
x_gpu=x_gpu[1]
r= b_gpu - A_gpu * x_gpu
@printf("[MINRES]Residual norm: %8.1e\n", norm(r))
# Solve Ax = b using the QMR method
@time x_gpu = Krylov.qmr(A_gpu, b_gpu)
x_gpu=x_gpu[1]
r= b_gpu - A_gpu * x_gpu
@printf("[QMR]Residual norm: %8.1e\n", norm(r))
# Solve Ax = b using the CGS method
@time x_gpu = Krylov.cgs(A_gpu, b_gpu)
x_gpu=x_gpu[1]
r= b_gpu - A_gpu * x_gpu
@printf("[CGS]Residual norm: %8.1e\n", norm(r))
# Solve Ax = b using the BiCGSTAB method with a preconditioner
F = ilu(A, τ = 0.05)
opM = LinearOperator(Float64, n, n, false, false, (y, v) -> forward_substitution!(y, F, v))
opN = LinearOperator(Float64, n, n, false, false, (y, v) -> backward_substitution!(y, F, v))
opP = LinearOperator(Float64, n, n, false, false, (y, v) -> ldiv!(y, F, v))
@time x, stats = bicgstab(A, b, history=true, M=opM, N=opN)
r= b - A * x
@printf("[BICGSTAB ILU PRECONDITIONER]Residual norm: %8.1e\n", norm(r))

# b_precond = F \ b
# # Move A and the preconditioned b to the GPU
# A_gpu = CuArray(A)
# b_gpu = CuArray(b_precond)
# # Solve Ax = b using BiCGSTAB
# @time x_gpu, stats = Krylov.bicgstab(A_gpu, b_gpu, history=true)
# # Compute the residual on the CPU
# r = b - A * Array(x_gpu)
# @printf("[BICGSTAB PRECONDITIONER]Residual norm: %8.1e\n", norm(r))

# # Solve Ax = b using the LSQR method with a preconditioner
# # Get the dimensions of A
# n, m = size(A)
# # Define the linear operator for the preconditioner
# opM = LinearOperator(Float64, n, m, false, false, (y, v) -> ldiv!(y, F, v))
# # Solve the system using the LSQR method with the preconditioner
# @time x, stats = Krylov.lsqr(A, b, M=opM)

# Solve Ax = b using the CGS method with a preconditioner
# Move A and b to the GPU
A_gpu = CUDA.CuArray(A)
b_gpu = CUDA.CuArray(b)
# Compute the Jacobi preconditioner on the CPU and move it to the GPU
M_cpu = Diagonal(1 ./ diag(A))
M_gpu = CUDA.CuArray(M_cpu)
# Define the linear operator for the preconditioner
opM = LinearOperator(Float64, size(A, 1), size(A, 2), false, false, (y, v) -> mul!(y, M_gpu, v))
# Solve Ax = b using CGS with preconditioner
@time x, stats = Krylov.cgs(A_gpu, b_gpu,history=true, M=opM)
r= b_gpu - A_gpu * x
@printf("[CGS JACOBI PRECONDITIONER]Residual norm: %8.1e\n", norm(r))

F = ilu(A, τ = 0.05)
opM = LinearOperator(Float64, n, n, false, false, (y, v) -> forward_substitution!(y, F, v))
opN = LinearOperator(Float64, n, n, false, false, (y, v) -> backward_substitution!(y, F, v))
opP = LinearOperator(Float64, n, n, false, false, (y, v) -> ldiv!(y, F, v))
@time x, stats = cgs(A, b, history=true, M=opM, N=opN)
r= b - A * x
@printf("[CGS ILU PRECONDITIONER]Residual norm: %8.1e\n", norm(r))

# Solve Ax = b using the GMRES method with a preconditioner
# Move A and b to the GPU
A_gpu = CUDA.CuArray(A)
b_gpu = CUDA.CuArray(b)
# Compute the Jacobi preconditioner on the CPU and move it to the GPU
M_cpu = Diagonal(1 ./ diag(A))
M_gpu = CUDA.CuArray(M_cpu)
# Define the linear operator for the preconditioner
opM = LinearOperator(Float64, size(A, 1), size(A, 2), false, false, (y, v) -> mul!(y, M_gpu, v))
# Solve Ax = b using GMRES with preconditioner
@time x, stats = Krylov.gmres(A_gpu, b_gpu, M=opM)
r= b_gpu - A_gpu * x
@printf("[GRMES PRECONDITIONER]Residual norm: %8.1e\n", norm(r))