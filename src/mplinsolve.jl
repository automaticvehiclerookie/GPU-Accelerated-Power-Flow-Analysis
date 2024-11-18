# todo list:
# 1. test the gpu based algorithms
# 2. test the cgs solvers
_get_type(J::CuSparseMatrixCSR) = CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
_is_csr(J::CuSparseMatrixCSR) = true
function mplinsolve(A, b, solver = "", opt = nothing)
    info = nothing

    # if solver in ["", "\\"]
    #     x = A \ b
    # elseif solver == "LU3"
    #     q = amd(A)
    #     if issparse(A)
    #         L, U, p = lu(A[q,q], Val(true))
    #     else
    #         L, U, p = lu(A[q,q])
    #     end
    #     x = zeros(size(A, 1))
    #     x[q] = U \ (L \ b[q[p]])
    # elseif solver == "LU3a"
    #     q = amd(A)
    #     L, U, p = lu(A[q,q])
    #     x = zeros(size(A, 1))
    #     x[q] = U \ (L \ b[q[p]])
    # elseif solver == "LU4"
    #     L, U, p, q = lu(A)
    #     x = zeros(size(A, 1))
    #     x[q] = U \ (L \ b[p])
    # elseif solver == "LU5"
    #     L, U, p, q, R = lu(A)
    #     x = zeros(size(A, 1))
    #     x[q] = U \ (L \ (R[:, p] \ b))
    # elseif solver == "cholesky"
    #     factor = cholesky(A)
    #     x = factor \ b
    # elseif solver == "gmres"
    #     ilu_fact = ilu(A)
    #     x = IterativeSolvers.gmres(A, b, Pl=ilu_fact, reltol=1e-8, maxiter = 1000)
    # elseif solver == "bicgstab"
    #     n = size(A,1)
    #     F = ilu(A, τ = 0.05)
    #     opM = LinearOperator(Float64, n, n, false, false, (y, v) -> forward_substitution!(y, F, v))
    #     opN = LinearOperator(Float64, n, n, false, false, (y, v) -> backward_substitution!(y, F, v))
    #     x , stats = bicgstab(A, b, history=false, M=opM, N=opN)
    # elseif solver == "cgs"
    #     x = cgs(A, b, rtol=1e-8, itmax=1000)
    if solver == "gpu"
        if Sys.iswindows()
            # n=length(b)
            # handle = CUDA.CUSOLVER.cusolverSpCreate()

            # # Step 4: Create matrix descriptor
            # descrA = Ref{Ptr{Nothing}}()  # Initialize a reference to hold the descriptor
            # ccall((:cusparseCreateMatDescr, CUDA.libcusparse), CUDA.CUSPARSE.cusparseStatus_t, (Ref{Ptr{Nothing}},), descrA)
            # ccall((:cusparseSetMatIndexBase, CUDA.libcusparse), CUDA.CUSPARSE.cusparseStatus_t, (Ptr{Nothing}, Cint), descrA[], CUDA.CUSPARSE.CUSPARSE_INDEX_BASE_ONE)
            # ccall((:cusparseSetMatType, CUDA.libcusparse), CUDA.CUSPARSE.cusparseStatus_t, (Ptr{Nothing}, Cint), descrA[], CUDA.CUSPARSE.CUSPARSE_MATRIX_TYPE_GENERAL)

            # # Step 5: Placeholder for solution x on GPU (must be Float32)
            # x = CUDA.fill(Float64(0.0), n)
            # info = Ref{Cint}()
            # tol = Float64(1e-6)
            # CUDA.CUSOLVER.cusolverSpDcsrlsvqr(
            #     handle,
            #     n,
            #     nnz(A),
            #     descrA[],              # Matrix descriptor (dereference the pointer)
            #     A.nzVal,                 # Non-zero values (CSR format)
            #     A.rowPtr,                # Row pointers (CSR format)
            #     A.colVal,                # Column indices (CSR format)
            #     b,                   # Right-hand side vector b
            #     tol,                   # Tolerance for convergence
            #     1,                     # Reordering enabled for stability
            #     x,                   # Solution vector on the GPU
            #     info                   # Output information about the solution status
            # )
            # CUDA.CUSOLVER.cusolverSpDestroy(handle)
            # ccall((:cusparseDestroyMatDescr, CUDA.libcusparse), CUDA.CUSPARSE.cusparseStatus_t, (Ptr{Nothing},), descrA[])
            #======================#
            #BlockJacobiPreconditioner
            # n=length(b)
            # m=length(b)
            # device=CUDABackend()
            # # x♯=CUDA.rand(n)
            # x = similar(b)
            # nblocks = 2
            # if _is_csr(A)
            #     scaling_csr!(A, b, device)
            # end
            # precond = BlockJacobiPreconditioner(A, nblocks, device)
            # update!(precond, A)
            # S = _get_type(A)
            # linear_solver = Krylov.BicgstabSolver(n, m, S)
            # Krylov.bicgstab!(
            #     linear_solver, A, b;
            #     N=precond,
            #     atol=1e-10,
            #     rtol=1e-10,
            #     verbose=0,
            #     history=true,
            # )
            # # n_iters = linear_solver.stats.niter
            # copyto!(x, linear_solver.x)
            #======================#
            #ILUZeroPreconditioner
            P = kp_ilu0(A)
            x, stats = Krylov.gmres(A, b, N=P, ldiv=true)
            stats=0
        end
  
    end

    return x, stats
end