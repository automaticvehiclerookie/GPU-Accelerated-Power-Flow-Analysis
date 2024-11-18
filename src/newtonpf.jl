
# Define the newtonpf function
function newtonpf(Ybus, Sbus_gpu, V0, ref, pv, pq, tol0, max_it0, alg="gpu")
    tol = tol0
    max_it = max_it0
    # Initialize
    converged = false
    i = 0
    V = V0
    Va = angle.(V)
    Vm = abs.(V)
    nb = length(V)
    #transfer to GPU
    V_gpu = CuArray(V)
    Va_gpu = CuArray(Va)
    Vm_gpu = CuArray(Vm)
    Ybus_gpu =CuSparseMatrixCSC(Ybus)
    pv_gpu = CuArray(pv)
    pq_gpu = CuArray(pq)
    # Set up indexing for updating V
    npv = length(pv)
    npq = length(pq)
    Cpv_pq_index =sparse(1:(npv+npq), vcat(pv, pq), ones(ComplexF64,npv+npq), npv+npq, nb);
    Cpv_pq_index=CuSparseMatrixCSC(Cpv_pq_index)
    Cpv_pq_index_transpose =sparse(vcat(pv, pq), 1:(npv + npq), ones(ComplexF64, npv + npq), nb, npv + npq)
    Cpv_pq_index_transpose=CuSparseMatrixCSC(Cpv_pq_index_transpose)

    Cpq_index =sparse(1:npq, pq, ones(ComplexF64,npq), npq, nb);
    Cpq_index=CuSparseMatrixCSC(Cpq_index)
    Cpq_index_transpose = sparse( pq,1:npq, ones(ComplexF64,npq), nb,npq)
    Cpq_index_transpose=CuSparseMatrixCSC(Cpq_index_transpose)

    Cpv_index = PowerFlow.sparse(1:npv, pv, ones(ComplexF64,npv), npv, nb);
    Cpv_index=PowerFlow.CuSparseMatrixCSR(Cpv_index)
    j1 = 1; j2 = npv; # j1:j2 - V angle of pv buses
    j3 = j2 + 1; j4 = j2 + npq; # j3:j4 - V angle of pq buses
    j5 = j4 + 1; j6 = j4 + npq; # j5:j6 - V mag of pq buses
    # Evaluate F(x0)
    mis = V_gpu .* conj.(Ybus_gpu *V_gpu) - Sbus_gpu(Vm_gpu)
    F = [real(mis[vcat(pv, pq)]); imag(mis[pq])]
    #allocate memory
    
     # Check tolerance
    normF = PowerFlow.norm(F, Inf)
    if normF < tol
        converged = true
    end
    # Do Newton iterations
    while (!converged && i < max_it)

        # Update iteration counter
        i += 1

        # Evaluate Jacobian
        dSbus_dVa, dSbus_dVm = PowerFlow.dSbus_dV(Ybus_gpu, V_gpu)
        # _, neg_dSd_dVm = Sbus_gpu(Vm_gpu)
        #dSbus_dVm .-= neg_dSd_dVm

        j11 = real(Cpv_pq_index*dSbus_dVa*Cpv_pq_index_transpose);
        j12 = real(Cpv_pq_index*dSbus_dVm*Cpq_index_transpose);
        j21 = imag(Cpq_index*dSbus_dVa*Cpv_pq_index_transpose);
        j22 = imag(Cpq_index*dSbus_dVm*Cpq_index_transpose);
        J = CuSparseMatrixCSR([j11 j12; j21 j22])
        # @time J,nnz=Parallel_execution_jacobi(dSbus_dVa, dSbus_dVm, pv, pq, pv_gpu, pq_gpu)
        # d_val, d_col, d_row_ptrs=dividecuarray(J,nzz_total_per_row)
        # nzz= sum(nzz_total_per_row)
        # Compute update step
        @time dx, info = mplinsolve(J, -F, alg)
        # Update voltage
        if npv > 0
            Va_gpu[pv] .+= dx[j1:j2]
        end
        if npq > 0
            Va_gpu[pq] .+= dx[j3:j4]
            Vm_gpu[pq] .+= dx[j5:j6]
        end
        V_gpu = Vm_gpu .* exp.(1im * Va_gpu)
        Vm_gpu = abs.(V_gpu) # Update Vm and Va again in case we wrapped around with a negative Vm
        Va_gpu = angle.(V_gpu)

        # Evaluate F(x)
        mis = V_gpu .* conj.(Ybus_gpu * V_gpu) - Sbus_gpu(Vm_gpu)
        F = [real(mis[vcat(pv, pq)]); imag(mis[pq])]

        # Check for convergence
        normF = norm(F, Inf)
        if normF < tol
            converged = true
        end
    end
    V = collect(V_gpu)
    return V, converged, i
end

