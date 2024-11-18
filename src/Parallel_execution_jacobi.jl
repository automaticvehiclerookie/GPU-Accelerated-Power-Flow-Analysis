function Parallel_execution_jacobi(dSbus_dVa, dSbus_dVm, pv, pq, pv_gpu, pq_gpu)
    # stream1 = CUDA.CuStream()
    # stream2 = CUDA.CuStream()
    # stream3 = CUDA.CuStream()
    # stream4 = CUDA.CuStream()

    dSbus_dVa_real = real(dSbus_dVa)
    dSbus_dVm_real = real(dSbus_dVm)
    dSbus_dVa_imag = imag(dSbus_dVa)
    dSbus_dVm_imag = imag(dSbus_dVm)

    threads_per_block = 512
    blocks_per_grid = ceil(Int, length(vcat(pv, pq)) / threads_per_block)

    row_va_1 = dSbus_dVa_real.rowVal
    col_va_1 = dSbus_dVa_real.colPtr
    val_va_1 = dSbus_dVa_real.nzVal
    j11 = CUDA.zeros(Float64, length(vcat(pv, pq)), length(vcat(pv, pq)))
    nnz_per_row1 = CUDA.zeros(Int, length(vcat(pv, pq)))
    @cuda threads=threads_per_block blocks=blocks_per_grid  kernel_index!(j11, row_va_1, col_va_1, val_va_1, vcat(pv_gpu, pq_gpu), vcat(pv_gpu, pq_gpu), length(vcat(pv, pq)), length(vcat(pv, pq)), nnz_per_row1)

    row_vm_1 = dSbus_dVm_real.rowVal
    col_vm_1 = dSbus_dVm_real.colPtr
    val_vm_1 = dSbus_dVm_real.nzVal
    j12 = CUDA.zeros(Float64, length(vcat(pv, pq)), length(pq))
    nnz_per_row2 = CUDA.zeros(Int, length(vcat(pv, pq)))
    @cuda threads=threads_per_block blocks=blocks_per_grid  kernel_index!(j12, row_vm_1, col_vm_1, val_vm_1, vcat(pv_gpu, pq_gpu), pq_gpu, length(vcat(pv, pq)), length(pq), nnz_per_row2)

    row_va_2 = dSbus_dVa_imag.rowVal
    col_va_2 = dSbus_dVa_imag.colPtr
    val_va_2 = dSbus_dVa_imag.nzVal
    j21 = CUDA.zeros(Float64, length(pq), length(vcat(pv, pq)))
    nnz_per_row3 = CUDA.zeros(Int, length(pq))
    @cuda threads=threads_per_block blocks=blocks_per_grid  kernel_index!(j21, row_va_2, col_va_2, val_va_2, pq_gpu, vcat(pv_gpu, pq_gpu), length(pq), length(vcat(pv, pq)), nnz_per_row3)

    row_vm_2 = dSbus_dVm_imag.rowVal
    col_vm_2 = dSbus_dVm_imag.colPtr
    val_vm_2 = dSbus_dVm_imag.nzVal
    j22 = CUDA.zeros(Float64, length(pq), length(pq))
    nnz_per_row4 = CUDA.zeros(Int, length(pq))
    @cuda threads=threads_per_block blocks=blocks_per_grid  kernel_index!(j22, row_vm_2, col_vm_2, val_vm_2, pq_gpu, pq_gpu, length(pq), length(pq), nnz_per_row4 )

    # Synchronize the streams to ensure all kernels have completed
    # synchronize(stream1)
    # synchronize(stream2)
    # synchronize(stream3)
    # synchronize(stream4)

    # 将调试数组从 GPU 复制到 CPU
    # debug_i1_host = Array(debug_i1)
    # debug_i2_host = Array(debug_i2)
    # debug_i3_host = Array(debug_i3)
    # debug_i4_host = Array(debug_i4)
    nzz_total_per_row = [nnz_per_row1.+nnz_per_row2;nnz_per_row3.+nnz_per_row4]
    J= Array([j11 j12; j21 j22])
    J=sparse(J)
    J= CuSparseMatrixCSC(J)
    GC.gc()
    return J, nzz_total_per_row
end