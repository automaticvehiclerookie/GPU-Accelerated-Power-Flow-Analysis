using CUDA
using CUDA.CUSPARSE
using JLD2

# Load the dense matrix from JLD2
@load "C:/Users/13733/Desktop/DistributionPowerFlow/dSbus_dVm.jld2" dSbus_dVm
@load "C:/Users/13733/Desktop/DistributionPowerFlow/dSbus_dVa.jld2" dSbus_dVa
@load "C:/Users/13733/Desktop/DistributionPowerFlow/pv.jld2" pv
@load "C:/Users/13733/Desktop/DistributionPowerFlow/pq.jld2" pq
dSbus_dVm = CuArray(dSbus_dVm)
dSbus_dVm = CUDA.CUSPARSE.CuSparseMatrixCSC(dSbus_dVm)
dSbus_dVm_real = real(dSbus_dVm)
row_vm_1 = dSbus_dVm_real.rowVal
col_vm_1 = dSbus_dVm_real.colPtr
val_vm_1 = dSbus_dVm_real.nzVal
j12 = CUDA.zeros(Float64, length(vcat(pv, pq)), length(pq))
pv_gpu = CuArray(pv)
pq_gpu = CuArray(pq)

# Array to store the count of non-zero elements for each row
nnz_per_row = CUDA.zeros(Int, length(vcat(pv, pq)))

function kernel_index!(result, row_indices, col_ptrs, val, a, b, na, nb, nnz_per_row)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x  # Ensure i starts from 1
    if i <= na
        local_nnz = 0
        for j in 1:nb
            row = a[i]
            col = b[j]
            value = 0.0
            if col >= 1 && col < length(col_ptrs)  # Check col range
                for k in col_ptrs[col]:(col_ptrs[col + 1] - 1)
                    if row_indices[k] == row
                        value = val[k]
                        break
                    end
                end
            end
            result[i, j] = value
            if value != 0.0
                local_nnz += 1
            end
        end
        nnz_per_row[i] = local_nnz
    end
    return
end

threads_per_block = 256
blocks_per_grid = ceil(Int, length(vcat(pv, pq)) / threads_per_block)
@cuda threads=threads_per_block blocks=blocks_per_grid kernel_index!(j12, row_vm_1, col_vm_1, val_vm_1, vcat(pv_gpu, pq_gpu), pq_gpu, length(vcat(pv, pq)), length(pq), nnz_per_row)


na = size(j12, 1)  # Number of rows
nb = size(j12, 2)  # Number of columns
d_val=CUDA.zeros(Float32,sum(nnz_per_row))  # Values of the sparse matrix
d_col=CUDA.zeros(Int32, sum(nnz_per_row))  # Row indices of the sparse matrix
d_row_ptrs=CUDA.zeros(Int32, na + 1)  # Column pointers of the sparse matrix

d_row_ptrs .= cumsum([0; nnz_per_row])

# Kernel function to populate d_val and d_col based on d_row_ptrs
function kernel_populate_csr!(d_val, d_col, d_row_ptrs, J, na, nb)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= na
        row_start = d_row_ptrs[i]+1
        row_end = d_row_ptrs[i + 1]
        m = row_start  # Set m to the starting index for this row
        for j in 1:nb
            val = J[i, j]
            if val != 0.0 && m <= row_end
                d_val[m] = val
                d_col[m] = j  # Store zero-based index
                m += 1
            end
        end
    end
    return
end

# Launch the second kernel to populate CSR format arrays
@cuda threads=threads_per_block blocks=blocks_per_grid kernel_populate_csr!(
    d_val, d_col, d_row_ptrs, j12, na, nb
)

# Transfer the CSR components back to the host for inspection or further use
d_val_host = collect(d_val)
d_col_host = collect(d_col)
d_row_ptrs_host = collect(d_row_ptrs)