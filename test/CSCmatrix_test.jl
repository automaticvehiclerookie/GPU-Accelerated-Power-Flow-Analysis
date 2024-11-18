using CUDA
using CUDA.CUSPARSE
using SparseArrays
using Printf

a = [1, 2, 3]
b = [1, 2, 3]
n = 1888  # size of the matrix
density = 0.005  # density of the non-zero elements
sparse_matrix = sprand(n, n, density)
na = length(a)
nb = length(b)
a_gpu = CUDA.CuArray(a)
b_gpu = CUDA.CuArray(b)
result = CUDA.zeros(Float32, na, nb)

# Convert to CuSparseMatrixCSC
sparse_matrix_gpu = CUDA.CUSPARSE.CuSparseMatrixCSC(sparse_matrix)
#sparse_matrix_gpu=dSbus_dVa
# sparse_matrix_gpu = dSbus_dVa
# The col_indices
col = sparse_matrix_gpu.colPtr
# The row_indices
row = sparse_matrix_gpu.rowVal
# The values
val = sparse_matrix_gpu.nzVal

function kernel_index!(result, row_indices, col_ptrs, val, a, b, na, nb)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x  # 确保 i 从 1 开始
    if i <= na
        for j in 1:nb
            row = a[i] 
            col = b[j]
            value = 0.0
            if col >= 1 && col < length(col_ptrs)  # 检查 col 的范围
                for k in col_ptrs[col]:(col_ptrs[col + 1] - 1)
                    if row_indices[k] == row
                        value = val[k]
                        break
                    end
                end
            end
            result[i, j] = value
        end
    end
    return
end

threads_per_block = 256
blocks_per_grid = ceil(Int, na / threads_per_block)
@cuda threads=threads_per_block blocks=blocks_per_grid kernel_index!(result, row, col, val, a_gpu, b_gpu, na, nb)

