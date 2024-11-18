#Define the kernel function
#The kernel function is used to slice the CusparsematrixCSC on GPU
#result: the result matrix
#row_indices: the row indices of the sparse matrix in CSC format
#col_ptrs: the column pointers of the sparse matrix in CSC format
#val: the non-zero values of the sparse matrix in CSC format
#a: the sliced part of the row indices
#b: the sliced part of the column indices
#na: the length of a
#nb: the length of b

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