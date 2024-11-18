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