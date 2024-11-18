function dividecuarray(J,nnz_per_row)
    na = size(J, 1)  # Number of rows
    nb = size(J, 2)  # Number of columns
    d_val=CUDA.zeros(Float32,sum(nnz_per_row))  # Values of the sparse matrix
    d_col=CUDA.zeros(Int32, sum(nnz_per_row))  # Row indices of the sparse matrix
    d_row_ptrs=CUDA.zeros(Int32, na + 1)  # Column pointers of the sparse matrix

    d_row_ptrs .= cumsum([0; nnz_per_row])

    threads_per_block = 256
    blocks_per_grid = ceil(Int, na / threads_per_block)
    @cuda threads=threads_per_block blocks=blocks_per_grid kernel_populate_csr!(
    d_val, d_col, d_row_ptrs, J, na, nb
)
    d_row_ptrs.+=1
    return d_val, d_col, d_row_ptrs
end