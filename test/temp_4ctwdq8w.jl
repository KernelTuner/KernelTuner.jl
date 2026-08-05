
using KernelAbstractions

@kernel function vector_add!(C, A, B, n, block_size_x,
                     ::Val{WORK_PER_THREAD}) where {WORK_PER_THREAD}
    tid = @index(Global)
    first_i = (tid - 1) * WORK_PER_THREAD + 1

    for offset in 0:(WORK_PER_THREAD - 1)
        i = first_i + offset
        if i <= n
            @inbounds C[i] = A[i] + B[i]
        end
    end
end
