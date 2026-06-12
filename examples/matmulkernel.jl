# matmul kernel
# source https://github.com/JuliaGPU/KernelAbstractions.jl/tree/main/examples

using KernelAbstractions,  StaticArrays

@kernel unsafe_indices = true function coalesced_matmul_kernel_v2!(
        output, @Const(input1), @Const(input2), N, R, M,  ::Val{TILE} = Val(16)
    ) where {TILE}
    gi, gj = @index(Group, NTuple)
    i, j = @index(Local, NTuple)

    # +1 to avoid bank conflicts on shared memory
    tile1 = @localmem eltype(output) (TILE + 1, TILE)
    tile2 = @localmem eltype(output) (TILE + 1, TILE)

    # private variable for tile output
    outval = @private eltype(output) 1
    @inbounds outval[1] = -zero(eltype(output))

    @uniform N = size(output, 1)
    # number of tiles depends on inner dimension
    @uniform NUM_TILES = div(R + TILE - 1, TILE)

    # loop over all tiles needed for this calculation
    for t in 0:(NUM_TILES - 1)
        # Can't use @index(Global), because we use a smaller ndrange
        I = (gi - 1) * TILE + i
        J = (gj - 1) * TILE + j

        # load inputs into tiles, with bounds checking for non-square matrices
        if I <= N && t * TILE + j <= R
            @inbounds tile1[i, j] = input1[I, t * TILE + j]
        else
            @inbounds tile1[i, j] = 0.0
        end
        if t * TILE + i <= R && J <= M
            @inbounds tile2[i, j] = input2[t * TILE + i, J]
        else
            @inbounds tile2[i, j] = 0.0
        end

        # wait for all tiles to be loaded
        @synchronize

        # get global values again
        I = (gi - 1) * TILE + i
        J = (gj - 1) * TILE + j

        # calculate value of spot in output, use temporary value to allow for vectorization
        out = zero(eltype(output))
        @simd for k in 1:TILE
            @inbounds out += tile1[i, k] * tile2[k, j]
        end
        outval[1] += out

        @synchronize
    end

    # get global indices again
    I = (gi - 1) * TILE + i
    J = (gj - 1) * TILE + j

    # save if inbounds
    if I <= N && J <= M
        @inbounds output[I, J] = outval[1]
    end
end


