"""Supporting file containing the kernels ran in the integration test."""

using KernelAbstractions.Extras: @unroll
using KernelAbstractions
using StaticArrays

@kernel cpu = false inbounds = true unsafe_indices = false function applyQorQt_unsafe_kernel_2d!(A, @Const(Min), @Const(tau),
    ::Val{TILESIZE}=Val(64), ::Val{TILEHEIGHT}=Val(64), ::Val{TILESIZEMUL}=Val(32)) where {TILESIZE,TILEHEIGHT,TILESIZEMUL}
    ## Constraints: TILEHEIGHT >= TILESIZE, i <= TILESIZEMUL
    g = @index(Group, Linear)
    i = @index(Local, Linear)
    tilecol = @private eltype(A) (TILEHEIGHT)
    M = @localmem eltype(A) (TILEHEIGHT)
    n_A_r = size(A, 1)
    n_A_c = size(A, 2)
    n_Min_r = size(Min, 1)
    n_Min_c = size(Min, 2)
    n_tau_r = size(tau, 1)

    col = (g - 1) * TILESIZEMUL + i
    @unroll for l in 1:TILEHEIGHT
        tilecol[l] = (l >= 1 && l <= n_A_r && col >= 1 && col <= n_A_c) ? A[l, col] : zero(eltype(A))
    end

    for k in 1:(TILESIZE-Int(TILESIZE==TILEHEIGHT))
        JM_MAX = (TILEHEIGHT + TILESIZEMUL - 1) ÷ TILESIZEMUL
        @unroll for jm in 0:(JM_MAX-1)
            if (jm * TILESIZEMUL + i <= TILEHEIGHT)
                mrow = jm * TILESIZEMUL + i
                M[mrow] = (mrow >= 1 && mrow <= n_Min_r && k >= 1 && k <= n_Min_c) ? Min[mrow, k] : zero(eltype(A))
            end
        end
        @synchronize
        if ((g - 1) * TILESIZEMUL + i <= size(A, 2))
            tmp_sum = zero(eltype(A))
            @unroll for l in (k+1):TILEHEIGHT
                tmp_sum += M[l] * tilecol[l]
            end
            tmp_sum += tilecol[k]
            tmp_sum *= (k >= 1 && k <= n_tau_r) ? tau[k] : zero(eltype(A))

            @unroll for l in (k+1):TILEHEIGHT
                tilecol[l] -= tmp_sum * M[l]
            end
            tilecol[k] -= tmp_sum
        end
        @synchronize
    end

    @unroll for l in 1:TILEHEIGHT
        if (l >= 1 && l <= n_A_r && col >= 1 && col <= n_A_c)
            A[l, col] = tilecol[l]
        end
    end
end




@kernel cpu=false inbounds=true unsafe_indices=false function applyQorQt_unsafe_kernel2_2d_fused_A!(A, B, @Const(Min), @Const(tau),
    ::Val{TILESIZE}=Val(64), ::Val{TILEHEIGHT}=Val(64), ::Val{TILESIZEMUL}=Val(32), ::Val{QRSPLIT}=Val(8), ::Val{QRFACTOR}=Val(8),
    ::Val{THREADFACTOR}=Val(1), ::Val{THREADFACTOR2}=Val(1), ::Val{THREADFACTOR3}=Val(1)) where {TILESIZE,TILEHEIGHT,TILESIZEMUL,QRSPLIT,QRFACTOR,THREADFACTOR,THREADFACTOR2,THREADFACTOR3}

    if (QRSPLIT == 1)
        g = @index(Group, Linear)
        i = @index(Local, Linear)
        j=1
    else
        g, _ = @index(Group, NTuple)
        i, j = @index(Local, NTuple)
    end
    tilecol = @private eltype(A) (QRFACTOR)
    tilecolA = @private eltype(A) (THREADFACTOR3)
    Mcurr = @localmem eltype(A) (TILEHEIGHT)
    tausmem = @localmem eltype(A) (TILESIZE)
    partial_sums = @localmem eltype(A) (TILESIZEMUL, QRSPLIT)
    tidx = i+(j-1)*TILESIZEMUL
    currstartrow=0 + (TILEHEIGHT - TILESIZE)
    nbtiles = cld(size(Min, 1) - (TILEHEIGHT - TILESIZE), TILEHEIGHT)
    n_A_r = size(A, 1)
    n_A_c = size(A, 2)
    n_B_r = size(B, 1)
    n_B_c = size(B, 2)
    n_Min_r = size(Min, 1)
    n_Min_c = size(Min, 2)
    n_tau_r = size(tau, 1)
    n_tau_c = size(tau, 2)

    if j <= TILESIZE
        @unroll for l in 1:THREADFACTOR3
            row = (j - 1) * THREADFACTOR3 + l
            col = i + (g - 1) * TILESIZEMUL
            tilecolA[l] = (row >= 1 && row <= n_A_r && col >= 1 && col <= n_A_c) ? A[row, col] : zero(eltype(A))
        end
    end
    for currtile in 1:nbtiles
        @unroll for l in 1:QRFACTOR
            row = currstartrow + (j - 1) * QRFACTOR + l
            col = i + (g - 1) * TILESIZEMUL
            tilecol[l] = (row >= 1 && row <= n_B_r && col >= 1 && col <= n_B_c) ? B[row, col] : zero(eltype(A))
        end
        @unroll for jtau in 0:(THREADFACTOR2-1)
            if (jtau*TILESIZEMUL*QRSPLIT+tidx <= TILESIZE)
                tau_row = jtau * TILESIZEMUL * QRSPLIT + tidx
                tau_col = currtile
                tausmem[tau_row] = (tau_row >= 1 && tau_row <= n_tau_r && tau_col >= 1 && tau_col <= n_tau_c) ? tau[tau_row, tau_col] : zero(eltype(A))
            end
        end
        for k in 1:TILESIZE
            tmp_sum = zero(eltype(A))
            @unroll for jm in 0:(THREADFACTOR-1)
                if (jm*TILESIZEMUL*QRSPLIT+tidx <= TILEHEIGHT)
                    mrow = jm * TILESIZEMUL * QRSPLIT + tidx
                    src_row = currstartrow + mrow
                    src_col = k
                    Mcurr[mrow] = (src_row >= 1 && src_row <= n_Min_r && src_col >= 1 && src_col <= n_Min_c) ? Min[src_row, src_col] : zero(eltype(A))
                end
            end
            @synchronize
            @unroll for l in 1:QRFACTOR
                tmp_sum += Mcurr[(j-1)*QRFACTOR+l] * tilecol[l]
            end
            partial_sums[i, j] = tmp_sum
            @synchronize
            k_local = k - (j-1)*THREADFACTOR3
            if j <= TILESIZE && k_local >= 1 && k_local <= THREADFACTOR3
                tmp_sum = zero(eltype(A))
                @unroll for jj in 1:QRSPLIT
                    tmp_sum += partial_sums[i, jj]
                end
                tmp_sum += tilecolA[k_local]
                tmp_sum *= tausmem[k]
                partial_sums[i, 1] = tmp_sum
                tilecolA[k_local] -= tmp_sum
            end
            @synchronize
            tmp_sum = partial_sums[i, 1]
            @unroll for l in 1:QRFACTOR
                tilecol[l] -= tmp_sum * Mcurr[(j-1)*QRFACTOR+l]
            end
            @synchronize
        end
        @unroll for l in 1:QRFACTOR
            row = currstartrow + (j - 1) * QRFACTOR + l
            col = i + (g - 1) * TILESIZEMUL
            if (row >= 1 && row <= n_B_r && col >= 1 && col <= n_B_c)
                B[row, col] = tilecol[l]
            end
        end
        currstartrow += TILEHEIGHT
    end

    if j <= TILESIZE
        @unroll for l in 1:THREADFACTOR3
            row = (j - 1) * THREADFACTOR3 + l
            col = i + (g - 1) * TILESIZEMUL
            if (row >= 1 && row <= n_A_r && col >= 1 && col <= n_A_c)
                A[row, col] = tilecolA[l]
            end
        end
    end

end
