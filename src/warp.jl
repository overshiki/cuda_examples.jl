"""
kernels using warp-level functions
following the blog: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
"""

using CUDA

function sum_kernel(data, i_length, warp_length, sum_result)
    @assert floor(log2(warp_length))==log2(warp_length) #i_length should be 2^N 
    i = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    ii = (threadIdx()).x
    if i <= i_length
        mask = CUDA.vote_ballot_sync(active_mask(), i<=i_length)
        for offset in reverse(0:log2(warp_length)-1)
            offset = 2^offset
            data[i] += CUDA.shfl_down_sync(mask, data[i], offset)
        end

        if ii==1
            CUDA.atomic_add!(CUDA.pointer(sum_result, 1), Float32(data[i]))
        end

    end
    return nothing
end


function (++)(v1::T, v2::T) where {T}
    append!(v1, v2)
    return v1
end

function warp_sum(data)
    @assert length(size(data))==1
    n_blocks = Int(floor(length(data) / 32))
    if n_blocks !== length(data)/32 
        n_blocks += 1
    end

    if length(data) < n_blocks * 32
        cudata = Vector{Float32}(deepcopy(data)) ++ zeros(Float32, (n_blocks*32 - length(data)))
    else 
        cudata = Vector{Float32}(deepcopy(data))
    end

    cudata = cudata |> cu
    sum_result = zeros(Float32, 1) |> cu

    @cuda threads=32 blocks=n_blocks sum_kernel(cudata, length(cudata), 32, sum_result)
    return Array(sum_result)[1]
end

