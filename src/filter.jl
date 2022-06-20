using CUDA


function filter_kernel(filter_func::Function, src, dst, nres, i_length::Int)
    i = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if i <= i_length
        if filter_func(src[i])
            # atomic_add! function returns old
            dst[CUDA.atomic_add!(CUDA.pointer(nres, 1), 1)] = src[i]
        end
    end
    return nothing
end


function filter_by_threshold(array_length, threshold)
    x = rand(Float32, (array_length, )) |> cu
    y = zeros(Float32, (array_length, )) |> cu
    nres = ones(Int64, (1, )) |> cu
    numblocks = ceil(Int, array_length / 512)
    @cuda threads = 512 blocks = numblocks filter_kernel(x->x>threshold, x, y, nres, array_length)
    y = Array(y)
    return y
end


@time y = filter_by_threshold(100, 0.5)