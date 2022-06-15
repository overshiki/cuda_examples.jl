using CUDA


function argknn_kernel(knn_out, indices_out, x, y, i, j_length, d_length, knn_length)
    for j in 1:j_length
        dij = 0
        for d in 1:d_length
            dij += (x[i,d] - y[j,d])^2 #/ (d_length * d_length * 10)
        end
        dij = sqrt(dij)#/d_length
        # dij = dij * d_length

        for k in 1:knn_length
            if knn_out[i,k]==-1
                knn_out[i,k] = dij
                indices_out[i,k] = j
                break
            elseif dij < knn_out[i,k]
                for o in reverse(k:knn_length-1)
                    knn_out[i, o+1] = knn_out[i, o]
                    indices_out[i, o+1] = indices_out[i, o]
                end 
                knn_out[i,k] = dij 
                indices_out[i,k] = j
                break
            end
        end 
    end
end



function knn_kernel(knn_out, x, y, i, j_length, d_length, knn_length)
    for j in 1:j_length
        dij = 0
        for d in 1:d_length
            dij += (x[i,d] - y[j,d])^2
        end
        dij = sqrt(dij)/d_length

        for k in 1:knn_length
            if knn_out[i,k]==-1
                knn_out[i,k] = dij
                break
            elseif dij < knn_out[i,k]
                for o in reverse(k:knn_length-1)
                    knn_out[i, o+1] = knn_out[i, o]
                end 
                knn_out[i,k] = dij 
                break
            end
        end 
    end
end

function device_fun(knn_out, indices_out, x, y, i_length, j_length, d_length, knn_length)
    i = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if i <= i_length
        argknn_kernel(knn_out, indices_out, x, y, i, j_length, d_length, knn_length)
    end
end




function knn(x, y, k)
    i_length = size(x)[1]
    j_length = size(y)[1]
    d_length = size(x)[2]
    @assert d_length==size(y)[2]

    knn_out = cu(ones(Float32, (i_length, k)) * (-1))
    indices_out = cu(ones(Int64, (i_length, k)))
    numblocks = ceil(Int, i_length / 512)
    @cuda threads = 512 blocks = numblocks device_fun(knn_out, indices_out, x, y, i_length, j_length, d_length, k)
    # include self
    return knn_out, indices_out
end

function knn(x, y, k, ::Val{:exclude_self})
    D, I = knn(x, y, k+1)
    return D[:, 2:end], I[:, 2:end]
end
