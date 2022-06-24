using Distributed

num_workers = 2

current_workers = nprocs() - 1
@show current_workers
if current_workers < num_workers
    addprocs(num_workers-current_workers)
elseif current_workers > num_workers
    rmprocs(reverse(collect(num_workers+2:current_workers+1))...)
end
println("workers:", current_workers, "=>", nworkers())

@everywhere using Flux
@everywhere using Flux.Optimise: update!

@everywhere using CUDA
@everywhere using CUDA: device!

@everywhere function test_train(device_id::Int)
    device!(device_id)
    x = rand(Float32, 10) |> cu
    model = Dense(10=>10) |> gpu
    opt = Descent(0.01)
    loss = x->x|>model|>sum

    for i in 1:10000
        grads = gradient(() -> loss(x), Flux.params(model))
        update!(opt, Flux.params(model), grads) # update parameters
    end
    return loss(x)
end



result = []
let
    for idx in 1:10
        process_id = workers()[idx % num_workers + 1]
        r = remotecall(test_train, process_id, 0)
        push!(result, r)
    end
end

result = map(fetch, result)