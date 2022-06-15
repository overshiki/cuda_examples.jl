using CUDA
using cuda_examples: knn, warp_sum
using Test

"""test knn"""
x = cu(rand(Float32, (60000, 100)))
@time D, I = knn(x, x, 10, Val(:exclude_self))

"""test warp_sum"""
data = rand(3210)
# @show warp_sum(data), sum(data) 
@show isapprox(warp_sum(data), sum(data); rtol=0.0001)

@time warp_sum(data)
@time sum(data)
@time sum(data|>cu)