include("src/TensorNetworks.jl")

D = 2
d = 2
N = 2
c = 0.5
s = -1.0

A = randn(Float64, D, d, d, D)
edgeleft = zeros(Float64, D)
edgeleft[D] = 1
edgeright = zeros(Float64, D)
edgeright[1] = 1

configs = [[1, 1], [1, 2], [2, 1], [2, 2]]

for i = 1:10000
    totalgrad = zeros(Float64, D, d, d, D)
    for config1 = configs
        for config2 = configs
            grad = 1
            H = 0
            if config1[1] == config2[1]
                if config1[2] == 1 && config2[2] == 1
                    H = 1 - c
                elseif config1[2] == 2 && config2[2] == 2
                    H = c
                else
                    H = -sqrt(c*(1-c))*exp(-s)
                end
            end

            contraction = contract(edgeleft, A[:, config1[1], config2[1], :], 1, 1)
            contraction = contract(contraction, A[:, config1[2], config2[2], :], 1, 1)
            contraction = contract(contraction, edgeright, 1, 1)
            grad *= contraction[1] - H

            t1 = zeros(Float64, 2)
            t1[config1[1]] = 1
            t2 = zeros(Float64, 2)
            t2[config2[1]] = 1
            t3 = zeros(Float64, 2)
            t3[config1[2]] = 1
            t4 = zeros(Float64, 2)
            t4[config2[2]] = 1

            partial1 = tensorproduct(edgeleft, [1], t1, [2])
            partial1 = tensorproduct(partial1, [1, 2], t2, [3])
            partial1 = tensorproduct(partial1, [1, 2, 3], A[:, config1[2], config2[2], :], [4, 5])
            partial1 = contract(partial1, edgeright, 5, 1)

            partial2 = contract(edgeleft, A[:, config1[1], config2[1], :], 1, 1)
            partial2 = tensorproduct(partial2, [1], t3, [2])
            partial2 = tensorproduct(partial2, [1, 2], t4, [3])
            partial2 = tensorproduct(partial2, [1, 2, 3], edgeright, [4])

            grad *= (partial1 + partial2)
            totalgrad += grad

        end
    end
    A -= 0.01 * totalgrad
    println(sum(totalgrad))
end
