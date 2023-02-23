"""
    TFIMHamiltonian(N::Int, h::Number, V::Number)

Create the operator list for the Hamiltonian of the TFIM.
h is the transverse field strength, J is the interaction strength.
"""
function TFIMHamiltonian(N::Int, h::Number, J::Number)
    H = OpList(N) # Creates a list of operators contained in the hamiltonian

    # Transverse field 
    for i = 1:N 
        add!(H, "x", i, h)
    end

    # Potential
    for i = 1:N-1
        add!(H, ["z", "z"], [i, i+1], J)
    end
    return H
end

