function spinhalf()
    # Create the sitetype
    st = sitetype(2)

    # Add the state
    add!(st, "up", Array([1, 0]))
    add!(st, "dn", Array([0, 1]))
    add!(st, "s", Array([0.5^0.5, 0.5^0.5]))
    add!(st, "as", Array([0.5^0.5, -0.5^0.5]))

    # Add operators
    add!(st, "x", Array([0 1; 1 0]), "x")
    add!(st, "y", Array([0 -1im; 1im 0]), "y")
    add!(st, "z", Array([1 0; 0 -1]), "z")
    add!(st, "id", Array([1 0; 0 1]), "id")
    add!(st, "pu", Array([1 0; 0 0]), "pu")
    add!(st, "pd", Array([0 0; 0 1]), "pd")
    add!(st, "n", Array([1 0; 0 0]), "n")
    add!(st, "s+", Array([0 1; 0 0]), "s-")
    add!(st, "s-", Array([0 0; 1 0]), "s+")

    return st
end
