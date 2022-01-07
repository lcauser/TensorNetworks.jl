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

function spinhalfDM()
    # Load in the qKCM spin half
    sh = spinhalf()

    # Make sitetype
    st = sitetype(4)

    # Add states
    add!(st, "up", kron(state(sh, "up"), state(sh, "up")))
    add!(st, "dn", kron(state(sh, "dn"), state(sh, "dn")))
    add!(st, "s", [1.0, 0.0, 0.0, 1.0])

    # Sigma operators
    add!(st, "xid", kron(op(sh, "x"), op(sh, "id")), "xid")
    add!(st, "idx", kron(op(sh, "id"), op(sh, "x")), "idx")
    add!(st, "yid", kron(op(sh, "y"), op(sh, "id")), "yid")
    add!(st, "idy", kron(op(sh, "id"), transpose(op(sh, "y"))), "idy")
    add!(st, "zid", kron(op(sh, "z"), op(sh, "id")), "zid")
    add!(st, "idz", kron(op(sh, "id"), op(sh, "z")), "idz")
    add!(st, "idid", kron(op(sh, "id"), op(sh, "id")), "idid")
    add!(st, "id", kron(op(sh, "id"), op(sh, "id")), "id")

    # Add spin projectors
    add!(st, "puid", kron(op(sh, "pu"), op(sh, "id")), "puid")
    add!(st, "idpu", kron(op(sh, "id"), op(sh, "pu")), "idpu")
    add!(st, "pdid", kron(op(sh, "pd"), op(sh, "id")), "pdid")
    add!(st, "idpd", kron(op(sh, "id"), op(sh, "pd")), "idpd")
    add!(st, "pupu", kron(op(sh, "pu"), op(sh, "pu")), "pupu")
    add!(st, "pdpd", kron(op(sh, "pd"), op(sh, "pd")), "pdpd")

    # Add jump operators
    add!(st, "s+id", kron(op(sh, "s+"), op(sh, "id")), "s-id")
    add!(st, "s-id", kron(op(sh, "s-"), op(sh, "id")), "s+id")
    add!(st, "ids+", kron(op(sh, "id"), op(sh, "s+")), "ids-")
    add!(st, "ids-", kron(op(sh, "id"), op(sh, "s-")), "ids+")
    add!(st, "s-s-", kron(op(sh, "s-"), op(sh, "s-")), "s+s+")
    add!(st, "s+s+", kron(op(sh, "s+"), op(sh, "s+")), "s-s-")

    return st

end
