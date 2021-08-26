function qKCMS(omega::Real, gamma::Real, kappa::Real=1.0)
    # Create the spinhalf sites
    sh = spinhalf()

    # Find the light and dark states
    if gamma == kappa
        light = [1, 0]
        dark = [0, 1]
    else
        ss = [[4*omega^2+gamma*(kappa+gamma), -2im*omega*(kappa-gamma)]
              [2im*omega*(kappa-gamma), 4*omega^2+kappa*(kappa+gamma)]]
        ss *= (8*omega^2 + 2*(kappa+gamma)^2)^(-1)
        F = eigen(ss)
        eigs = F.values
        vecs = F.vectors
        light = vecs[:, argmin(real(eigs))]
        dark = vecs[:, argmax(real(eigs))]
    end

    add!(sh, "l", light)
    add!(sh, "da", dark)
    add!(sh, "pl", light' .* conj(light), "pl")
    add!(sh, "pda", dark' .* conj(dark), "pda")
    return sh
end


function qKCMSDM(omega::Real, gamma::Real, kappa::Real=1.0)
    # Load in the qKCM spin half
    sh = qKCMS(omega, gamma, kappa)

    # Make sitetype
    st = sitetypes(4)

    # Add states
    add!(st, "up", kron(state(sh, "up"), state(sh, "up")))
    add!(st, "dn", kron(state(sh, "dn"), state(sh, "dn")))
    add!(st, "l", kron(state(sh, "l"), state(sh, "l")))
    add!(st, "da", kron(state(sh, "da"), state(sh, "da")))

    # Sigma operators
    add!(st, "xid", kron(op(sh, "x"), op(sh, "id")), "xid")
    add!(st, "idx", kron(op(sh, "id"), op(sh, "x")), "idx")
    add!(st, "yid", kron(op(sh, "y"), op(sh, "id")), "yid")
    add!(st, "idy", kron(op(sh, "id"), transpose(op(sh, "y"))), "idy")
    add!(st, "zid", kron(op(sh, "z"), op(sh, "id")), "zid")
    add!(st, "idz", kron(op(sh, "id"), op(sh, "z")), "idz")
    add!(st, "idid", kron(op(sh, "id"), op(sh, "id")), "idid")

    # Add spin projectors
    add!(st, "puid", kron(op(sh, "pu"), op(sh, "id")), "puid")
    add!(st, "idpu", kron(op(sh, "id"), op(sh, "pu")), "idpu")
    add!(st, "pdid", kron(op(sh, "pd"), op(sh, "id")), "pdid")
    add!(st, "idpd", kron(op(sh, "id"), op(sh, "pd")), "idpd")

    # Add jump operators
    add!(st, "s+id", kron(op(sh, "s+"), op(sh, "id")), "s-id")
    add!(st, "s-id", kron(op(sh, "s-"), op(sh, "id")), "s+id")
    add!(st, "ids+", kron(op(sh, "id"), op(sh, "s+")), "ids-")
    add!(st, "ids-", kron(op(sh, "id"), op(sh, "s-")), "ids+")
    add!(st, "s-s-", kron(op(sh, "s-"), op(sh, "s-")), "s+s+")
    add!(st, "s+s+", kron(op(sh, "s+"), op(sh, "s+")), "s-s-")

    # Add light and dark operators
    add!(st, "plid", kron(op(sh, "pl"), op(sh, "id")), "plid")
    add!(st, "idpl", kron(op(sh, "id"), transpose(op(sh, "pl"))), "idpl")
    add!(st, "plpl", kron(op(sh, "pl"), transpose(op(sh, "pl"))), "plpl")

    return st

end
