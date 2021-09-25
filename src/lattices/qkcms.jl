function qKCMS(omega::Real, gamma::Real, kappa::Real=1.0)
    # Create the spinhalf sites
    sh = spinhalf()

    # Find the light and dark states
    if gamma == kappa
        light = [1, 0]
        dark = [0, 1]
    else
        ss = [4*omega^2+gamma*(kappa+gamma) -2im*omega*(kappa-gamma);
              2im*omega*(kappa-gamma) 4*omega^2+kappa*(kappa+gamma)]
        ss *= (8*omega^2 + 2*(kappa+gamma)^2)^(-1)
        F = eigen(ss)
        eigs = F.values
        vecs = F.vectors
        light = vecs[:, argmin(real(eigs))]
        dark = vecs[:, argmax(real(eigs))]
    end

    add!(sh, "l", light)
    add!(sh, "da", dark)
    add!(sh, "pl", light .*  light', "pl")
    add!(sh, "pda", dark .* dark', "pda")
    return sh
end


function qKCMSDM(omega::Real, gamma::Real, kappa::Real=1.0)
    # Load in the qKCM spin half
    sh = qKCMS(omega, gamma, kappa)

    # Make sitetype
    st = sitetype(4)

    # Add states
    add!(st, "up", kron(state(sh, "up"), state(sh, "up")))
    add!(st, "dn", kron(state(sh, "dn"), state(sh, "dn")))
    add!(st, "l", kron(state(sh, "l"), state(sh, "l")))
    add!(st, "da", kron(state(sh, "da"), state(sh, "da")))
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


function qKCMSDM2(omega::Real, gamma::Real, kappa::Real=1.0)
    # Load in the qKCM spin half
    sh = qKCMS(omega, gamma, kappa)

    add!(sh, "x2", Array([0 (kappa/gamma)^0.25; (gamma/kappa)^0.25 0]), "x2dag")
    add!(sh, "x2dag", Array([0 (gamma/kappa)^0.25; (kappa/gamma)^0.25 0]), "x2")
    P = [gamma^0.25 0; 0 kappa^(0.25)]
    P2 = [gamma^(-0.25) 0; 0 kappa^(-0.25)]
    pl = P2*op(sh, "pl")*P
    add!(sh, "pl2", pl)
    print(pl)

    # Make sitetype
    st = sitetype(4)

    # Add states
    add!(st, "up", kron(state(sh, "up"), state(sh, "up")))
    add!(st, "dn", kron(state(sh, "dn"), state(sh, "dn")))
    add!(st, "l", kron(state(sh, "l"), state(sh, "l")))
    add!(st, "da", kron(state(sh, "da"), state(sh, "da")))
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
    add!(st, "x2id", kron(op(sh, "x2"), op(sh, "id")), "x2dagid")
    add!(st, "x2dagid", kron(op(sh, "x2dag"), op(sh, "id")), "x2id")
    add!(st, "idx2", kron(op(sh, "id"), op(sh, "x2")), "idx2dag")
    add!(st, "idx2dag", kron(op(sh, "id"), op(sh, "x2dag")), "idx2")

    # Add spin projectors
    add!(st, "puid", kron(op(sh, "pu"), op(sh, "id")), "puid")
    add!(st, "idpu", kron(op(sh, "id"), transpose(op(sh, "pu"))), "idpu")
    add!(st, "pdid", kron(op(sh, "pd"), op(sh, "id")), "pdid")
    add!(st, "idpd", kron(op(sh, "id"),  transpose(op(sh, "pd"))), "idpd")

    # Add jump operators
    add!(st, "s+id", kron(op(sh, "s+"), op(sh, "id")), "s-id")
    add!(st, "s-id", kron(op(sh, "s-"), op(sh, "id")), "s+id")
    add!(st, "ids+", kron(op(sh, "id"), op(sh, "s+")), "ids-")
    add!(st, "ids-", kron(op(sh, "id"), op(sh, "s-")), "ids+")
    add!(st, "s-s-", kron(op(sh, "s-"), op(sh, "s-")), "s+s+")
    add!(st, "s+s+", kron(op(sh, "s+"), op(sh, "s+")), "s-s-")

    # Add light and dark operators
    add!(st, "plid", kron(op(sh, "pl2"), op(sh, "id")), "plid")
    add!(st, "idpl", kron(op(sh, "id"), transpose(op(sh, "pl2"))), "idpl")
    add!(st, "plpl", kron(op(sh, "pl2"), transpose(op(sh, "pl2"))), "plpl")

    return st

end

"""
    vectodm(psi::MPS)

Transform a matrix product density state into an MPO. """
function vectodm(psi::MPS)
    dim = Int(sqrt(psi.dim))
    # Create the MPO
    rho = MPO(dim, length(psi))

    # Loop through each tensor, and split the tensor into the physical dims
    for i = 1:length(psi)
        A = psi[i]
        O = moveidx(A, 2, -1)
        O = reshape(O, (size(A)[1], size(A)[3], dim, dim))
        O = moveidx(O, 2, -1)
        rho[i] = O
    end

    return rho
end
