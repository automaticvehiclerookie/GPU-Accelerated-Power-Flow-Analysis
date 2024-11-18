function makehYbus(baseMVA, bus, branch)
    ## define named indices into bus, branch matrices
    (PQ, PV, REF,Pdc,Vdc,NONE, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA,
    VM,VA, BASE_KV, ZONE, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN) = idx_bus();
    (F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C,TAP, SHIFT,
    BR_STATUS, PF, QF, PT, QT, MU_SF, MU_ST,ANGMIN, ANGMAX, MVSC1,
     MVSC2, BRANCHMODE, ETACR,ETACI, PHI, MU_ANGMIN, MU_ANGMAX) = idx_brch();
    # remove DC branches and VSC branches
        m1=branch[:,F_BUS]
        m2=branch[:,T_BUS]
        n1=bus[Int.(m1),BUS_TYPE].<4
        n2=bus[Int.(m2),BUS_TYPE].<4
        branch=branch[n1.*n2,:]
        on=findall(branch[:,BRANCHMODE].==0)
        branch=branch[on,:]
    #remove DC buses and VSC buses
        sizebus=size(bus,1)
        n3=bus[:,BUS_TYPE].<4
        bus=bus[n3,:]
        
    # constants
    nb = size(bus, 1)          # number of buses
    nl = size(branch, 1)       # number of lines
    # check that bus numbers are equal to indices to bus (one set of bus numbers)

    if any(bus[:, BUS_I] .!= (1:nb))
        error("makeYbus: buses must be numbered consecutively in bus matrix; use ext2int() to convert to internal ordering")
    end
    # for each branch, compute the elements of the branch admittance matrix where
    stat = branch[:, BR_STATUS]                    # ones at in-service branches
    Ys = stat ./ (branch[:, BR_R] .+ 1im * branch[:, BR_X])  # series admittance
    Bc = stat .* branch[:, BR_B]                           # line charging susceptance
    tap = ones(nl,1)                              # default tap ratio = 1
    i = findall(branch[:, TAP] .!= 0)                       # indices of non-zero tap ratios
    tap[i] = branch[i, TAP]                        # assign non-zero tap ratios
    tap = tap .* exp.(1im*pi/180 * branch[:, SHIFT]) # add phase shifters
    Ytt = Ys + 1im*Bc/2
    Yff = Ytt ./ (tap .* conj.(tap))
    Yft = - Ys ./ conj.(tap)
    Ytf = - Ys ./ tap
     # compute shunt admittance
     Ysh = (bus[:, GS] .+ 1im * bus[:, BS]) / baseMVA  # vector of shunt admittances
     # bus indices
    f = branch[:, F_BUS]                           # list of "from" buses
    t = branch[:, T_BUS]                           # list of "to" buses
    ## build connection matrices
    Cf = sparse(1:nl, f, vec(ones(nl, 1)), nl, nb);      ## connection matrix for line & from buses
    Ct = sparse(1:nl, t, vec(ones(nl, 1)), nl, nb);      ## connection matrix for line & to buses
    ## build Yf and Yt such that Yf * V is the vector of complex branch currents injected
    ## at each branch's "from" bus, and Yt is the same for the "to" bus end
    Yf = sparse(1:nl, 1:nl, vec(Yff), nl, nl) * Cf + sparse(1:nl, 1:nl, vec(Yft), nl, nl) * Ct
    Yt = sparse(1:nl, 1:nl, vec(Ytf), nl, nl) * Cf + sparse(1:nl, 1:nl, vec(Ytt), nl, nl) * Ct
    ## build Ybus
    Ybus = Cf' * Yf + Ct' * Yt +             ## branch admittances
        sparse(1:nb, 1:nb, Ysh, nb, nb);    ## shunt admittance
    # end
    # Determine the new size
    new_size = max(size(Ybus, 1), sizebus)

    # Convert new_size to an integer
    new_size = round(Int, new_size)

    # Create a new, larger sparse matrix
    Ybus_new = spzeros(ComplexF64, new_size, new_size)

    # Copy the values from Ybus into the corresponding positions in Ybus_new
    Ybus_new[1:size(Ybus, 1), 1:size(Ybus, 2)] = Ybus

    # Update Ybus
    Ybus = Ybus_new

    return Ybus, Yf, Yt
end