function makeSbus(baseMVA, bus, gen, Vm, Sg=nothing, nargout=1)
    # Define named indices into bus, gen matrices
    (PQ, PV, REF,Pdc,Vdc,NONE, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA,
    VM,VA, BASE_KV, ZONE, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN) = idx_bus();
    (GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, 
        MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN, PC1, PC2, QC1MIN, QC1MAX, 
        QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF) = idx_gen();

    nb = size(bus, 1)

    # Get load parameters
    Sd = makeSdzip(baseMVA, bus)

    if nargout == 2
        Sbus = []
        if isempty(Vm)
            dSbus_dVm = spzeros(nb, nb)
        else
            dSbus_dVm = -(spdiagm(0 => Sd.i + 2 .* Vm .* Sd.z))
        end
        return dSbus_dVm
    else
        # Compute per-bus generation in p.u.
        on = findall(gen[:, GEN_STATUS] .> 0)  # which generators are on?
        gbus = gen[on, GEN_BUS]  # what buses are they at?
        ngon = length(on)
        Cg = sparse(gbus, 1:ngon, 1, nb, ngon)  # connection matrix
        # element i, j is 1 if gen on(j) at bus i is ON
        if Sg != nothing
            Sbusg = Cg * Sg[on]
        else
            Sbusg = Cg * (gen[on, PG] .+ 1im * gen[on, QG]) / baseMVA
        end

        # Compute per-bus loads in p.u.
        Sbusd = Sd.p .+ Sd.i .* Vm .+ Sd.z .* Vm.^2

        # Form net complex bus power injection vector
        # (power injected by generators + power injected by loads)
        Sbus = Sbusg - Sbusd
        return Sbus
    end
end
