function total_load(bus)
    # define constants
    (PQ, PV, REF,Pdc,Vdc,NONE, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA,
    VM,VA, BASE_KV, ZONE, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN) = idx_bus();
    GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN = idx_gen()

    nb = size(bus, 1) # number of buses

    # default options
    want_Q = 1

    # fixed load at each bus, & initialize dispatchable
        Sd = makeSdzip(1, bus)
        Vm = bus[:, VM]
        Sbusd = Sd.p + Sd.i .* Vm + Sd.z .* Vm.^2
        Pdf = real(Sbusd) # real power
        if want_Q==1
            Qdf = imag(Sbusd) # reactive power
        end

    # dispatchable load at each bus
    
        Pdd = zeros(nb, 1)
        if want_Q==1
            Qdd = zeros(nb, 1)
        end

    # compute load sums
    
        Pd = (Pdf + Pdd) .* (bus[:, BUS_TYPE] .!= NONE-2)
        if want_Q==1
            Qd = (Qdf + Qdd) .* (bus[:, BUS_TYPE] .!= NONE-2)
        end
    return Pd, Qd
    end