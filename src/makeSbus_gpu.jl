function makeSbus_gpu(baseMVA::Float64, bus::Matrix{Float64}, gen::Matrix{Float64}, Vm_gpu::CuArray{Float64, 1, CUDA.DeviceMemory}, Sg=nothing, nargout::Int=1)
    # Define named indices into bus, gen matrices
    (PQ, PV, REF,Pdc,Vdc,NONE, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA,
    VM,VA, BASE_KV, ZONE, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN) = PowerFlow.idx_bus();
    (GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, 
        MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN, PC1, PC2, QC1MIN, QC1MAX, 
        QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF) = PowerFlow.idx_gen();

    nb = size(bus, 1)

    # Get load parameters
    z,i,p = PowerFlow.makeSdzip_gpu(baseMVA, bus)

    if nargout == 2
        Sbus = ComplexF64[]
        if isempty(Vm_gpu)
            dSbus_dVm = spzeros(nb, nb)
        else
            diag_elements_gpu= i + 2 .* Vm_gpu .* z
            n = length(diag_elements_gpu)
            I = CUDA.CuArray(1:n)
            J = CUDA.CuArray(1:n)
            V = diag_elements_gpu
            dSbus_dVm = -sparse(I, J, V, n, n)
        end
        # dSbus_dVm_gpu=PowerFlow.CuSparseMatrixCSC(dSbus_dVm)
        return dSbus_dVm
    else
        # Compute per-bus generation in p.u.
        on = findall(gen[:, GEN_STATUS] .> 0)  # which generators are on?
        gbus = gen[on, GEN_BUS]  # what buses are they at?
        ngon = length(on)
        Cg = PowerFlow.sparse(gbus, 1:ngon, 1, nb, ngon)  # connection matrix
        # element i, j is 1 if gen on(j) at bus i is ON
        if Sg != nothing
            Sbusg = Cg * Sg[on]
        else
            Sbusg = Cg * (gen[on, PG] .+ 1im * gen[on, QG]) / baseMVA
        end

        # Compute per-bus loads in p.u.
        Sbusd = p .+ i .* Vm_gpu .+ z .* Vm_gpu.^2
        # Form net complex bus power injection vector
        # (power injected by generators + power injected by loads)
        Sbus = CuArray(Sbusg) - Sbusd
        return Sbus
    end
end
