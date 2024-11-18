function runmodel(bus_ac,bus_dc,converter,Gnm,Bnm,Gdc,Cc_ac,Cc_dc,Pg_ac,Qg_ac,Cg_ac,ref,pv,pq,pdc,vdc,ηi,ηr,baseMVA)
    (PQ, PV, REF,Pdc,Vdc,NONE, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA,
    VM,VA, BASE_KV, ZONE, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN) = idx_bus();

    model = JuMP.Model(Ipopt.Optimizer)
    N=size(bus_ac,1)
    n=size(bus_dc,1)
    nc=size(converter,1)
    #define the variables
    @variable(model, θn_ac[1:N])
    @variable(model, Qd_ac[1:N])
    @variable(model, Vn_ac[1:N])
    @variable(model, Vn_dc[1:n])
    @variable(model, Pd_ac[1:N])
    @variable(model, Pd_dc[1:n])
    @variable(model, Pc_ac[1:n] >= 0)
    @variable(model, Qc_ac[1:n] >= 0)
    @variable(model, Pc_dc[1:n] >= 0)
    
    θnm=hcat(θn_ac,θn_ac,θn_ac,θn_ac,θn_ac)
    θnm=θnm.-θnm'
    #define constraints for known values
    for i in ref
        @constraint(model, θn_ac[i] == 0)
        @constraint(model, Vn_ac[i] == bus_ac[i, VM])
    end
    for i in pv
        @constraint(model, Pd_ac[i] == bus_ac[i, PD]/baseMVA)
        @constraint(model, Vn_ac[i] == bus_ac[i, VM])
    end
    for i in pq
        @constraint(model, Pd_ac[i] == bus_ac[i, PD]/baseMVA)
        @constraint(model, Qd_ac[i] == bus_ac[i, QD]/baseMVA)
    end
    for i in pdc
        @constraint(model, Pd_dc[i] == bus_dc[i, 4]/baseMVA)
    end
    for i in vdc
        @constraint(model, Vn_dc[i] == bus_dc[i, 5])
    end
    
    #define converter constraints
    for i in 1:nc
        @NLconstraint(model, Pc_ac[i] * Pc_dc[i] == 0)
        @NLconstraint(model, Qc_ac[i] == Pc_ac[i] * converter[i,21])
        @NLconstraint(model, Vn_dc[i] == converter[i,22] *Vn_ac[Int(bus_dc[i,2])] )
    end
    #define the system constraints
    expr11=@expression(model,sum(Gnm.*Vn_ac.^2,dims=2))
    expr12=@expression(model,expr11.-(Gnm.*cos.(θnm) .+Bnm.*sin.(θnm)).*Vn_ac*Vn_ac)
    #expr13=@expression(model,Cg_ac*Pg_ac-Pd_ac-Cc_ac*Pc_ac+Cc_ac*(Pc_dc.*ηi))
    expr13=@expression(model,Cg_ac*Pg_ac-Cc_ac*Pc_ac)
    expr14=@expression(model,-Pd_ac+Cc_ac*(Pc_dc.*ηi))
    expr15=@expression(model,expr13+expr14)
    for i in 1:length(expr11)
        @NLconstraint(model, expr12[i] == expr15[i])
    end

    expr21=@expression(model,sum(Bnm.*Vn_ac.^2,dims=2))
    expr22=@expression(model,-expr21.-(Gnm.*sin.(θnm) .-Bnm.*cos.(θnm)).*Vn_ac*Vn_ac)
    expr23=@expression(model,Cg_ac*Qg_ac-Qd_ac-Cc_ac*Qc_ac)
    for i in 1:length(expr21)
        @NLconstraint(model, expr22[i] == expr23[i])
    end

    expr31=@expression(model,Gdc*( Vn_dc'.-Vn_dc)*Vn_dc)
    expr32=@expression(model,-Pd_dc-Cc_dc*Pc_dc+Cc_dc*(Pc_ac.*ηr))
    for i in 1:length(expr31)
        @NLconstraint(model, expr31[i] == expr32[i])
    end
    #define the objective function
    expr_loss=@expression(model,sum(Gnm .* (Vn_ac .^ 2)))
     @NLobjective(model, Min,expr_loss)
    #solver configurations
    set_optimizer(model, Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 1000)
    set_optimizer_attribute(model, "tol", 1e-6)
    set_optimizer_attribute(model, "print_level", 5)  # Increase verbosity for diagnostics

    #Add the start value to the model
    for i in pv
        set_start_value(θn_ac[i], bus_ac[i, VA])
        set_start_value(Qd_ac[i], bus_ac[i, QD]/baseMVA)
    end
    for i in pq
        set_start_value(θn_ac[i], bus_ac[i, VA])
        set_start_value(Vn_ac[i], bus_ac[i, VM])
    end
    for i in pdc
        set_start_value(Vn_dc[i], bus_dc[i, 5])
    end
    for i in vdc
        set_start_value(Pd_dc[i], bus_dc[i, 4]/baseMVA)
    end
    #solve the model
    JuMP.optimize!(model)
    #return the results
    bus_ac[:,VA]=value.(θn_ac)
    bus_ac[:,VM]=value.(Vn_ac)
    bus_dc[:,5]=value.(Vn_dc)
    bus_ac[:,PD]=value.(Pd_ac)*baseMVA
    bus_ac[:,QD]=value.(Qd_ac)*baseMVA
    bus_dc[:,4]=value.(Pd_dc)*baseMVA
    return bus_ac,bus_dc
end