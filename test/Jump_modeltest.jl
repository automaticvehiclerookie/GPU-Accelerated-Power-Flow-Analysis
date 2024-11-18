using JuMP
using SparseArrays
using Ipopt
using CUDA
using MadNLP
using ExaModels

#configurations of CUDA
CUDA.allowscalar(false);
CUDA.device!(0)

#using Gurobi
const baseMVA = 100.0
# the function is used to calculate the injected power at the generator buses
    # create a model
    # model = JuMP.Model(Ipopt.Optimizer)
    model = JuMP.Model(Ipopt.Optimizer)
set_optimizer_attribute(model, "autodiff", true)
    # register(model, :.*, N, .*, autodiff=true)
    # set_attribute(model, "TimeLimit", 100)
    # set_attribute(model, "Presolve", 0)
    # add variables to the model
    @variable(model, θn_ac[1:5])
    @variable(model, Qd_ac[1:5])
    @variable(model, Vn_ac[1:5])
    @variable(model, Vn_dc[1:2])
    @variable(model, Pd_ac[1:5])
    @variable(model, Pd_dc[1:2])
    @variable(model, Pc_ac[1:2] >= 0)
    @variable(model, Qc_ac[1:2] >= 0)
    @variable(model, Pc_dc[1:2] >= 0)
    # add parameters to the model
    Gnm = [
        6.25 -5 -1.25 0.0 0.0 ;
        -5 10.8333 -1.66667 -1.66667 -2.5 ;
        -1.25 -1.66667 12.9167 -10 0.0 ;
        0.0 -1.66667 -10 12.9167 -1.25 ;
        0.0 -2.5 0.0 -1.25 3.75 
    ]
    Bnm = [
        -18.695 15.0 3.75 0.0 0.0;
        15.0 -32.415 5 5 7.5;
        3.75 5 -38.695 30.0 0.0 ;
        0.0 5 30.0 -38.695 3.75 ;
        0.0 7.5 0.0 3.75 -11.21 
    ]
    Gdc = [
         0.0 19.2308;
         19.2308 0.0
    ]
    Cc_ac=[
        0 0;
        1 0;
        0 1;
        0 0;
        0 0    
        ]
    Cc_dc=[
        1 0;
        0 1
    ]
    ηi=[
        0.975;
        0.97
    ]
    ηr=[
        0.975;
        0.97
    ]
    Ybus=Gnm+im*Bnm
    Pg_ac=[
        0;
        40/baseMVA;
        0;
        0;
        0
    ]
    Qg_ac=[
        0;
        0;
        0;
        0;
        0
    ]
    Cg_ac=[
        1 0 0 0 0;
        0 1 0 0 0;
        0 0 0 0 0;
        0 0 0 0 0;
        0 0 0 0 0
    ]
    θnm=hcat(θn_ac,θn_ac,θn_ac,θn_ac,θn_ac)
    θnm=θnm.-θnm'
    # add constraints to the model
    @constraint(model, θn_ac[1] == 0)
    @constraint(model, Vn_ac[1] == 1.06)
    @constraint(model, Pd_ac[2] == 20/baseMVA)
    @constraint(model, Qd_ac[2] == 10.0/baseMVA)
    @constraint(model, Pd_ac[3] == 45/baseMVA)
    @constraint(model, Vn_ac[3] == 1.0)
    @constraint(model, Pd_ac[4] == 40/baseMVA)
    @constraint(model, Qd_ac[4] == 5/baseMVA)
    @constraint(model, Pd_ac[5] == 60/baseMVA)
    @constraint(model, Qd_ac[5] == 10/baseMVA)
    @constraint(model, Pd_dc[1] == 0)
    @constraint(model, Vn_dc[2] == 1)
    #define converter constraints
    @NLconstraint(model, Pc_ac[1] * Pc_dc[1] == 0)
    @NLconstraint(model, Pc_ac[2] * Pc_dc[2] == 0)
    @NLconstraint(model, Qc_ac[1] == Pc_ac[1] * 40/60)
    @NLconstraint(model, Qc_ac[2] == Pc_ac[2] * 0)
    @NLconstraint(model, Vn_dc[1] == 1.015 * Vn_ac[2])
    @NLconstraint(model, Vn_dc[2] == 1.0 * Vn_ac[3])
    #define the system constraints
    #For the AC part
    # expr11=@expression(model,real(Vn_ac .* conj.(Ybus * Vn_ac)))
    # expr12=@expression(model,Cg_ac*Pg_ac-Pd_ac-Cc_ac*Pc_ac+Cc_ac*(Pc_dc.*ηi))
    # for i in 1:length(expr11)
    #     @NLconstraint(model, expr11[i] == expr12[i])
    # end
    expr11=@expression(model,sum(Gnm.*Vn_ac.^2,dims=2))
    expr12=@expression(model,expr11.-(Gnm.*cos.(θnm) .+Bnm.*sin.(θnm)).*Vn_ac*Vn_ac)
    expr13=@expression(model,Cg_ac*Pg_ac-Pd_ac-Cc_ac*Pc_ac+Cc_ac*(Pc_dc.*ηi))
    for i in 1:length(expr11)
        @NLconstraint(model, expr12[i] == expr13[i])
    end


    # expr21=@expression(model,imag(Vn_ac .* conj.(Ybus * Vn_ac)))
    # expr22=@expression(model,Cg_ac*Qg_ac-Qd_ac-Cc_ac*Qc_ac)
    # for i in 1:length(expr21)
    #     @NLconstraint(model, expr21[i] == expr22[i])
    # end
    expr21=@expression(model,sum(Bnm.*Vn_ac.^2,dims=2))
    expr22=@expression(model,-expr21.-(Gnm.*sin.(θnm) .-Bnm.*cos.(θnm)).*Vn_ac*Vn_ac)
    expr23=@expression(model,Cg_ac*Qg_ac-Qd_ac-Cc_ac*Qc_ac)
    for i in 1:length(expr21)
        @NLconstraint(model, expr22[i] == expr23[i])
    end

    # # @NLconstraint(model,Qg_ac-Qd_ac-Cc_ac*Qc_ac==imag(Vn_ac .* conj.(Ybus * Vn_ac) ))
    # #For the DC part
    # expr31=@expression(model,real(Vn_dc .* conj.(Gdc * Vn_dc)))
    # expr32=@expression(model,Pd_dc-Cc_dc*Pc_dc+Cc_dc*(Pc_ac.*ηr))
    # for i in 1:length(expr31)
    #     @NLconstraint(model, expr31[i] == expr32[i])
    # end
    expr31=@expression(model,Gdc*( Vn_dc'.-Vn_dc)*Vn_dc)
    expr32=@expression(model,-Pd_dc-Cc_dc*Pc_dc+Cc_dc*(Pc_ac.*ηr))
    for i in 1:length(expr31)
        @NLconstraint(model, expr31[i] == expr32[i])
    end
    # @NLconstraint(model,Pd_dc-Cc_dc*Pc_dc+Cc_dc*(Pc_ac.*ηr)==real(Vn_dc .* conj.(Gdc * Vn_dc) ))
    # define the objective function
     @NLobjective(model, Min, -Pd_ac[2] - Pc_ac[1] + 0.975 *Pc_dc[1] + 0.4)


# 设置求解器
set_optimizer(model, Ipopt.Optimizer)
set_optimizer_attribute(model, "max_iter", 1000)
set_optimizer_attribute(model, "tol", 1e-6)
set_optimizer_attribute(model, "print_level", 5)  # Increase verbosity for diagnostics

#set the initial value of the variables
    #set start value to variables
    set_start_value(θn_ac[2], 0.0)
    set_start_value(θn_ac[3], 0.0)
    set_start_value(θn_ac[4], 0.0)
    set_start_value(θn_ac[5], 0.0)
    set_start_value(Vn_ac[2], 1.0)
    set_start_value(Qd_ac[3], 15.0/baseMVA)
    set_start_value(Vn_ac[4], 1.0)
    set_start_value(Vn_ac[5], 1.0)
    set_start_value(Vn_dc[1], 1.0)
    set_start_value(Pd_dc[2], 0.0)
    set_start_value(Pc_ac[1], 60.0/baseMVA)
    set_start_value(Pc_ac[2], 0.0)
    set_start_value(Qc_ac[1], 40.0/baseMVA)
    set_start_value(Qc_ac[2], 0.0)
    set_start_value(Pc_dc[1], 0.0)
    set_start_value(Pc_dc[2], 0.0)

JuMP.optimize!(model)

# Check if the model has a solution
# if JuMP.termination_status(model) == MOI.OPTIMAL
#     println("Objective Value: ", JuMP.objective_value(model))
#     # Assuming θnm_un, Qn_un, etc., are defined in your model
#     println("θnm_un: ", JuMP.value.(θnm_un))
#     println("Qn_un: ", JuMP.value.(Qn_un))
#     # Add similar lines for other variables
# else
#     println("Model did not solve to optimality.")
# end
println("θn_ac: ", JuMP.value.(θn_ac))
println("Qd_ac: ", JuMP.value.(Qd_ac))
println("Vn_ac: ", JuMP.value.(Vn_ac))
println("Vn_dc: ", JuMP.value.(Vn_dc))
println("Pd_ac: ", JuMP.value.(Pd_ac))
println("Pd_dc: ", JuMP.value.(Pd_dc))
println("Pc_ac: ", JuMP.value.(Pc_ac))
println("Qc_ac: ", JuMP.value.(Qc_ac))
println("Pc_dc: ", JuMP.value.(Pc_dc))

