using CSV
using DataFrames
using GraphPlot
using LightGraphs
using XLSX
using StringEncodings

path = "C:\\Users\\13733\\Desktop\\DistributionPowerFlow-main"
data = "\\data\\dmsrtnet_cal_info.dat"
output_dir = path * "\\data"

input_file_path = path * data
output_file_path = path * "\\data\\test_data.xlsx"

# Read the content of the text file with GBK encoding
enc = StringEncodings.Encoding("GBK")
file_content = open(input_file_path) do file
    StringEncodings.decode(read(file), enc)
end
lines = split(file_content, "\n")

# Define section headers
headers = Dict(
    "switch" => ["@", "编号", "开关状态", "P量测", "Q量测", "IA量测", "IB量测", "IC量测", "P计算", "Q计算", "IA计算", "IB计算", "IC计算", "Uab计算", "相角", "基准电压U", "设备ID", "设备名称", "所属馈线", "首端点号", "末端点号", "首端节点", "末端节点"],
    "load" => ["@", "编号", "母线编号", "P量测", "Q量测", "I量测", "P计算", "Q计算", "I计算", "Uab计算", "Pa计算", "Qa计算", "Pb计算", "Qb计算", "Pc计算", "Qc计算", "基准电压U", "节点号", "设备ID", "设备名", "厂站名", "所属馈线"],
    "bus" => ["@", "母线编号", "V量测", "V标幺", "Va", "Vb", "Vc", "Ubase", "PhaseA", "PhaseB", "PhaseC", "节点数", "节点描述", "所属馈线"],
    "branch" => ["@", "序号", "编号", "pos", "fBus", "tBus", "Pij", "Qij", "Pji", "Qji", "Vf", "Vt", "IA", "IB", "IC", "Imax", "电流负载率", "首节点", "末节点", "支路ID", "支路名称", "所属馈线"],
    "source" => ["@", "BusNo3", "PG", "QG", "Vmea", "Vmag", "busid", "母线id", "断路器描述", "所属馈线"],
    "island" => ["@", "序号", "islid", "设备描述"]
)

# Initialize containers for each section's data
data_dict = Dict(
    "switch" => [],
    "load" => [],
    "bus" => [],
    "branch" => [],
    "source" => [],
    "island" => []
)


global current_section = ""
global skip_lines = 0
global island_name = ""
# Function to parse a line into sections
function parse_line(line)
    parts = split(replace(line, r"[,\\s]\\s*" => " "), " ")
    return parts
end

# Parse the file content into sections
for line in lines
    line = strip(line)
    if startswith(line, "<cb_info>")
        global current_section = "switch"
        global skip_lines = 1
    elseif startswith(line, "</cb_info>")
        global current_section = ""
        global skip_lines = 1
    elseif startswith(line, "<load_info>")
        global current_section = "load"
        global skip_lines = 1
    elseif startswith(line, "</load_info>")
        global current_section = ""
        global skip_lines = 1
    elseif startswith(line, "<bus_info>")
        global current_section = "bus"
        global skip_lines = 1
    elseif startswith(line, "</bus_info>")
        global current_section = ""
        global skip_lines = 1
    elseif startswith(line, "<Branch_info>")
        global current_section = "branch"
        global skip_lines = 1
    elseif startswith(line, "</Branch_info>")
        global current_section = ""
    elseif startswith(line, "<Gen_info>")
        global current_section = "source"
        global skip_lines = 1
    elseif startswith(line, "</Gen_info>")
        global current_section = ""
    elseif startswith(line, "<Island_info>")
        global current_section = "island"
        global skip_lines = 1
    elseif startswith(line, "</Island_info>")
        global current_section = ""
        global skip_lines = 1
    elseif skip_lines > 0
        global skip_lines -= 1
    elseif skip_lines <= 0 && current_section in keys(headers)
        info = split(replace(line, r"[,\s]\s*" => " "), " ")
        if current_section == "island"
            global island_name = info[end]
            push!(data_dict[current_section], info)
        elseif length(info) != length(headers[current_section])
            if current_section == "switch"
                info[17] = ""
            elseif current_section in ["load", "bus", "branch", "source"]
                push!(info, island_name)
                push!(data_dict[current_section], info)
            end
        else
            push!(data_dict[current_section], info)
        end
    end
end

function islandpf(island)
    if island==0
        length_source=1
        length_load=1
        length_bus=3
        length_branch=2
        source_start=1
        load_start=1
        bus_start=1
        branch_start=1
    elseif island==1
        length_source=1
        length_load=9
        length_bus=4
        length_branch=3
        source_start=2
        load_start=2
        bus_start=4
        branch_start=3
    elseif island==2
        length_source=1
        length_load=5
        length_bus=6
        length_branch=5
        source_start=3
        load_start=11
        bus_start=8
        branch_start=6
    elseif island==3
        length_source=1
        length_load=2
        length_bus=3
        length_branch=2
        source_start=4
        load_start=16
        bus_start=14
        branch_start=11
    elseif island==4
        length_source=2
        length_load=127
        length_bus=112
        length_branch=118
        source_start=5
        load_start=18
        bus_start=17
        branch_start=13
    elseif island==5
        length_source=1
        length_load=1
        length_bus=4
        length_branch=3
        source_start=7
        load_start=145
        bus_start=129
        branch_start=131
    elseif island==6
        length_source=1
        length_load=3
        length_bus=5
        length_branch=4
        source_start=8
        load_start=146
        bus_start=133
        branch_start=134
    else
        length_source=1
        length_load=1
        length_bus=3
        length_branch=2
        source_start=9
        load_start=149
        bus_start=138
        branch_start=138
    end
    # ## Write the data to an matpower case file
    #rewrite the source data
    #source  BusNo3  PG  QG  Vmea  Vmag  busid  母线id  断路器描述  所属馈线
        #source has 1 row
        gen=data_dict["source"][source_start:source_start+length_source-1,:]
        matrix=zeros(length_source,5)
        for i=1 : length_source
            matrix[i, 1:5] = [parse(Float64, x) for x in gen[i][2:6]] 
        end
        for i=1:length_source
            if(matrix[i,1]<0)
                matrix[i,1]+=2
            else
                matrix[i,1]+=1
            end
        end
        gen=zeros(length_source,21)
        
        gen[:,1]=matrix[:,1]    #GEN_BUS
        gen[:,2]=matrix[:,2]    #Pg
        gen[:,3]=matrix[:,3]    #Qg
        #gen[:,4]=matrix[:,4]
        #gen[:,5]=matrix[:,5]
        gen[:,6]=matrix[:,5]/10 #Vg
        gen[:,7].=100           #MBASE
        gen[:,8].=1             #status
        gen[:,9].=100           #Pmax
        gen[:,10].=0            #Pmin
    #rewrite the load data
    #load  P量测  Q量测  I量测  P计算  Q计算  I计算  Uab计算  Pa计算  Qa计算  Pb计算  Qb计算  Pc计算  Qc计算  基准电压U  
        #load has 1 row
        load=data_dict["load"][load_start:length_load+load_start-1,:]
        matrix=zeros(length_load,15)
        for i=1 : length_load
            matrix[i, 1:15] = [parse(Float64, x) for x in load[i][3:17]]  
            
        end
        matrix[:,1].+=1
        load=matrix
    #rewrite the bus data 
    #bus  Vm   V标幺     Va  Vb   Vc   Vbase    PhaseA	PhaseB	PhaseC	节点数
        #bus has 3 rows
        bus=data_dict["bus"][bus_start:length_bus+bus_start-1,:]
        matrix=zeros(length_bus,14)
        for i=1:length_bus
            matrix[i, 1:11] = [parse(Float64, x) for x in bus[i][2:12]]
        end
        matrix = sortslices(matrix; dims=1, by=x->x[1])
        matrix[:,1].+=1
        bus=zeros(length_bus,13)
        bus[:,1]=matrix[:,1]    #bus
        #define bustype         
        bus[:,2].=1             
        #bus[round.(Int, load[:,1]), 2] .= 1
        bus[1,2]=3
        #define Pd
        bus[:,3].=0     
        for i=1:length_load
            bus[round.(Int, load[i,1]),3]+=load[i,2]
        end
        #define Qd
        bus[:,4].=0
        for  i=1:length_load
        bus[round.(Int, load[i,1]),4]+=load[i,3]
        end
        #Gs,Bs
        bus[:,5:6].=0
        #area
        bus[:,7].=1
        #Vm
        bus[:,8].=matrix[:,3]
        #Va
        bus[:,9].=0
        #baseKV
        bus[:,10].=10
        #zone
        bus[:,11].=1
        #Vmax
        bus[:,12].=1.06
        #Vmin
        bus[:,13].=0.94
    #rewrite the branch data
    #  fBus  tBus  Pij  Qij  Pji  Qji  Vf  Vt  IA  IB  IC  Imax  电流负载率 
    #branch has 2 row
        branch=data_dict["branch"][branch_start:branch_start+length_branch-1,:]
        matrix=zeros(length_branch,13)
        for i=1 : length_branch
            matrix[i, 1:13] = [parse(Float64, x) for x in branch[i][5:17]] 
        end
        matrix[:,1].+=1
        matrix[:,2].+=1
        delta_P=matrix[:,3].-matrix[:,5]
        delta_P = abs.(delta_P)
        delta_Q=matrix[:,4]-matrix[:,6]
        delta_Q = abs.(delta_Q)
        r=delta_P./(matrix[:,9].^2)
        x=delta_Q./(matrix[:,9].^2)
        branch=zeros(length_branch,13)
        #fbus,tbus
        branch[:,1:2]=matrix[:,1:2]
        #r
        branch[:,3]=r
        #x
        branch[:,4]=x
        #b
        branch[:,5].=0
        #rateA,rateB,rateC
        branch[:,6:8].=0
        #ratio
        branch[:,9]=matrix[:,13]
        #angle
        branch[:,10].=0
        #status
        branch[:,11].=1
        #angmin
        branch[:,12].=-360
        #angmax
        branch[:,13].=360

        mpc=Dict(       
            "version" => "2",
            "baseMVA" => 100.0,
            "bus" => bus,
            "gen" => gen,
            "branch" => branch,
        )
        return mpc
end


# Detect the current working operating system
if Sys.iswindows()
    # Add the path to the data folder
    push!(LOAD_PATH, pwd()*"\\src\\")
    include(pwd()*"\\data\\case118.jl")
else
    # Add the path to the data folder
    push!(LOAD_PATH, pwd()*"/src/")
    include(pwd()*"/data/case118.jl")
    using AppleAccelerate
end
# push!(LOAD_PATH, pwd()*"\\data\\");
using PowerFlow
using MATLAB
using Plots
# Start a MATLAB engine
# mat"addpath('C:\\Codes\\matpower')"
# mpc = case118();
opt = PowerFlow.options() # The initial settings 
opt["PF"]["NR_ALG"] = "bicgstab";
opt["PF"]["ENFORCE_Q_LIMS"]=0
matrix=zeros(140,3)
gmatrix=zeros(9,3)
    mpc=islandpf(0)
    @time mpc = PowerFlow.runpf(mpc, opt);
    matrix[1:3,1]=mpc["bus"][:,1]
    matrix[1:3,3]=mpc["bus"][:,8]
    gmatrix[1,1]=mpc["gen"][1,2]
    gmatrix[1,2]=mpc["gen"][1,3]
    # data_dict["branch"][1:2,:]=mpc["branch"]
    mpc=islandpf(1)
    @time mpc = PowerFlow.runpf(mpc, opt);
    matrix[4:7,1]=mpc["bus"][:,1]
    matrix[4:7,3]=mpc["bus"][:,8]
    gmatrix[2,1]=mpc["gen"][1,2]
    gmatrix[2,2]=mpc["gen"][1,3]
    # data_dict["branch"][3,:]=mpc["branch"]
    mpc=islandpf(2)
    @time mpc = PowerFlow.runpf(mpc, opt);
    matrix[8:13,1]=mpc["bus"][:,1]
    matrix[8:13,3]=mpc["bus"][:,8]
    gmatrix[3,1]=mpc["gen"][1,2]
    gmatrix[3,2]=mpc["gen"][1,3]
    # data_dict["branch"][4,:]=mpc["branch"]
    mpc=islandpf(3)
    @time mpc = PowerFlow.runpf(mpc, opt);
    matrix[14:16,1]=mpc["bus"][:,1]
    matrix[14:16,3]=mpc["bus"][:,8]
    gmatrix[4,1]=mpc["gen"][1,2]
    gmatrix[4,2]=mpc["gen"][1,3]
    # data_dict["branch"][5,:]=mpc["branch"]
    mpc=islandpf(4)
    @time mpc = PowerFlow.runpf(mpc, opt);
    matrix[17:128,1]=mpc["bus"][:,1]
    matrix[17:128,3]=mpc["bus"][:,8]
    gmatrix[5,1]=mpc["gen"][1,2]
    gmatrix[5,2]=mpc["gen"][1,3]
    gmatrix[6,1]=mpc["gen"][2,2]
    gmatrix[6,2]=mpc["gen"][2,3]
    # data_dict["branch"][6,:]=mpc["branch"]
    mpc=islandpf(5)
    @time mpc = PowerFlow.runpf(mpc, opt);
    matrix[129:132,1]=mpc["bus"][:,1]
    matrix[129:132,3]=mpc["bus"][:,8]
    gmatrix[7,1]=mpc["gen"][1,2]
    gmatrix[7,2]=mpc["gen"][1,3]
    # data_dict["branch"][131,:]=mpc["branch"]
    mpc=islandpf(6)
    @time mpc = PowerFlow.runpf(mpc, opt);
    matrix[133:137,1]=mpc["bus"][:,1]
    matrix[133:137,3]=mpc["bus"][:,8]
    gmatrix[8,1]=mpc["gen"][1,2]
    gmatrix[8,2]=mpc["gen"][1,3]
    # data_dict["branch"][134,:]=mpc["branch"]
    mpc=islandpf(7)
    @time mpc = PowerFlow.runpf(mpc, opt);
    matrix[138:140,1]=mpc["bus"][:,1]
    matrix[138:140,3]=mpc["bus"][:,8]
    gmatrix[9,1]=mpc["gen"][1,2]
    gmatrix[9,2]=mpc["gen"][1,3]
    # data_dict["branch"][138,:]=mpc["branch"]

    for i in 1:length(data_dict["bus"])
        if length(data_dict["bus"][i]) >= 2
            data_dict["bus"][i][2] = string(matrix[i,1])  # convert Float64 to String
            data_dict["bus"][i][3] = string(matrix[i,3])  # convert Float64 to String
        end
    end
    for i in 1:length(data_dict["source"])
        if length(data_dict["source"][i]) >= 2
            data_dict["source"][i][3] = string(gmatrix[i,1])  # convert Float64 to String
            data_dict["source"][i][4] = string(gmatrix[i,2])  # convert Float64 to String
        end
    end
    #save the data to the file and adjust the format
    #island0
    new_data_dict = Dict(
        "switch" => data_dict["switch"][1:4,:],
        "load" => data_dict["load"][1:1,:],
        "bus" => data_dict["bus"][1:3,:],
        "branch" => data_dict["branch"][1:2,:],
        "source" => data_dict["source"][1:1,:],
        "island" => data_dict["island"][1:1,:]
    )
    using DataStructures
    ordered_data_dict = OrderedDict(
        "island" => new_data_dict["island"],   
        "switch" => new_data_dict["switch"], 
        "load" => new_data_dict["load"],
        "bus" => new_data_dict["bus"],
        "branch" => new_data_dict["branch"],
        "source" => new_data_dict["source"]
    )
    buffer = IOBuffer()

    for (key, values) in ordered_data_dict
        write(buffer, "<$key>\n")
        for value in values
            write(buffer, "#\t" * join(value, "\t") * "\n")
        end
        write(buffer, "</$key>\n")
    end
    
    open("data.dat", "w") do io
        write(io, String(take!(buffer)))
    end
    #island1
    new_data_dict = Dict(
        "switch" => data_dict["switch"][5:19,:],
        "load" => data_dict["load"][2:10,:],
        "bus" => data_dict["bus"][4:7,:],
        "branch" => data_dict["branch"][3:5,:],
        "source" => data_dict["source"][2:2,:],
        "island" => data_dict["island"][2:2,:]
    )
    ordered_data_dict = OrderedDict(
        "island" => new_data_dict["island"],   
        "switch" => new_data_dict["switch"], 
        "load" => new_data_dict["load"],
        "bus" => new_data_dict["bus"],
        "branch" => new_data_dict["branch"],
        "source" => new_data_dict["source"]
    )
    buffer = IOBuffer()
    for (key, values) in ordered_data_dict
        write(buffer, "<$key>\n")
        for value in values
            write(buffer, "#\t" * join(value, "\t") * "\n")
        end
        write(buffer, "</$key>\n")
    end
    
    open("data.dat", "a") do io
        write(io, String(take!(buffer)))
    end
    #island2
    new_data_dict = Dict(
        "switch" => data_dict["switch"][20:32,:],
        "load" => data_dict["load"][11:15,:],
        "bus" => data_dict["bus"][8:13,:],
        "branch" => data_dict["branch"][6:10,:],
        "source" => data_dict["source"][3:3,:],
        "island" => data_dict["island"][3:3,:]
    )
    ordered_data_dict = OrderedDict(
        "island" => new_data_dict["island"],   
        "switch" => new_data_dict["switch"], 
        "load" => new_data_dict["load"],
        "bus" => new_data_dict["bus"],
        "branch" => new_data_dict["branch"],
        "source" => new_data_dict["source"]
    )
    buffer = IOBuffer()
    for (key, values) in ordered_data_dict
        write(buffer, "<$key>\n")
        for value in values
            write(buffer, "#\t" * join(value, "\t") * "\n")
        end
        write(buffer, "</$key>\n")
    end

    open("data.dat", "a") do io
        write(io, String(take!(buffer)))
    end
    #island3
    new_data_dict = Dict(
        "switch" => data_dict["switch"][33:35,:],
        "load" => data_dict["load"][16:17,:],
        "bus" => data_dict["bus"][14:16,:],
        "branch" => data_dict["branch"][11:12,:],
        "source" => data_dict["source"][4:4,:],
        "island" => data_dict["island"][4:4,:]
    )
    ordered_data_dict = OrderedDict(
        "island" => new_data_dict["island"],   
        "switch" => new_data_dict["switch"], 
        "load" => new_data_dict["load"],
        "bus" => new_data_dict["bus"],
        "branch" => new_data_dict["branch"],
        "source" => new_data_dict["source"]
    )
    buffer = IOBuffer()
    for (key, values) in ordered_data_dict
        write(buffer, "<$key>\n")
        for value in values
            write(buffer, "#\t" * join(value, "\t") * "\n")
        end
        write(buffer, "</$key>\n")
    end

    open("data.dat", "a") do io
        write(io, String(take!(buffer)))
    end
    #island4
    new_data_dict = Dict(
        "switch" => data_dict["switch"][36:535,:],
        "load" => data_dict["load"][18:144,:],
        "bus" => data_dict["bus"][17:128,:],
        "branch" => data_dict["branch"][13:130,:],
        "source" => data_dict["source"][5:6,:],
        "island" => data_dict["island"][5:5,:]
    )
    ordered_data_dict = OrderedDict(
        "island" => new_data_dict["island"],   
        "switch" => new_data_dict["switch"], 
        "load" => new_data_dict["load"],
        "bus" => new_data_dict["bus"],
        "branch" => new_data_dict["branch"],
        "source" => new_data_dict["source"]
    )
    buffer = IOBuffer()
    for (key, values) in ordered_data_dict
        write(buffer, "<$key>\n")
        for value in values
            write(buffer, "#\t" * join(value, "\t") * "\n")
        end
        write(buffer, "</$key>\n")
    end

    open("data.dat", "a") do io
        write(io, String(take!(buffer)))
    end
    #island5
    new_data_dict = Dict(
        "switch" => data_dict["switch"][536:540,:],
        "load" => data_dict["load"][145:145,:],
        "bus" => data_dict["bus"][129:132,:],
        "branch" => data_dict["branch"][131:133,:],
        "source" => data_dict["source"][7:7,:],
        "island" => data_dict["island"][6:6,:]
    )
    ordered_data_dict = OrderedDict(
        "island" => new_data_dict["island"],   
        "switch" => new_data_dict["switch"], 
        "load" => new_data_dict["load"],
        "bus" => new_data_dict["bus"],
        "branch" => new_data_dict["branch"],
        "source" => new_data_dict["source"]
    )
    buffer = IOBuffer()
    for (key, values) in ordered_data_dict
        write(buffer, "<$key>\n")
        for value in values
            write(buffer, "#\t" * join(value, "\t") * "\n")
        end
        write(buffer, "</$key>\n")
    end

    open("data.dat", "a") do io
        write(io, String(take!(buffer)))
    end
    #island6
    new_data_dict = Dict(
        "switch" => data_dict["switch"][541:545,:],
        "load" => data_dict["load"][146:148,:],
        "bus" => data_dict["bus"][133:137,:],
        "branch" => data_dict["branch"][134:137,:],
        "source" => data_dict["source"][8:8,:],
        "island" => data_dict["island"][7:7,:]
    )
    ordered_data_dict = OrderedDict(
        "island" => new_data_dict["island"],   
        "switch" => new_data_dict["switch"], 
        "load" => new_data_dict["load"],
        "bus" => new_data_dict["bus"],
        "branch" => new_data_dict["branch"],
        "source" => new_data_dict["source"]
    )
    buffer = IOBuffer()
    for (key, values) in ordered_data_dict
        write(buffer, "<$key>\n")
        for value in values
            write(buffer, "#\t" * join(value, "\t") * "\n")
        end
        write(buffer, "</$key>\n")
    end

    open("data.dat", "a") do io
        write(io, String(take!(buffer)))
    end

    #island7
    new_data_dict = Dict(
        "switch" => data_dict["switch"][546:547,:],
        "load" => data_dict["load"][149:149,:],
        "bus" => data_dict["bus"][138:140,:],
        "branch" => data_dict["branch"][138:139,:],
        "source" => data_dict["source"][9:9,:],
        "island" => data_dict["island"][8:8,:]
    )
    ordered_data_dict = OrderedDict(
        "island" => new_data_dict["island"],   
        "switch" => new_data_dict["switch"], 
        "load" => new_data_dict["load"],
        "bus" => new_data_dict["bus"],
        "branch" => new_data_dict["branch"],
        "source" => new_data_dict["source"]
    )
    buffer = IOBuffer()
    for (key, values) in ordered_data_dict
        write(buffer, "<$key>\n")
        for value in values
            write(buffer, "#\t" * join(value, "\t") * "\n")
        end
        write(buffer, "</$key>\n")
    end

    open("data.dat", "a") do io
        write(io, String(take!(buffer)))
    end