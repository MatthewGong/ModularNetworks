import torch


def DenseParameters(num_layers,start=20,scale=1.3,mode="same",min_nodes=5,num_elements=118):
    """
    num layers, how many layers to make parameters for
    mode, how to make connections within layers
        same, oval, triangle 
    """

    param = [num_elements,start]
    number = start

    if mode =="same":
        for i in range(1,num_layers):
            param.append(number)

    if mode == "triangle":
        for i in range(1,num_layers):
            number = int(number/scale)
            param.append(max(number,min_nodes))

    if mode == "wedge":
        for i in range(1,num_layers):
            number = int(number*scale)
            param.append(max(number,min_nodes))

    if mode == "oval":
        for i in range(1,num_layers):
            if i <= num_layers//2:
                number = int(number*scale)
            else:
                number = int(number/scale)

            param.append(max(number,min_nodes))

    return param

def ConvolutionalParameters(num_layers,start={"out":10,"stride":1,"k":2,"p":2,"length":900},
                    scale=1.3,mode="same",min_nodes=5,num_bins=900):
    """
    num layers, how many layers to make parameters for
    mode, how to make connections within layers
        same, oval, triangle 
    """

    params = [{"in":1,"out":start["out"],"k":start["k"],"stride":start["stride"],
                "p":start["p"],"length":int(start["length"]/start["p"])}]
    
    if mode =="same":
        for i in range(0,num_layers):
            temp_param = {}
            temp_param["in"] = params[-1]["out"]
            temp_param["out"] = params[-1]["out"]
            temp_param["k"] = params[-1]["k"]
            temp_param["p"] = params[-1]["p"]
            temp_param["length"] = max(int(params[-1]["length"]/params[-1]["p"]),1)
            temp_param["stride"] = params[-1]["stride"]
            
            params.append(temp_param)
    
    if mode == "triangle":
        for i in range(0,num_layers):
            temp_param = {}
            temp_param["in"] = params[-1]["out"]
            temp_param["out"] = int(params[-1]["out"]*scale)
            temp_param["k"] = params[-1]["k"]
            temp_param["p"] = params[-1]["p"]
            temp_param["length"] = max(int(params[-1]["length"]/params[-1]["p"]),1)
            temp_param["stride"] = params[-1]["stride"]
            
            params.append(temp_param)
    
    if mode == "wedge":
        for i in range(0,num_layers):
            temp_param = {}
            temp_param["in"] = params[-1]["out"]
            temp_param["out"] = int(params[-1]["out"]/scale)
            temp_param["k"] = params[-1]["k"]
            temp_param["p"] = params[-1]["p"]
            temp_param["length"] = max(int(params[-1]["length"]/params[-1]["p"]),1)
            temp_param["stride"] = params[-1]["stride"]
            
            params.append(temp_param)
    if mode == "oval":
        number = start["out"]
        for i in range(0,num_layers):
            if i <= num_layers//2:
                number = int(number*scale)
            else:
                number = int(number/scale)

            temp_param = {}
            temp_param["in"] = params[-1]["out"]
            temp_param["out"] = number
            temp_param["k"] = params[-1]["k"]
            temp_param["p"] = params[-1]["p"]
            temp_param["length"] = max(int(params[-1]["length"]/params[-1]["p"]),1)
            temp_param["stride"] = params[-1]["stride"]
            
            params.append(temp_param)

    return params

def TaskParameters(num_clases,num_layers,conv_params=[],dense_params=[],scale=1.3,mode="same",min_nodes=5):
    """
    num layers, how many layers to make parameters for
    mode, how to make connections within layers
        same, oval, triangle 
    """
    start = 0 
    for mod in conv_params:
        start += mod[-1]["out"]*mod[-1]["length"]    
    for mod in dense_params:
        start += mod[-1]

    params = [start]
    
    number = start

    if mode =="same":
        for i in range(1,num_layers):
            params.append(number)

    if mode == "triangle":
        for i in range(1,num_layers):
            number = min(int(number/scale),500)
            params.append(max(number,min_nodes))

    if mode == "wedge":
        for i in range(1,num_layers):
            number = min(int(number*scale),500)
            params.append(max(number,min_nodes))

    if mode == "oval":
        for i in range(1,num_layers):
            if i <= num_layers//2:
                number = min(int(number*scale),500)
            else:
                number = min(int(number/scale),500)

            params.append(max(number,min_nodes))
    
    params.append(num_clases)
    return params

