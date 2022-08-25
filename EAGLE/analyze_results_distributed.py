# analyze the results for rule 17 
import os
import pickle
from torchsnapshot import Snapshot
import torch
import sys

from newrules.rule_17_pytorch_script import run, compare_result
from newrules.torchrec_benchmark import gen_ebc_comparison_dlrm, gen_fused_ebc_uvm, gen_ebc_comparison_scaling, get_ebc_fused_ebc_model, get_fused_ebc_uvm_model, get_random_dataset

def calculate_max_error(result_list):
    match = True
    max_diff = 0
    # print(result_list[0])
    if result_list[0] is None:
        match = False
        max_diff = None
        # print("result_list[0] is None")
    else:
        if type(result_list[0]) == torch.Tensor:
            for i in range(1, len(result_list)):
                if result_list[i] is None:
                    match = False
                    continue
                
                #diff = torch.linalg.norm(result_list[0] - result_list[i])
                diff = torch.max(torch.abs((result_list[0].to("cpu") - result_list[i].to("cpu")).view(-1)))
                # print(diff)
                if diff > max_diff:
                    max_diff = diff

                if not torch.allclose(result_list[0].to("cpu"), result_list[i].to("cpu")):
                    # print("Error: result mismatch, 0 vs {}".format(i))
                    match = False

        else: # if the output is dictionary
            # for results of different world_size or nondistributed
            for i in range(1, len(result_list)):
                if result_list[i] is None:
                    match = False
                    continue
                if result_list[0].keys() != result_list[i].keys():
                    # print("Error: result key mismatch, 0 vs {}".format(i))
                    match = False
                    continue
                    
                for key, value in result_list[0].items():
                    # print((value - result_list[i][key]).view(-1).shape)
                    #diff = torch.linalg.norm((value - result_list[i][key]))
                    diff = torch.max(torch.abs((value.to("cpu") - result_list[i][key].to("cpu")).view(-1)))
                    
                    # print(diff)
                    if diff > max_diff:
                        max_diff = diff
                    if not torch.allclose(value.to("cpu"), result_list[i][key].to("cpu")):
                        # print("Error: result value mismatch, 0 vs {}".format(i))
                        match = False
                print("0 vs {}: diff {}".format(i, diff))
                        
    return match, max_diff

def main(argv):
    model_saving_root = "./data"

    # device = torch.device("meta")

    # backend = "gloo"
    # backend = "nccl"

    result_dir_root="data/outputs"
    if not os.path.exists(result_dir_root):
        os.makedirs(result_dir_root)

    model_config_list_1 = []  # feed to get_ebc_fused_ebc_model
    model_config_list_2 = []  # feed to get_fused_ebc_uvm_model

    model_config_list_1.extend(gen_ebc_comparison_dlrm())
    model_config_list_1.extend(gen_ebc_comparison_scaling())
    model_config_list_2.extend(gen_fused_ebc_uvm())


    # print(len(model_config_list_1), len(model_config_list_2))

    if len(sys.argv) < 2:
        print('Using default model number (using 1 model) to analyze results.')
        modelnum = 1
    else: 
        modelnum = int(sys.argv[1])
        print('Using the entered modelnum ' + str(modelnum) +' to analyze results.')

    for i in range(len(model_config_list_1)):
        if i >= modelnum:
            break
        output_dir = "outputs"
        with open(os.path.join(model_saving_root, output_dir, "result_1_ebc_%d" % i), "rb") as f:
            result_ebc_list = pickle.load(f)
        match, diff = calculate_max_error(result_ebc_list)
        if not match:
            print("Fail: result_1_ebc_{}, diff: {}".format(i, diff))
        else:
            print("Pass: result_1_ebc_{}, diff: {}".format(i, diff))
        
        with open(os.path.join(model_saving_root, output_dir, "result_1_fused_ebc_%d" % i), "rb") as f:
            result_fused_ebc_list = pickle.load(f)
        match, diff = calculate_max_error(result_fused_ebc_list)
        if not match:
            print("Fail: result_1_fused_ebc_{}, diff: {}".format(i, diff))
        else:
            print("Pass: result_1_fused_ebc_{}, diff: {}".format(i, diff))
    
    return


if __name__ == "__main__":
    main(sys.argv[1:])
