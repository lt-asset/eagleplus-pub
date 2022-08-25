# to generate models to test the new rules
import os
import pickle

import torch
from torchsnapshot import Snapshot

def gen_all_model_dataset(model_saving_root, device=torch.device("cuda")):
    model_path = os.path.join(model_saving_root, "models")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    dataset_path = os.path.join(model_saving_root, "dataset")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    # generate model and random dataset
    from newrules.torchrec_benchmark import gen_ebc_comparison_dlrm, gen_fused_ebc_uvm, gen_ebc_comparison_scaling, get_ebc_fused_ebc_model, get_fused_ebc_uvm_model, get_random_dataset
    model_config_list_1 = []  # feed to get_ebc_fused_ebc_model
    model_config_list_2 = []  # feed to get_fused_ebc_uvm_model

    # getting models from torchrec benchmark
    # model_config_list_1.extend(gen_ebc_comparison_dlrm())
    model_config_list_1.extend(gen_ebc_comparison_scaling())
    # model_config_list_2.extend(gen_fused_ebc_uvm())

    print(len(model_config_list_1), len(model_config_list_2))

    for i in range(len(model_config_list_1)):
        config = model_config_list_1[i]

        model, model_fuse = get_ebc_fused_ebc_model(config, device)

        # save models to CPU memory?
        # model = model.to("cpu")
        # model_fuse = model_fuse.to("cpu")

        app_state = {"model": model}
        model_saving_path = os.path.join(model_path, "model_1_ebc_%d" % i)
        snapshot = Snapshot.take(path=model_saving_path, app_state=app_state)

        app_state = {"model": model_fuse}
        model_saving_path = os.path.join(model_path, "model_1_fused_ebc_%d" % i)
        snapshot = Snapshot.take(path=model_saving_path, app_state=app_state)

        dataset = get_random_dataset(config)
        example = next(iter(dataset))
        with open(os.path.join(dataset_path, "dataset_1_%d" % i), "wb") as f:
            pickle.dump(example, f)
    return


# a simple test to check if the generated and saved models can run 
def load_and_test(model_saving_root, device=torch.device("cpu")):
    # generate model and random dataset
    from newrules.torchrec_benchmark import gen_ebc_comparison_dlrm, gen_fused_ebc_uvm, gen_ebc_comparison_scaling, get_ebc_fused_ebc_model, get_fused_ebc_uvm_model, get_random_dataset
    model_config_list_1 = []  # feed to get_ebc_fused_ebc_model
    model_config_list_2 = []  # feed to get_fused_ebc_uvm_model

    # model_config_list_1.extend(gen_ebc_comparison_dlrm())
    model_config_list_1.extend(gen_ebc_comparison_scaling())
    # model_config_list_2.extend(gen_fused_ebc_uvm())


    print(len(model_config_list_1), len(model_config_list_2))

    for i in range(len(model_config_list_1)):
        config = model_config_list_1[i]

        model, model_fuse = get_ebc_fused_ebc_model(config, device)

        app_state = {"model": model}
        model_saving_path = os.path.join(model_saving_root, "models", "model_1_ebc_%d" % i)
        snapshot = Snapshot(path=model_saving_path)
        snapshot.restore(app_state)

        app_state = {"model": model_fuse}
        model_saving_path = os.path.join(model_saving_root, "models", "model_1_fused_ebc_%d" % i)
        snapshot = Snapshot(path=model_saving_path)
        snapshot.restore(app_state)

        with open(os.path.join(model_saving_root, "dataset", "dataset_1_%d" % i), "rb") as f:
            example = pickle.load(f)
        
        print(model(example.sparse_features).to_dict())
        print(model_fuse(example.sparse_features).to_dict())
        break
    return

if __name__ == "__main__":
    gen_all_model_dataset("./data")
    # load_and_test("./data")