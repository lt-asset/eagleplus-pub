# main file for rule 17 (DistributedModelParallel inference: distributed vs non-distributed)

from cmath import log
import numpy as np
from pkg_resources import working_set
import torch
import os
from datetime import datetime
import argparse
import traceback
import pickle
import importlib
import random
from timeit import default_timer as timer

import torchrec
from torchsnapshot import Snapshot

from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.types import ShardingType
from typing import Dict
import shutil


# execute model on a single rank
def single_rank_execution(
    rank: int,
    world_size: int,
    constraints,
    module: torch.nn.Module,
    input_data,
    backend: str,

    distributed = True,
    save_model = False,
    model_saving_path = None,
    load_model = False,
    model_loading_path = None,
    result_path = None,
    log_file = None
) -> None:
    try:
        import torch.distributed as dist
        from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
        from torchrec.distributed.model_parallel import DistributedModelParallel
        from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
        from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingPlan
        from typing import cast

        # initialize process setting
        def init_distributed_single_host(
            rank: int,
            world_size: int,
            backend: str,
            # pyre-fixme[11]: Annotation `ProcessGroup` is not defined as a type.
        ) -> dist.ProcessGroup:
            os.environ["RANK"] = f"{rank}"
            os.environ["WORLD_SIZE"] = f"{world_size}"
            dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
            return dist.group.WORLD

        # decide using cuda or cpu
        if backend == "nccl":
            device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(device)
            device_str = "cuda"
        else:
            device = torch.device("cpu")
            device_str = "cpu"
        topology = Topology(world_size=world_size, compute_device=device_str)
        pg = init_distributed_single_host(rank, world_size, backend)
        # print(topology, constraints)
        planner = EmbeddingShardingPlanner(
            topology=topology,
            constraints=constraints,
        )
        sharders = [cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())]
        # generate sharding plan
        plan: ShardingPlan = planner.collective_plan(module, sharders, pg)
        
        # wrap model with DistributedModelParallel, which will shard the model and place it on different devices
        sharded_model = DistributedModelParallel(
            module,
            env=ShardingEnv.from_process_group(pg),
            plan=plan,
            sharders=sharders,
            device=device,
        )
        print(f"rank:{rank},sharding plan: {plan}")

        app_state = {"model": sharded_model}

        # save model to given path
        if save_model:
            assert model_saving_path is not None, "model saving path is None."
            snapshot = Snapshot.take(path=model_saving_path, app_state=app_state)

        # load model from the given path
        if load_model:
            assert model_loading_path is not None, "model loading path is None."
            dist.barrier()
            snapshot = Snapshot(path=model_loading_path)
            snapshot.restore(app_state)

        # in non-distributed mode, unwrap the model
        if not distributed:
            sharded_model = sharded_model.module

        sharded_model.eval()

        # execute model
        input_data = input_data.to(device)
        result = sharded_model(input_data)

        if not type(result) == torch.Tensor:
            result = result.to_dict()
        
        print(result)
        print(result_path + "_rank_{}.p".format(rank))
        # save result to the given path
        with open(result_path + "_rank_{}.p".format(rank), "wb") as f:
            pickle.dump(result, f)
    except Exception as e:
        print(e)
        traceback.print_exc()
        with open(log_file+"_rank_{}".format(rank), "w") as f:
            f.write(str(e) + "\n")
            f.write(traceback.format_exc() + "\n")

    return sharded_model

import multiprocess

# execute model on multiple ranks
def spmd_sharing_simulation(
    module,
    constraints,
    input_data,
    backend,

    world_size = 2,
    distributed = True,
    save_model = False,
    model_saving_path = None,
    load_model = False,
    model_loading_path = None,
    result_path = None,
    log_file = None
):
    try:
        ctx = multiprocess.get_context("spawn")
        processes = []
        for rank in range(world_size):
            # generate one process for each rank
            p = ctx.Process(
                target=single_rank_execution,
                args=(
                    rank,
                    world_size,
                    constraints,
                    module,
                    input_data,
                    backend,

                    distributed,
                    save_model,
                    model_saving_path,
                    load_model,
                    model_loading_path,
                    result_path,
                    log_file
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            assert 0 == p.exitcode
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        with open(log_file, "a") as f:
            f.write(str(e) + "\n")
            f.write(traceback.format_exc() + "\n")
    finally:
        if len(processes) != 0:
            for p in processes:
                p.terminate()
                p.join()


def test_rule_distributed(input_data, module, constraints, backend, world_size=2, tmp_dir=None, log_file=None):
    try:
        seed = 0
        # fix seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # execute model on multiple ranks
        spmd_sharing_simulation(
            module,
            constraints,
            input_data,
            backend,
            world_size, 
            True, 
            True, 
            os.path.join(tmp_dir, "modelsnapshot"), 
            False, 
            None, 
            result_path=os.path.join(tmp_dir, "result_distributed_ws_multi"),
            log_file=log_file
        )
        
        print("distributed multiple ranks model finished")

        # execute model on a single rank
        spmd_sharing_simulation(
            module,
            constraints,
            input_data,
            backend,
            1, 
            True, 
            False, 
            None, 
            True, 
            os.path.join(tmp_dir, "modelsnapshot"), 
            result_path=os.path.join(tmp_dir, "result_distributed_ws_single"),
            log_file=log_file
        )
        
        print("distributed single rank model finished")

        # execute nondistributed model
        spmd_sharing_simulation(
            module,
            constraints,
            input_data,
            backend,
            1, 
            False, 
            False, 
            None, 
            True, 
            os.path.join(tmp_dir, "modelsnapshot"), 
            result_path=os.path.join(tmp_dir, "result_nondistributed"),
            log_file=log_file
        )

        print("nondistributed model finished")

        # collecting all results in three modes
        result_list = []
        for i in range(world_size):
            path = os.path.join(tmp_dir, "result_distributed_ws_multi_rank_{}.p".format(i))
            if os.path.exists(path):
                with open(path, "rb") as f:
                    result_list.append(pickle.load(f))
            else:
                result_list.append(None)
        
        path = os.path.join(tmp_dir, "result_distributed_ws_single_rank_0.p")
        if os.path.exists(path):
            with open(path, "rb") as f:
                result_list.append(pickle.load(f))
        else:
            result_list.append(None)
        
        path = os.path.join(tmp_dir, "result_nondistributed_rank_0.p")
        if os.path.exists(path):
            with open(path, "rb") as f:
                result_list.append(pickle.load(f))
        else:
            result_list.append(None)
                    
        return result_list
    
    except:
        print(traceback.format_exc())
        with open(log_file, "a+") as f:
            f.write(traceback.format_exc())
        return None
    finally:
        print("enter finally")
        print(traceback.format_exc())
        with open(log_file, "a") as f:
            f.write("enter finally")
            f.write(traceback.format_exc())
            f.write("\n")

        model_path = os.path.join(tmp_dir, "modelsnapshot")
        print(model_path)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        
        result_path_list = []
        for i in range(world_size):
            result_path_list.append(os.path.join(tmp_dir, "result_distributed_ws_multi_rank_{}.p".format(i)))
        result_path_list.append(os.path.join(tmp_dir, "result_distributed_ws_single_rank_0.p"))
        result_path_list.append(os.path.join(tmp_dir, "result_nondistributed_rank_0.p"))
        for result_path in result_path_list:
            if os.path.exists(result_path):
                os.remove(result_path)


# generate constraints for models given sharding type
def gen_constraints(model, sharding_type):
    constraints = {}
    for name, _ in model.named_modules():
        constraints[name] = ParameterConstraints(sharding_types=[sharding_type])
    return constraints


# compare all the results and check if there's any mismatch
def compare_result(result_list):
    match = True
    if result_list[0] is None:
        match = False
    else:
        if type(result_list[0]) == torch.Tensor:
            for i in range(1, len(result_list)):
                if result_list[i] is None:
                    match = False
                    break
                if not torch.allclose(result_list[0], result_list[i]):
                    print("Error: result mismatch, 0 vs {}".format(i))
                    match = False
                    break
        else:
            for i in range(1, len(result_list)):
                if result_list[i] is None:
                    match = False
                    break
                if result_list[0].keys() != result_list[i].keys():
                    print("Error: result key mismatch, 0 vs {}".format(i))
                    match = False
                    break
                for key, value in result_list[0].items():
                    if not torch.allclose(value, result_list[i][key]):
                        print("Error: result value mismatch, 0 vs {}".format(i))
                        match = False
                        break
    return match


# test run of an example model with some sample input data
def test_run_with_sample_input():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29520"

    tmp_dir = "./"
    log_file = "./log.txt"
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("")

    test_x = torchrec.KeyedJaggedTensor(
        keys = ["large_table_feature_0", "small_table_feature_0", "large_table_feature_1", "small_table_feature_1"],
        values = torch.tensor([101, 202, 303, 404, 505, 606, 101, 202, 303, 404, 505, 606]),
        lengths = torch.tensor([2, 0, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1], dtype=torch.int64),
    ).to("cpu")

    # define model using embeddingbag
    large_table_cnt = 2
    small_table_cnt = 2
    large_tables=[
        torchrec.EmbeddingBagConfig(
            name="large_table_" + str(i),
            embedding_dim=8,
            num_embeddings=4096,
            feature_names=["large_table_feature_" + str(i)],
            pooling=torchrec.PoolingType.SUM,
        ) for i in range(large_table_cnt)
    ]
    small_tables=[
        torchrec.EmbeddingBagConfig(
            name="small_table_" + str(i),
            embedding_dim=8,
            num_embeddings=1024,
            feature_names=["small_table_feature_" + str(i)],
            pooling=torchrec.PoolingType.SUM,
        ) for i in range(small_table_cnt)
    ]

    # define model as embeddingbagcollection of 4 tables/embeddingbags
    model = torchrec.EmbeddingBagCollection(
        device="meta",
        tables=large_tables + small_tables
    )

    constraints = gen_constraints(model, "TABLE_WISE")

    backend = "gloo"
    world_size = 2

    # run test
    result_list = test_rule_distributed(test_x, model, constraints, backend, world_size, tmp_dir, log_file)

    compare_result(result_list)
    return


def run(model, test_x, sharding_type="TABLE_WISE", backend="gloo", world_size=2, result_dir="data/outputs", tmp_dir="./", log_file="./log.txt"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29510" # use a  port if running multiple experiments at the same time
    
    start_time = timer()
    with open(log_file, "w") as f:
        f.write("start time: " + str(start_time) + "\n")

    # generate constraints
    constraints = gen_constraints(model, sharding_type)  # "TABLE_WISE"

    # run test
    result_list = test_rule_distributed(test_x, model, constraints, backend, world_size, tmp_dir, log_file)
    print(result_list)

    with open(result_dir, "wb") as f:
        pickle.dump(result_list, f)

    end_time = timer()
    with open(log_file, "a") as f:
        f.write("end time: " + str(end_time) + "\n")
        f.write("total time: " + str(end_time - start_time) + "\n")

    return


if __name__ == "__main__":
    # run()
    test_run_with_sample_input()
