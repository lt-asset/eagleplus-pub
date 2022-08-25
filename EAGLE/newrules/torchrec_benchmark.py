# supporting functions for generating models to test the new rules
# copied and adapted from https://github.com/pytorch/torchrec/blob/main/benchmarks/ebc_benchmarks.py 



import multiprocessing
import pickle
import sys
import os
import traceback
from typing import Dict, List, Tuple, Optional

import torch
import torchrec
from fbgemm_gpu.split_table_batched_embeddings_ops import EmbeddingLocation
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchsnapshot import Snapshot
from torch.utils.data.dataset import IterableDataset
from torchrec.datasets.random import RandomRecDataset
from torchrec.datasets.utils import Batch
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.modules.fused_embedding_modules import FusedEmbeddingBagCollection


# preset model configs
DLRM_NUM_EMBEDDINGS_PER_FEATURE = [
    45833188,
    36746,
    17245,
    7413,
    20243,
    3,
    7114,
    1441,
    62,
    29275261,
    1572176,
    345138,
    10,
    2209,
    11267,
    128,
    4,
    974,
    14,
    48937457,
    11316796,
    40094537,
    452104,
    12606,
    104,
    35,
]


# reduce large embedding tables given reduction degree
def get_shrunk_dlrm_num_embeddings(reduction_degree: int) -> List[int]:
    return [
        num_emb if num_emb < 10000000 else int(num_emb / reduction_degree)
        for num_emb in DLRM_NUM_EMBEDDINGS_PER_FEATURE
    ]


# generate configs for dlrm
def gen_ebc_comparison_dlrm():
    # Running EBC vs. FusedEBC on DLRM EMB
    model_config_list = []
    # reduction degree for the dimensions in DLRM_NUM_EMBEDDINGS_PER_FEATURE
    for reduction_degree in [128, 64, 32]:
        embedding_bag_configs: List[EmbeddingBagConfig] = [
            EmbeddingBagConfig(
                name=f"ebc_{idx}",
                embedding_dim=128,
                num_embeddings=num_embeddings,
                feature_names=[f"ebc_{idx}_feat_1"],
            )
            for idx, num_embeddings in enumerate(
                get_shrunk_dlrm_num_embeddings(reduction_degree)
            )
        ]

        # model_ebc, model_fused_ebc = get_ebc_fused_ebc_model(embedding_bag_configs, device)
        model_config_list.append(embedding_bag_configs)

    return model_config_list


# generate configs for uvm/uvm caching
def gen_fused_ebc_uvm():
    # Running DLRM EMB on FusedEBC with UVM/UVM-caching
    model_config_list = []
    embedding_bag_configs: List[EmbeddingBagConfig] = [
        EmbeddingBagConfig(
            name=f"ebc_{idx}",
            embedding_dim=128,
            num_embeddings=num_embeddings,
            feature_names=[f"ebc_{idx}_feat_1"],
        )
        for idx, num_embeddings in enumerate(get_shrunk_dlrm_num_embeddings(2))
    ]

    # model = get_fused_ebc_uvm_model(embedding_bag_configs, device, EmbeddingLocation.MANAGED_CACHING)
    model_config_list.append([embedding_bag_configs, EmbeddingLocation.MANAGED_CACHING])

    embedding_bag_configs: List[EmbeddingBagConfig] = [
        EmbeddingBagConfig(
            name=f"ebc_{idx}",
            embedding_dim=128,
            num_embeddings=num_embeddings,
            feature_names=[f"ebc_{idx}_feat_1"],
        )
        for idx, num_embeddings in enumerate(DLRM_NUM_EMBEDDINGS_PER_FEATURE)
    ]
    
    # model = get_fused_ebc_uvm_model(embedding_bag_configs, device, EmbeddingLocation.MANAGED)
    model_config_list.append([embedding_bag_configs, EmbeddingLocation.MANAGED])

    return model_config_list


# generate configs for different table sizes
def gen_ebc_comparison_scaling():
    # Running EBC vs. FusedEBC scaling experiment
    model_config_list = []

    num_tables_list = [10, 100, 1000]
    embedding_dim_list = [4, 8, 16, 32, 64, 128]
    num_embeddings_list = [4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096, 8192]

    for num_tables in num_tables_list:
        for num_embeddings in num_embeddings_list:
            for embedding_dim in embedding_dim_list:
                embedding_bag_configs: List[EmbeddingBagConfig] = [
                    EmbeddingBagConfig(
                        name=f"ebc_{idx}",
                        embedding_dim=embedding_dim,
                        num_embeddings=num_embeddings,
                        feature_names=[f"ebc_{idx}_feat_1"],
                    )
                    for idx in range(num_tables)
                ]
                # model_ebc, model_fused_ebc = get_ebc_fused_ebc_model(embedding_bag_configs, device)
                model_config_list.append(embedding_bag_configs)

    return model_config_list


# generate fused ebc model for uvm/uvm caching related configs
def get_fused_ebc_uvm_model(
    embedding_bag_configs: List[EmbeddingBagConfig],
    device: torch.device,
    location: EmbeddingLocation,
) -> Tuple[float, float]:

    fused_ebc = FusedEmbeddingBagCollection(
        tables=embedding_bag_configs,
        optimizer_type=torch.optim.SGD,
        optimizer_kwargs={"lr": 0.02},
        device=device,
        location=location,
    )
    return fused_ebc


# generate ebc model and fused ebc model for given configs
def get_ebc_fused_ebc_model(
    embedding_bag_configs: List[EmbeddingBagConfig],
    device: torch.device,
) -> Tuple[float, float, float, float, float]:

    # Simple EBC module wrapping a list of nn.EmbeddingBag
    ebc = EmbeddingBagCollection(
        tables=embedding_bag_configs,
        device=device,
    )

    # EBC with fused optimizer backed by fbgemm SplitTableBatchedEmbeddingBagsCodegen
    fused_ebc = FusedEmbeddingBagCollection(
        tables=embedding_bag_configs,
        optimizer_type=torch.optim.SGD,
        optimizer_kwargs={"lr": 0.02},
        device=device,
    )
    return ebc, fused_ebc


#generate random dataset given model configs
def get_random_dataset(
    embedding_bag_configs: List[EmbeddingBagConfig],
    batch_size: int = 64,
    num_batches: int = 10,
    num_dense_features: int = 1024,
    pooling_factors: Optional[Dict[str, int]] = None,
) -> IterableDataset[Batch]:
    # Generate a random dataset according to the embedding bag configs
    
    if pooling_factors is None:
        pooling_factors = {}

    keys = []
    ids_per_features = []
    hash_sizes = []

    for table in embedding_bag_configs:
        for feature_name in table.feature_names:
            keys.append(feature_name)
            # guess a pooling factor here
            ids_per_features.append(pooling_factors.get(feature_name, 64))
            hash_sizes.append(table.num_embeddings)

    return RandomRecDataset(
        keys=keys,
        batch_size=batch_size,
        hash_sizes=hash_sizes,
        ids_per_features=ids_per_features,
        num_dense=num_dense_features,
        num_batches=num_batches,
    )


# test model performance on random dataset
def main():
    CUDA = False
    if CUDA:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    model_config_list_1 = []  # feed to get_ebc_fused_ebc_model
    model_config_list_2 = []  # feed to get_fused_ebc_uvm_model

    model_config_list_1.extend(gen_ebc_comparison_dlrm())
    model_config_list_1.extend(gen_ebc_comparison_scaling())
    model_config_list_2.extend(gen_fused_ebc_uvm())

    print(len(model_config_list_1), len(model_config_list_2))

    config = model_config_list_1[0]
    model, model_fuse = get_ebc_fused_ebc_model(config, device)
    dataset = get_random_dataset(config)

    input = 0
    for data in dataset:
        input = data
        break
    sparse_features = input.sparse_features
    print(model(sparse_features))
    
    print(model_fuse(sparse_features))


if __name__ == "__main__":
    main()

