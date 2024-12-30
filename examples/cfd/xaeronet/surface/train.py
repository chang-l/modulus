# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This code defines a distributed training pipeline for training MeshGraphNet at scale,
which operates on partitioned graph data for the AWS drivaer dataset. It includes
loading partitioned graphs from .bin files, normalizing node and edge features using
precomputed statistics, and training the model in parallel using DistributedDataParallel
across multiple GPUs. The training loop involves computing predictions for each graph
partition, calculating loss, and updating model parameters using mixed precision.
Periodic checkpointing is performed to save the model, optimizer state, and training
progress. Validation is also conducted every few epochs, where predictions are compared
against ground truth values, and results are saved as point clouds. The code logs training
and validation metrics to TensorBoard and optionally integrates with Weights and Biases for
experiment tracking.
"""

import os
import sys
import json
import dgl
import pyvista as pv
import torch
import hydra
import numpy as np
from hydra.utils import to_absolute_path
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig

from modulus.distributed import DistributedManager, mark_module_as_shared
from modulus.launch.logging import initialize_wandb
from modulus.models.meshgraphnet import MeshGraphNet
import time

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from dataloader import create_dataloader
from utils import (
    find_bin_files,
    save_checkpoint,
    load_checkpoint,
    count_trainable_params,
)

from modulus.models.gnn_layers import (
    CuGraphCSC,
    partition_graph_nodewise,
)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # Enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = cfg.enable_cudnn_benchmark

    # Instantiate the distributed manager
    torch.set_num_threads(1) ## Very import to limit the number of threads to 1 for performance (especially for dist-mgn)
    DistributedManager.initialize()
    dist = DistributedManager()
    if cfg.dist_mgn.enabled:
        DistributedManager.create_process_subgroup(
            name=cfg.dist_mgn.proc_group_name,
            size=dist.world_size,
        )

    device = dist.device
    print(f"Rank {dist.rank} of {dist.world_size}")

    # Instantiate the writers
    if dist.rank == 0:
        writer = SummaryWriter(log_dir="tensorboard")
        initialize_wandb(
            project="aws_drivaer",
            entity="Modulus",
            name="aws_drivaer",
            mode="disabled",
            group="group",
            save_code=True,
        )

    # AMP Configs
    amp_dtype = torch.bfloat16
    amp_device = "cuda"

    # Find all .bin files in the directory
    train_dataset = find_bin_files(to_absolute_path(cfg.partitions_path))
    valid_dataset = find_bin_files(to_absolute_path(cfg.validation_partitions_path))

    # Prepare the stats
    with open(to_absolute_path(cfg.stats_file), "r") as f:
        stats = json.load(f)
    mean = stats["mean"]
    std = stats["std_dev"]

    # Create DataLoader
    train_dataloader = create_dataloader(
        train_dataset,
        mean,
        std,
        batch_size=1,
        prefetch_factor=None,
        #use_ddp=True,
        #use_ddp=dist.world_size > 1 and not cfg.dist_mgn.enabled,
        use_ddp=False, #dist.world_size > 1 and not cfg.dist_mgn.enabled,
        num_workers=4,
    )
    # graphs is a list of graphs, each graph is a list of partitions
    graphs = [graph_partitions for graph_partitions, _ in train_dataloader]
    if cfg.dist_mgn.enabled and cfg.dist_mgn.metis_reorder_part > 0:
        if dist.rank == 0:
            npart = cfg.dist_mgn.metis_reorder_part
            for i in range(len(graphs)):
                file_path = os.path.join(cfg.partitions_path, "one_partitions_reorder_{i}.bin")
                subgraphs = graphs[i]
                assert len(subgraphs) == 1
                graph = subgraphs[0]
                graph = dgl.reorder_graph(graph, node_permute_algo='metis', permute_config={'k': npart})
                print("Reordered the graph", graph.ndata['_ID'])
                dgl.save_graphs(file_path, [graph])
        torch.distributed.barrier()
        num_graphs = len(graphs)
        graphs = []
        for i in range(num_graphs):
            file_path = os.path.join(cfg.partitions_path, "one_partitions_reorder_{i}.bin")
            subgraphs, _ = dgl.load_graphs(file_path)
            assert len(subgraphs) == 1
            graphs.append(subgraphs)

    if dist.rank == 0:
        validation_dataloader = create_dataloader(
            valid_dataset,
            mean,
            std,
            batch_size=1,
            prefetch_factor=None,
            use_ddp=False,
            num_workers=4,
        )
        validation_graphs = [
            graph_partitions for graph_partitions, _ in validation_dataloader
        ]
        validation_ids = [id[0] for _, id in validation_dataloader]
        print(f"Training dataset size: {len(graphs)*dist.world_size}")
        print(f"Validation dataset size: {len(validation_dataloader)}")

    ######################################
    # Training #
    ######################################

    # Initialize model
    model = MeshGraphNet(
        input_dim_nodes=24,
        input_dim_edges=4,
        output_dim=4,
        processor_size=cfg.num_message_passing_layers,
        aggregation="sum",
        hidden_dim_node_encoder=cfg.hidden_dim,
        hidden_dim_edge_encoder=cfg.hidden_dim,
        hidden_dim_node_decoder=cfg.hidden_dim,
        mlp_activation_fn=cfg.activation,
        do_concat_trick=cfg.use_concat_trick,
        num_processor_checkpoint_segments=cfg.checkpoint_segments,
        checkpoint_offloading=cfg.checkpoint_offloading,
    ).to(device)
    print(f"Number of trainable parameters: {count_trainable_params(model)}")

    # DistributedDataParallel wrapper
    #if dist.world_size > 1 and not cfg.dist_mgn.enabled:
    if not cfg.dist_mgn.enabled: # always DDP model if xaeronet
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
            gradient_as_bucket_view=True,
            #static_graph=True, # disable as it does not work with ddp.no_sync()
        )
    if cfg.dist_mgn.enabled and dist.world_size > 1:
        mark_module_as_shared(model, cfg.dist_mgn.proc_group_name)

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=2000, eta_min=1e-6
    )
    scaler = GradScaler()
    print("Instantiated the model and optimizer")

    # Disable checkpoint for perf benchmarking
    #start_epoch, _ = load_checkpoint(
    #    model, optimizer, scaler, scheduler, cfg.checkpoint_filename
    #)
    start_epoch = 0

    # Training loop
    print("Training started")
    total_time = []
    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        total_loss = 0
        for i in range(len(graphs)):
            optimizer.zero_grad()
            subgraphs = graphs[i]  # Get the partitions of the graph
            if cfg.num_partitions == 1 and cfg.dist_mgn.enabled:
                graph = subgraphs[0]
                offsets, indices, edge_perm = graph.adj_tensors("csc")
                graph_partition = partition_graph_nodewise(
                    offsets.to(dtype=torch.int64),
                    indices.to(dtype=torch.int64),
                    dist.world_size,
                    dist.rank,
                    dist.device,
                    matrix_decomp=True,
                )
                graph_multi_gpu = CuGraphCSC(
                    offsets.to(dist.device),
                    indices.to(dist.device),
                    graph.num_src_nodes(),
                    graph.num_dst_nodes(),
                    partition_size=dist.world_size,
                    partition_group_name=cfg.dist_mgn.proc_group_name,
                    graph_partition=graph_partition,
                )
                torch.cuda.synchronize()
                torch.distributed.barrier()
                start_timer = time.time()
                with torch.autocast(amp_device, enabled=True, dtype=amp_dtype):
                    part = graph
                    ndata = torch.cat(
                        (
                            part.ndata["coordinates"],
                            part.ndata["normals"],
                            torch.sin(2 * np.pi * part.ndata["coordinates"]),
                            torch.cos(2 * np.pi * part.ndata["coordinates"]),
                            torch.sin(4 * np.pi * part.ndata["coordinates"]),
                            torch.cos(4 * np.pi * part.ndata["coordinates"]),
                            torch.sin(8 * np.pi * part.ndata["coordinates"]),
                            torch.cos(8 * np.pi * part.ndata["coordinates"]),
                        ),
                        dim=1,
                    )
                    node_feats = graph_multi_gpu.get_dst_node_features_in_partition(
                        ndata.to(device)
                    )
                    edata = graph.edata["x"][edge_perm]
                    edge_feats = graph_multi_gpu.get_edge_features_in_partition(
                        edata.to(device)
                    )
                    y = torch.cat(
                        (part.ndata["pressure"], part.ndata["shear_stress"]), dim=1
                    )
                    target = graph_multi_gpu.get_dst_node_features_in_partition(
                        y.to(device)
                    )
                    pred = model(node_feats, edge_feats, graph_multi_gpu)
                    loss = (
                        torch.mean((pred - target) ** 2)
                        / dist.world_size
                    )
                    total_loss += loss.item()
                scaler.scale(loss).backward()
            else:
                assert cfg.num_partitions == len(subgraphs)
                assert cfg.num_partitions % dist.world_size == 0 # num_partitions should be divisible by world_size for this experiment
                micro_batch_size = cfg.num_partitions // dist.world_size
                torch.cuda.synchronize()
                torch.distributed.barrier()
                start_timer = time.time()
                # iterate over micro-batches except the last one
                for j in range(micro_batch_size * dist.rank, micro_batch_size * (dist.rank + 1) - 1):
                    with model.no_sync():
                        with torch.autocast(amp_device, enabled=True, dtype=amp_dtype):
                            part = subgraphs[j].to(device)
                            ndata = torch.cat(
                                (
                                    part.ndata["coordinates"],
                                    part.ndata["normals"],
                                    torch.sin(2 * np.pi * part.ndata["coordinates"]),
                                    torch.cos(2 * np.pi * part.ndata["coordinates"]),
                                    torch.sin(4 * np.pi * part.ndata["coordinates"]),
                                    torch.cos(4 * np.pi * part.ndata["coordinates"]),
                                    torch.sin(8 * np.pi * part.ndata["coordinates"]),
                                    torch.cos(8 * np.pi * part.ndata["coordinates"]),
                                ),
                                dim=1,
                            )
                            pred = model(ndata, part.edata["x"], part)
                            pred_filtered = pred[part.ndata["inner_node"].bool(), :]
                            target = torch.cat(
                                (part.ndata["pressure"], part.ndata["shear_stress"]), dim=1
                            )
                            target_filtered = target[part.ndata["inner_node"].bool()]
                            loss = (
                                torch.mean((pred_filtered - target_filtered) ** 2)
                                / cfg.num_partitions
                            )
                            total_loss += loss.item()
                        scaler.scale(loss).backward()
                # last micro-batch
                j = micro_batch_size * (dist.rank + 1) - 1
                with torch.autocast(amp_device, enabled=True, dtype=amp_dtype):
                    part = subgraphs[j].to(device)
                    ndata = torch.cat(
                        (
                            part.ndata["coordinates"],
                            part.ndata["normals"],
                            torch.sin(2 * np.pi * part.ndata["coordinates"]),
                            torch.cos(2 * np.pi * part.ndata["coordinates"]),
                            torch.sin(4 * np.pi * part.ndata["coordinates"]),
                            torch.cos(4 * np.pi * part.ndata["coordinates"]),
                            torch.sin(8 * np.pi * part.ndata["coordinates"]),
                            torch.cos(8 * np.pi * part.ndata["coordinates"]),
                        ),
                        dim=1,
                    )
                    pred = model(ndata, part.edata["x"], part)
                    pred_filtered = pred[part.ndata["inner_node"].bool(), :]
                    target = torch.cat(
                        (part.ndata["pressure"], part.ndata["shear_stress"]), dim=1
                    )
                    target_filtered = target[part.ndata["inner_node"].bool()]
                    loss = (
                        torch.mean((pred_filtered - target_filtered) ** 2)
                        / cfg.num_partitions
                    )
                    total_loss += loss.item()
                scaler.scale(loss).backward() # last micro-batch sync gradient

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 32.0)
            scaler.step(optimizer)
            scaler.update()

            torch.cuda.synchronize()
            torch.distributed.barrier()
            end_timer = time.time()
            total_time.append(end_timer - start_timer)

        scheduler.step()

        # Log the training loss
        total_loss = torch.tensor(total_loss).cuda()
        torch.distributed.reduce(total_loss, 0)
        if dist.rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1}, Learning Rate: {current_lr}, Total Loss: {total_loss.item() / len(graphs)}"
            )
            writer.add_scalar("training_loss", total_loss.item() / len(graphs), epoch)
            writer.add_scalar("learning_rate", current_lr, epoch)

        # Save checkpoint periodically
        if (epoch) % cfg.save_checkpoint_freq == 0:
            if dist.world_size > 1:
                torch.distributed.barrier()
            if dist.rank == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scaler,
                    scheduler,
                    epoch + 1,
                    loss.item(),
                    cfg.checkpoint_filename,
                )
        ######################################
        # Validation #
        ######################################

        # turn off validation for perf benchmarking
        if dist.rank == 0 and epoch % cfg.validation_freq == 0 and False:
            valid_loss = 0

            for i in range(len(validation_graphs)):
                # Placeholder to accumulate predictions and node features for the full graph's nodes
                num_nodes = sum(
                    [subgraph.num_nodes() for subgraph in validation_graphs[i]]
                )

                # Initialize accumulators for predictions and node features
                pressure_pred = torch.zeros(
                    (num_nodes, 1), dtype=torch.float32, device=device
                )
                shear_stress_pred = torch.zeros(
                    (num_nodes, 3), dtype=torch.float32, device=device
                )
                pressure_true = torch.zeros(
                    (num_nodes, 1), dtype=torch.float32, device=device
                )
                shear_stress_true = torch.zeros(
                    (num_nodes, 3), dtype=torch.float32, device=device
                )
                coordinates = torch.zeros(
                    (num_nodes, 3), dtype=torch.float32, device=device
                )
                normals = torch.zeros(
                    (num_nodes, 3), dtype=torch.float32, device=device
                )
                area = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)

                # Accumulate predictions and node features from all partitions
                for j in range(cfg.num_partitions):
                    part = validation_graphs[i][j].to(device)

                    # Get node features (coordinates and normals)
                    ndata = torch.cat(
                        (
                            part.ndata["coordinates"],
                            part.ndata["normals"],
                            torch.sin(2 * np.pi * part.ndata["coordinates"]),
                            torch.cos(2 * np.pi * part.ndata["coordinates"]),
                            torch.sin(4 * np.pi * part.ndata["coordinates"]),
                            torch.cos(4 * np.pi * part.ndata["coordinates"]),
                            torch.sin(8 * np.pi * part.ndata["coordinates"]),
                            torch.cos(8 * np.pi * part.ndata["coordinates"]),
                        ),
                        dim=1,
                    )

                    with torch.no_grad():
                        with torch.autocast(amp_device, enabled=True, dtype=amp_dtype):
                            pred = model(ndata, part.edata["x"], part)
                            pred_filtered = pred[part.ndata["inner_node"].bool()]
                            target = torch.cat(
                                (part.ndata["pressure"], part.ndata["shear_stress"]),
                                dim=1,
                            )
                            target_filtered = target[part.ndata["inner_node"].bool()]
                            loss = (
                                torch.mean((pred_filtered - target_filtered) ** 2)
                                / cfg.num_partitions
                            )
                            valid_loss += loss.item()

                            # Store the predictions based on the original node IDs (using `dgl.NID`)
                            original_nodes = part.ndata[dgl.NID]
                            inner_original_nodes = original_nodes[
                                part.ndata["inner_node"].bool()
                            ]

                            # Accumulate the predictions
                            pressure_pred[inner_original_nodes] = (
                                pred_filtered[:, 0:1].clone().to(torch.float32)
                            )
                            shear_stress_pred[inner_original_nodes] = (
                                pred_filtered[:, 1:].clone().to(torch.float32)
                            )

                            # Accumulate the ground truth
                            pressure_true[inner_original_nodes] = (
                                target_filtered[:, 0:1].clone().to(torch.float32)
                            )
                            shear_stress_true[inner_original_nodes] = (
                                target_filtered[:, 1:].clone().to(torch.float32)
                            )

                            # Accumulate the node features
                            coordinates[original_nodes] = (
                                part.ndata["coordinates"].clone().to(torch.float32)
                            )
                            normals[original_nodes] = (
                                part.ndata["normals"].clone().to(torch.float32)
                            )
                            area[original_nodes] = (
                                part.ndata["area"].clone().to(torch.float32)
                            )

                # Denormalize predictions and node features using the global stats
                pressure_pred_denorm = (
                    pressure_pred.cpu() * torch.tensor(std["pressure"])
                ) + torch.tensor(mean["pressure"])
                shear_stress_pred_denorm = (
                    shear_stress_pred.cpu() * torch.tensor(std["shear_stress"])
                ) + torch.tensor(mean["shear_stress"])
                pressure_true_denorm = (
                    pressure_true.cpu() * torch.tensor(std["pressure"])
                ) + torch.tensor(mean["pressure"])
                shear_stress_true_denorm = (
                    shear_stress_true.cpu() * torch.tensor(std["shear_stress"])
                ) + torch.tensor(mean["shear_stress"])
                coordinates_denorm = (
                    coordinates.cpu() * torch.tensor(std["coordinates"])
                ) + torch.tensor(mean["coordinates"])
                normals_denorm = (
                    normals.cpu() * torch.tensor(std["normals"])
                ) + torch.tensor(mean["normals"])
                area_denorm = (area.cpu() * torch.tensor(std["area"])) + torch.tensor(
                    mean["area"]
                )

                # Save the full point cloud after accumulating all partition predictions
                # Create a PyVista PolyData object for the point cloud
                point_cloud = pv.PolyData(coordinates_denorm.numpy())
                point_cloud["coordinates"] = coordinates_denorm.numpy()
                point_cloud["normals"] = normals_denorm.numpy()
                point_cloud["area"] = area_denorm.numpy()
                point_cloud["pressure_pred"] = pressure_pred_denorm.numpy()
                point_cloud["shear_stress_pred"] = shear_stress_pred_denorm.numpy()
                point_cloud["pressure_true"] = pressure_true_denorm.numpy()
                point_cloud["shear_stress_true"] = shear_stress_true_denorm.numpy()

                # Save the point cloud
                point_cloud.save(f"point_cloud_{validation_ids[i]}.vtp")

            print(
                f"Epoch {epoch+1}, Validation Error: {valid_loss / len(validation_graphs)}"
            )
            writer.add_scalar(
                "validation_loss", valid_loss / len(validation_graphs), epoch
            )

    # Save final checkpoint
    if dist.world_size > 1:
        torch.distributed.barrier()
    if dist.rank == 0:
        save_checkpoint(
            model,
            optimizer,
            scaler,
            scheduler,
            cfg.num_epochs,
            loss.item(),
            "final_model_checkpoint.pth",
        )
        print(f"Average time per batch (exclude first batch): {np.mean(total_time[1:])}")

        print("Training complete")



if __name__ == "__main__":
    main()
