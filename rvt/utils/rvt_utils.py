# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Utility function for Our Agent
"""
import pdb
import argparse
import sys
import signal
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from torch.nn.parallel import DistributedDataParallel as DDP

import rvt.utils.peract_utils as peract_utils
from rvt.models.peract_official import PreprocessAgent2
import torch.distributed.nn
from torch import distributed as dist
import os

def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size

def gather_features(
        image_features,
        text_features,
        rank=0,
        world_size=1
):
    # We gather tensors from all gpus
    gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
    gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
    dist.all_gather(gathered_image_features, image_features)
    dist.all_gather(gathered_text_features, text_features)
    # ensure grads for local rank when all_* features don't have a gradient
    gathered_image_features[rank] = image_features
    gathered_text_features[rank] = text_features
    all_image_features = torch.cat(gathered_image_features, dim=0)
    all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


def get_pc_img_feat(obs, pcd, bounds=None):
    """
    preprocess the data in the peract to our framework
    """
    # obs, pcd = peract_utils._preprocess_inputs(batch)
    bs = obs[0][0].shape[0]
    # concatenating the points from all the cameras
    # (bs, num_points, 3)
    pc = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in pcd], 1)
    _img_feat = [o[0] for o in obs]
    img_dim = _img_feat[0].shape[1]
    # (bs, num_points, 3)
    img_feat = torch.cat(
        [p.permute(0, 2, 3, 1).reshape(bs, -1, img_dim) for p in _img_feat], 1
    )

    img_feat = (img_feat + 1) / 2

    # x_min, y_min, z_min, x_max, y_max, z_max = bounds
    # inv_pnt = (
    #     (pc[:, :, 0] < x_min)
    #     | (pc[:, :, 0] > x_max)
    #     | (pc[:, :, 1] < y_min)
    #     | (pc[:, :, 1] > y_max)
    #     | (pc[:, :, 2] < z_min)
    #     | (pc[:, :, 2] > z_max)
    # )

    # # TODO: move from a list to a better batched version
    # pc = [pc[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    # img_feat = [img_feat[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]

    return pc, img_feat


def move_pc_in_bound(pc, img_feat, bounds, no_op=False):
    """
    :param no_op: no operation
    """
    if no_op:
        return pc, img_feat

    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    inv_pnt = (
        (pc[:, :, 0] < x_min)
        | (pc[:, :, 0] > x_max)
        | (pc[:, :, 1] < y_min)
        | (pc[:, :, 1] > y_max)
        | (pc[:, :, 2] < z_min)
        | (pc[:, :, 2] > z_max)
        | torch.isnan(pc[:, :, 0])
        | torch.isnan(pc[:, :, 1])
        | torch.isnan(pc[:, :, 2])
    )

    # TODO: move from a list to a better batched version
    pc = [pc[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    img_feat = [img_feat[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    return pc, img_feat


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TensorboardManager:
    def __init__(self, path):
        self.writer = SummaryWriter(path)

    def update(self, split, step, vals):
        for k, v in vals.items():
            if "image" in k:
                for i, x in enumerate(v):
                    self.writer.add_image(f"{split}_{step}", x, i)
            elif "hist" in k:
                if isinstance(v, list):
                    self.writer.add_histogram(k, v, step)
                elif isinstance(v, dict):
                    hist_id = {}
                    for i, idx in enumerate(sorted(v.keys())):
                        self.writer.add_histogram(f"{split}_{k}_{step}", v[idx], i)
                        hist_id[i] = idx
                    self.writer.add_text(f"{split}_{k}_{step}_id", f"{hist_id}")
                else:
                    assert False
            else:
                self.writer.add_scalar("%s_%s" % (split, k), v, step)

    def close(self):
        self.writer.flush()
        self.writer.close()


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def short_name(cfg_opts):
    SHORT_FORMS = {
        "peract": "PA",
        "sample_distribution_mode": "SDM",
        "optimizer_type": "OPT",
        "lr_cos_dec": "LCD",
        "num_workers": "NW",
        "True": "T",
        "False": "F",
        "pe_fix": "pf",
        "transform_augmentation_rpy": "tar",
        "lambda_weight_l2": "l2",
        "resume": "RES",
        "inp_pre_pro": "IPP",
        "inp_pre_con": "IPC",
        "cvx_up": "CU",
        "stage_two": "ST",
        "feat_ver": "FV",
        "lamb": "L",
        "img_size": "IS",
        "img_patch_size": "IPS",
        "rlbench": "RLB",
        "move_pc_in_bound": "MPIB",
        "rend": "R",
        "xops": "X",
        "warmup_steps": "WS",
        "epochs": "E",
        "amp": "A",
    }

    if "resume" in cfg_opts:
        cfg_opts = cfg_opts.split(" ")
        res_idx = cfg_opts.index("resume")
        cfg_opts.pop(res_idx + 1)
        cfg_opts = " ".join(cfg_opts)

    cfg_opts = cfg_opts.replace(" ", "_")
    cfg_opts = cfg_opts.replace("/", "_")
    cfg_opts = cfg_opts.replace("[", "")
    cfg_opts = cfg_opts.replace("]", "")
    cfg_opts = cfg_opts.replace("..", "")
    for a, b in SHORT_FORMS.items():
        cfg_opts = cfg_opts.replace(a, b)

    return cfg_opts


def get_num_feat(cfg):
    num_feat = cfg.num_rotation_classes * 3
    # 2 for grip, 2 for collision
    num_feat += 4
    return num_feat


def get_eval_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tasks", type=str, nargs="+", default=["insert_onto_square_peg"]
    )
    parser.add_argument("--model-folder", type=str, default=None)
    parser.add_argument("--eval-datafolder", type=str, default="./data/val/")
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="start to evaluate from which episode",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="how many episodes to be evaluated for each task",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=25,
        help="maximum control steps allowed for each episode",
    )
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--ground-truth", action="store_true", default=False)
    parser.add_argument("--exp_cfg_path", type=str, default=None)
    parser.add_argument("--mvt_cfg_path", type=str, default=None)
    parser.add_argument("--peract_official", action="store_true")
    parser.add_argument(
        "--peract_model_dir",
        type=str,
        default="runs/peract_official/seed0/weights/600000",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log-name", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--use-input-place-with-mean", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--skip", action="store_true")

    return parser


RLBENCH_TASKS = [
    "put_item_in_drawer",
    "reach_and_drag",
    "turn_tap",
    "slide_block_to_color_target",
    "open_drawer",
    "put_groceries_in_cupboard",
    "place_shape_in_shape_sorter",
    "put_money_in_safe",
    "push_buttons",
    "close_jar",
    "stack_blocks",
    "place_cups",
    "place_wine_at_rack_location",
    "light_bulb_in",
    "sweep_to_dustpan_of_size",
    "insert_onto_square_peg",
    "meat_off_grill",
    "stack_cups",
]


OPENVOCAB_TASKS = [
    "put_item_in_drawer",
    "reach_and_drag",
    "turn_tap",
    "slide_block_to_color_target",
    "open_drawer",
    "put_groceries_in_cupboard",
    "place_shape_in_shape_sorter",
    "put_money_in_safe",
    "push_buttons",
    "close_jar",
    "stack_blocks",
    "place_cups",
    "place_wine_at_rack_location",
    "light_bulb_in",
    "sweep_to_dustpan_of_size",
    "insert_onto_square_peg",
    "meat_off_grill",
    "stack_cups",
    "basketball_in_hoop",
    "close_box",
    "close_laptop_lid",
    "empty_dishwasher",  
    "get_ice_from_fridge",     
    "hockey",     
    "meat_on_grill",
    "move_hanger",
    "scoop_with_spatula",  
    "setup_chess",
    "slide_block_to_target",     
    "straighten_rope",     
    "turn_oven_on",     
    "wipe_desk",
]


def load_agent(agent_path, agent=None, only_epoch=False):
    if isinstance(agent, PreprocessAgent2):
        assert not only_epoch
        agent._pose_agent.load_weights(agent_path)
        return 0

    checkpoint = torch.load(agent_path, map_location="cpu")
    epoch = checkpoint["epoch"]

    if not only_epoch:
        if hasattr(agent, "_q"):
            model = agent._q
        elif hasattr(agent, "_network"):
            model = agent._network
        optimizer = agent._optimizer
        lr_sched = agent._lr_sched

        if isinstance(model, DDP):
            model = model.module
                        
        try:
            model.load_state_dict(checkpoint["model_state"])
        except RuntimeError:
            try:
                print(
                    "WARNING: loading states in mvt1. "
                    "Be cautious if you are using a two stage network."
                )
                model.mvt1.load_state_dict(checkpoint["model_state"])
            except RuntimeError:
                print(
                    "WARNING: loading states with strick=False! "
                    "KNOW WHAT YOU ARE DOING!!"
                )
                model.load_state_dict(checkpoint["model_state"], strict=False)

        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        else:
            print(
                "WARNING: No optimizer_state in checkpoint" "KNOW WHAT YOU ARE DOING!!"
            )

        if "lr_sched_state" in checkpoint:
            lr_sched.load_state_dict(checkpoint["lr_sched_state"])
        else:
            print(
                "WARNING: No lr_sched_state in checkpoint" "KNOW WHAT YOU ARE DOING!!"
            )

    return epoch
