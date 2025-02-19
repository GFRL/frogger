import time
import warnings
from concurrent.futures import TimeoutError

import numpy as np
import trimesh
from pydrake.math import RigidTransform, RotationMatrix

from frogger import ROOT
from frogger.baselines import WuBaselineConfig
from frogger.metrics import ferrari_canny_L1, min_weight_metric
from frogger.objects import MeshObject, MeshObjectConfig
from frogger.robots.robot_core import RobotModel
from frogger.robots.robots import (
    AlgrModelConfig,
    BH280ModelConfig,
    FR3AlgrModelConfig,
    FR3AlgrZed2iModelConfig,
)
from frogger.sampling import (
    HeuristicAlgrICSampler,
    HeuristicBH280ICSampler,
    HeuristicFR3AlgrICSampler,
)
from frogger.solvers import Frogger, FroggerConfig
from frogger.utils import timeout

import os
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("--obj_name", type=str, default="core_bottle_1a7ba1f4c892e2da30711cdbdbc73924")
argparser.add_argument("--obj_scale", type=float, default=0.06)
argparser.add_argument("--Trail_id", type=int, default=0)
argparser.add_argument("--save_root", type=str, default="../results")
args=argparser.parse_args()
save_root = args.save_root
# [Feb. 22, 2024] suppress annoying torch warning about LUSolve from qpth
warnings.filterwarnings("ignore", category=UserWarning)

# all example robot models
model_sampler_pairs = [
    # (
    #     "Allegro_WuBaseline",
    #     WuBaselineConfig.from_cfg(AlgrModelConfig),
    #     HeuristicAlgrICSampler,
    # ),
    ("Allegro", AlgrModelConfig, HeuristicAlgrICSampler),
    # ("BH280", BH280ModelConfig, HeuristicBH280ICSampler),
    # ("FR3-Allegro", FR3AlgrModelConfig, HeuristicFR3AlgrICSampler),
    # ("FR3-Allegro-Zed2i", FR3AlgrZed2iModelConfig, HeuristicFR3AlgrICSampler),
]

# all objects from paper
obj_names = [args.obj_name]

tot_setup_time = 0.0
tot_gen_time = 0.0
TIMEOUT_PERIOD_SEC = 60.0  # [EDIT THIS] the number of seconds to wait before timing out
NUM_SAMPLES = 20  # [EDIT THIS] number of grasps to sample per object
EVAL = False  # [EDIT THIS] whether to eval the grasps on min-weight/Ferrari-Canny
VIZ = False  # [EDIT THIS] whether to visualize the results every grasp

if __name__ == "__main__":
    # looping over the example models
    for pair in model_sampler_pairs:
        model_name, ModelConfig, Sampler = pair
        translation = 0.7 if model_name in ["FR3-Allegro", "FR3-Allegro-Zed2i"] else 0.0
        print(f"model: {model_name}")

        # looping over all objects
        for obj_name in obj_names:
            if model_name == "FR3-Allegro" and obj_name in [
                "048_hammer",
                "051_large_clamp",
                "052_extra_large_clamp",
            ]:
                # our heuristic is very bad for flat objects when they lie on a table
                continue

            print(f"  {obj_name}")
            start = time.time()

            # loading object
            mesh = trimesh.load(f"../assets/DGNObj/{obj_name}/mesh/simplified.obj")
            mesh.apply_scale(args.obj_scale)
            bounds = mesh.bounds
            lb_O = bounds[0, :]
            ub_O = bounds[1, :]
            X_WO = RigidTransform(
                RotationMatrix(),
                np.array([translation, 0.0, -lb_O[-1]]),
            )
            obj = MeshObjectConfig(
                X_WO=X_WO,
                mesh=mesh,
                name=obj_name,
                enforce_watertight=True,  # you can turn off the check for watertightness
                clean=False,
            ).create()

            # example showing how to use the custom collision callback
            if "Allegro" in model_name:

                def _custom_coll_callback(
                    model: RobotModel, name_A: str, name_B: str
                ) -> float:
                    """A custom collision callback.

                    Given two collision geom names, indicates what the lower bound on separation
                    should be between them in meters.

                    WARNING: for now, if you overwrite this, you MUST ensure manually that the fingertips
                    are allowed some penetration with the object!
                    """
                    # organizing names
                    has_tip = (
                        "FROGGERCOL" in name_A or "FROGGERCOL" in name_B
                    )  # MUST MANUALLY DO THIS!
                    has_palm = "palm" in name_A or "palm" in name_B
                    has_ds = (
                        "ds_collision" in name_A or "ds_collision" in name_B
                    )  # non-tip distal geoms
                    has_md = "md" in name_A or "md" in name_B  # medial geoms
                    has_px = "px" in name_A or "px" in name_B  # proximal geoms
                    has_bs = "bs" in name_A or "bs" in name_B  # base geoms
                    has_mp = (
                        "mp" in name_A or "mp" in name_B
                    )  # metacarpal geoms, thumb only
                    has_obj = "obj" in name_A or "obj" in name_B

                    # provide custom bounds on different geom pairs
                    if has_tip and has_obj:
                        # allow tips to penetrate object - MUST MANUALLY DO THIS!
                        return -model.d_pen
                    elif has_palm and has_obj:
                        return 0.01  # ensure at least 1cm separation
                    elif has_ds and has_obj:
                        return 0.002  # ensure at least 2mm separation
                    elif has_md and has_obj:  # noqa: SIM114
                        return 0.005  # ensure at least 5mm separation
                    elif has_px and has_obj:  # noqa: SIM114
                        return 0.005  # ensure at least 5mm separation
                    elif has_bs and has_obj:  # noqa: SIM114
                        return 0.01  # ensure at least 1cm separation
                    elif has_mp and has_obj:  # noqa: SIM114
                        return 0.01  # ensure at least 1cm separation
                    else:
                        return model.d_min  # default case: use d_min

                custom_coll_callback = _custom_coll_callback
            else:
                custom_coll_callback = None
            custom_coll_callback = None

            # loading model and sampler
            model = ModelConfig(
                obj=obj,
                ns=4,
                mu=0.7,
                d_min=0.001,
                d_pen=0.005,
                l_bar_cutoff=0.3,
                ignore_mass_inertia=True,  # don't use obj mass/inertia at all
                viz=VIZ,
                custom_coll_callback=custom_coll_callback,
            ).create()
            sampler = Sampler(
                model,
                z_axis_fwd=model_name in ["FR3-Allegro", "FR3-Allegro-Zed2i"],
            )

            # loading grasp generator
            frogger = FroggerConfig(
                model=model,
                sampler=sampler,
                tol_surf=1e-3,
                tol_joint=1e-2,
                tol_col=1e-3,
                tol_fclosure=1e-5,
                xtol_rel=1e-6,
                xtol_abs=1e-6,
                maxeval=1000,
            ).create()
            end = time.time()
            print(f"    setup time: {end - start}")
            tot_setup_time += end - start

            # timing test
            sub_time = 0.0
            save_dir=os.path.join(save_root,obj_name,f"scale{int(100*args.obj_scale):03d}")
            os.makedirs(save_dir,exist_ok=True)
            for id in range(NUM_SAMPLES):
                start = time.time()
                try:
                    q_star = timeout(TIMEOUT_PERIOD_SEC)(
                        frogger.generate_grasp
                    )()  # only time generation
                except TimeoutError:
                    print(
                        f"        Grasp generation timed out after {TIMEOUT_PERIOD_SEC} seconds!"
                    )
                    continue
                end = time.time()
                sub_time += end - start

                # evaluate the grasps if requested
                if EVAL:
                    print(f"        min-weight: {min_weight_metric(model, q_star)}")
                    print(f"        Ferrari-Canny: {ferrari_canny_L1(model, q_star)}")

                # visualize the grasp if requested
                if VIZ:
                    model.viz_config(q_star)
                
                save_path=os.path.join(save_dir,f"{id}.npy")
                q_star=np.array(q_star).reshape(-1)#(4+3+qpos)
                save_dict = {
                    "obj_scale": args.obj_scale,
                    "obj_pose": np.array([translation, 0.0, -lb_O[-1],1.0,0.0,0.0,0.0]),
                    'obj_path': os.path.join("assets/DGNObj",obj_name),
                    "grasp_qpos": np.concatenate([q_star[4:7],q_star[0:4],q_star[7:]]),
                    "grasp_error": 0.0,
                }
                np.save(save_path,save_dict,allow_pickle=True)

            print(f"    grasp generation time: {end - start}")
            tot_gen_time += sub_time

        # computing total times
        avg_setup_time = tot_setup_time / (len(obj_names) * NUM_SAMPLES)
        avg_synthesis_time = tot_gen_time / (len(obj_names) * NUM_SAMPLES)
        print("  Finished!")
        print(f"  Average setup time: {avg_setup_time}")
        print(f"  Average synthesis time: {avg_synthesis_time}")
        print("---------------------------------------------------------------------------")
