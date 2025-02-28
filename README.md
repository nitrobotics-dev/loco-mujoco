<p align="center">
  <img width="70%" src="https://github.com/robfiras/loco-mujoco/assets/69359729/bd2a219e-ddfd-4355-8024-d9af921fb92a">
</p>

![continous integration](https://github.com/robfiras/loco-mujoco/actions/workflows/continuous_integration.yml/badge.svg?branch=dev)
[![Documentation Status](https://readthedocs.org/projects/loco-mujoco/badge/?version=latest)](https://loco-mujoco.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/loco-mujoco)](https://pypi.org/project/loco-mujoco/)
[![Join our Discord](https://img.shields.io/badge/Discord-Join%20Us-7289DA?style=flat&logo=discord&logoColor=white)](https://discord.gg/gEqR3xCVdn)


TODO: INTRO WILL BE UPDATED FOR RELEASE, BUT INSTALLATION IS CORRECT.

**LocoMuJoCo** is an **imitation learning benchmark** specifically targeted towards **locomotion**. It encompasses a diverse set of environments, including quadrupeds, bipeds, and musculoskeletal human models, each accompanied by comprehensive datasets, such as real noisy motion capture data, ground truth expert data, and ground truth sub-optimal data,
enabling evaluation across a spectrum of difficulty levels. 

**LocoMuJoCo** also allows you to specify your own reward function to use this benchmark for **pure reinforcement learning**! Checkout the example below!

<p align="center">
  <img src="https://github.com/robfiras/loco-mujoco/assets/69359729/c16dfa4a-4fdb-4701-9a42-54cbf7644301">
</p>

### Key Advantages 
✅ Easy to use with [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) or [Mushroom-RL](https://github.com/MushroomRL/mushroom-rl) interface \
✅ Many environments including humanoids and quadrupeds \
✅ Diverse set of datasets --> e.g., noisy motion capture or ground truth datasets with actions \
✅ Wide spectrum of difficulty levels \
✅ Built-in domain randomization \
✅ Many baseline algorithms for quick benchmarking \
✅ [Documentation](https://loco-mujoco.readthedocs.io/)

---

## Installation

[//]: # ()
[//]: # (You have the choice to install the latest release via PyPI by running )

[//]: # ()
[//]: # (```bash)

[//]: # (pip install loco-mujoco )

[//]: # (```)

Clone this repo and do an editable installation:

```bash
cd loco-mujoco
pip install -e . 
```

By default, this will install the CPU-version of Jax. If you want to use Jax on the GPU, you need to install the following:

```bash
pip install jax["cuda12"]
````

> [!NOTE]
> If you want to run the **MyoSkeleton** environment, you need to additionally run
> `loco-mujoco-myomodel-init` to accept the license and download the model.


### Datasets

LocoMuJoCo provides three sources of motion capture (mocap) data: default (provided by us), LAFAN1, and AMASS. The first two datasets
are available on the [LocoMujoCo HuggingFace dataset repository](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets)
and will downloaded and cached automatically for you. AMASS needs to be downloaded and installed separately due to
their licensing. See [here](loco_mujoco/smpl) for more information about the installation.

This is how you can visualize the datasets:

```python
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf, AMASSDatasetConf


# # example --> you can add as many datasets as you want in the lists!
env = ImitationFactory.make("UnitreeH1",
                            default_dataset_conf=DefaultDatasetConf(["squat"]),
                            lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4", "walk1_subject1"]),
                            # if SMPL and AMASS are installed, you can use the following:
                            #amass_dataset_conf=AMASSDatasetConf(["DanceDB/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses"])
                            )

env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)
```

#### Speeding up Dataset Loading
LocoMuJoCo only stores datasets with joint positions and velocities to save memory. All other attributes are calculated 
using forward kinematics upon loading. If you want to speed up the dataset loading, you can define caches for the datasets. This will
store the forward kinematics results in a cache file, which will be loaded on the next run: 

```bash
loco-mujoco-set-all-caches --path <path to cache>
```

For instance, you could run:
```bash
loco-mujoco-set-all-caches --path "$HOME/.loco-mujoco-caches"
````

---
## Citation
```
@inproceedings{alhafez2023b,
title={LocoMuJoCo: A Comprehensive Imitation Learning Benchmark for Locomotion},
author={Firas Al-Hafez and Guoping Zhao and Jan Peters and Davide Tateo},
booktitle={6th Robot Learning Workshop, NeurIPS},
year={2023}
}
```

---
## Credits 
Both Unitree models were taken from the [MuJoCo menagerie](https://github.com/google-deepmind/mujoco_menagerie)



