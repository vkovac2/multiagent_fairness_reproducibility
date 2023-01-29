# Fairness for Cooperative Multi-Agent Learning with Equivariant Policies


This repository is the official implementation of Fairness for Cooperative Multi-Agent Learning with Equivariant Policies Reproducibility STudy.

## Cloning the repository

To clone the repository with the simple_particle_envs submodule:
```
git clone --recurse-submodules https://github.com/gerardPlanella/multiagent_fairness_reproducibility.git
```

## Setup
Installing the Recommended Anaconda environment (Python 3.9.15)

Windows:
```
conda env create -f environment_windows.yml
```
Linux:
```
conda env create -f environment_linux.yml
```

To setup multi-agent environments:
```
cd simple_particle_envs
pip install -e .
```

To verify installation, run:
```
xvfb-run -a python baselines/baselines.py --mode test --render
```

## Training

To train a Fair-E model, run:

```train
python main.py --env simple_torus --algorithm ddpg_symmetric
```

To train a Fair-E model with equivariance and shared reward, run:

```train
python main.py --env simple_torus --algorithm ddpg_symmetric --equivariant --collaborative
```

To train a Fair-ER model, run:

``` train
python main.py --env simple_torus --algorithm ddpg_speed_fair
```
* The control parameter of fairness can be adjusted in _configs.py_.

To resume training from a checkpoint, run:
```
python main.py --env simple_torus --algorithm ddpg_symmetric --checkpoint_path /path/to/model/checkpoints
```

## Evaluation

To collect trajectories from a trained model, run _eval/collect_actions.py_. Here are a few examples:
* Greedy pursuers against random-moving evader: 
```eval
python eval/collect_actions.py --env simple_torus --pred_policy greedy --prey_policy random --seed 75 
```
* CD-DDPG pursuers against sophisticated evader: 
```eval
python eval/collect_actions.py --env simple_torus --pred_policy ddpg --prey_policy cosine --seed 72 --checkpoint_path /path/to/model/checkpoints
```

To create fairness vs. utility plots, run:
```eval
python eval/fairness_vs_utility.py --fp /path/to/folder/of/results
```


<!-- 0.3: results/ddpg_speed_fair_simple_torus/exp_01_22_2023__14_03_55 -->

<!-- symm: results/ddpg_symmetric_simple_torus/exp_01_24_2023__21_06_09 -->