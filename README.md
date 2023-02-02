# Fairness for Cooperative Multi-Agent Learning with Equivariant Policies Reproducibility Study

This repository is the official implementation of Fairness for Cooperative Multi-Agent Learning with Equivariant Policies Reproducibility Study.

## Cloning the repository

To clone the repository with the simple_particle_envs submodule:
```
git clone --recurse-submodules <link_to_repository>
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
python main.py --env simple_torus --algorithm ddpg_speed_fair --lambda_coeff 0.5
```
* The control parameter of fairness can be adjusted in _configs.py_.

To resume training from a checkpoint, run:
```
python main.py --env simple_torus --algorithm ddpg_symmetric --checkpoint_path /path/to/model/checkpoints
```
To train with a varying number of evaders and pursuers we use the simple_torus.py scenario:
```
python main.py --env simple_torus --algorithm ddpg_symmetric --nb_agents 5 --nb_prey 1
```


## Evaluation

It is important to set the same flags that you used for training, so if you have used the **equivariant** and **collaborative** flag, you should also set them when running the evaluation.

To collect trajectories from a trained model, run _eval/collect_actions.py_ or _eval/collect_actions_symmetric.py_. Here are a few examples:
* Greedy pursuers against random-moving evader: 
```eval
python eval/collect_actions.py --env simple_torus --pred_policy greedy --prey_policy random --seed 75 
```
* CD-DDPG pursuers (Fair-E) against sophisticated evader: 
```eval
python eval/collect_actions_symmetric.py --env simple_torus --pred_policy ddpg --prey_policy cosine --seed 72 --checkpoint_path /path/to/model/checkpoints
```
```
* CD-DDPG pursuers (Fair-ER) against sophisticated evader: 
```eval
python eval/collect_actions.py --env simple_torus --pred_policy ddpg --prey_policy cosine --seed 72 --checkpoint_path /path/to/model/checkpoints
```

To collect trajectories trained with a varying number of evaders and pursuers we use the simple_torus scenario again. For example, with a Fair-E model:
```
python eval/collect_actions_symmetric.py --env simple_torus --pred_policy ddpg --prey_policy cosine --seed 72 --checkpoint_path /path/to/model/checkpoints --nb_agents 5 --nb_prey 1
```

To create the plots, run:
```eval
python eval/make_plots.py --fp path/of/trajectories --plot (1-5)
```

```
for 4 agents:
python eval/make_plots_4_predators.py --fp path/of/trajectories --plot (1-3)
```
