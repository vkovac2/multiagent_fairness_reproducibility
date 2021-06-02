# Curriculum-Driven Multi-Agent Learning and the Role of Implicit Communication in Teamwork

<!-- This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345).  -->
This repository is the official implementation of Curriculum-Driven Multi-Agent Learning and the Role of Implicit Communication in Teamwork.

## Setup
Python 3.5.4 (anaconda environment recommended)

To install requirements:
```
pip install -r requirements.txt
```
To setup multi-agent environments:
```
cd simple_particle_envs
pip install -e .
```

To verify installation, run:
```
python baselines/baselines.py --mode test --render
```

## Training

To train a CD-DDPG model, run:

```train
python main.py --env simple_torus --algorithm ddpg_speed
```

To train a CD-DDPG (partial) model, run with the _simple_torus_blind_ environment:

```train
python main.py --env simple_torus_blind --algorithm ddpg_speed
```

To resume training from a checkpoint, run: 
```
python main.py --env simple_torus --algorithm ddpg_speed --checkpoint_path /path/to/model/checkpoints
```

## Evaluation

To collect trajectories from a trained model, run _eval/collect_actions.py_. Here are a few examples:
* Greedy pursuers against random-moving evader: 
```eval
python eval/collect_actions.py --env simple_torus --pred_policy greedy --prey_policy random --seed 75 
```
* CD-DDPG pursuers against sophisticated evader: 
```eval
python eval/collect_actions.py --env simple_torus --pred_policy ddpg --prey_policy cosine --seed 75 --checkpoint_path /path/to/model/checkpoints
```




