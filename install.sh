#!/bin/bash
echo "Do you wish to install this program? (Make sure conda is installed before installation)"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) break;;
        No ) exit;;
    esac
done


git clone --recurse-submodules https://github.com/gerardPlanella/multiagent_fairness_reproducibility.git
echo "----- Entering Repo -----"
cd multiagent_fairness_reproducibility
echo "----- Creating Conda Environment -----"
conda create -n "fact22" python=3.5.4
echo "----- Activating Conda Environment -----"
conda activate fact22
echo "----- Installing PyTorch -----"
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org torch===1.5.0 torchvision===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
echo "----- Installing Remaining Libraries -----"
pip install -r requirements_v2.txt

if [[ $OSTYPE == 'darwin'* ]]; then
  echo '----- macOS detected, installing specific libraries -----'
  pip install pyglet==1.5.11
fi

cd simple_particle_envs
echo "----- Installing Simple Particle Environment -----"
pip install -e .
cd ..
echo "----- Deactivating conda environment -----"
conda deactivate
echo "-----Installation finished -----"