# LS-MolGen
LS-MolGen: Ligand-and-structure Dual-driven Deep Reinforcement Learning for Target-specific Molecular Generation Improves Binding Affinity and Novelty
# Installation
You can use the environment.yml file to create a new conda environment with all the necessary dependencies for LS-MolGen:
```
git clone https://github.com/songleee/LS-MolGen.git
cd LS-MolGen
conda env create -f environment.yml
conda activate LS-MolGen
```
# Usage
```
python pre_train.py

python transfer_learning.py

python reinforcement_learning.py
```
