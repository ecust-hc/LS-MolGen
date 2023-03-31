#!/usr/bin/env python
import torch
import numpy as np
import argparse
import warnings
import time
from rdkit import rdBase
import os
import re
import shutil
from shutil import copyfile
from model import RNN
import data_structs as ds
from data_structs import Vocabulary, Experience
import utils
from utils import Variable, seq_to_smiles, fraction_valid_smiles, unique
import linecache
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED

rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")

def train_agent(restore_prior_from='data/Transfer.ckpt',
                restore_agent_from='data/Transfer.ckpt',
                save_dir='data/Agent.ckpt', voc_file='data/Voc',
                learning_rate=0.0005,
                batch_size=128, n_steps=100,
                sigma=20, experience_replay=0):

    voc = Vocabulary(init_from_file=voc_file)

    start_time = time.time()

    Prior = RNN(voc)
    Agent = RNN(voc)


    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(restore_prior_from))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from))
    else:
        Prior.rnn.load_state_dict(torch.load(restore_prior_from, map_location=lambda storage, loc: storage))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

    # We dont need gradients with respect to Prior
    for param in Prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=0.0005)

    # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
    # occur more often (which means the agent can get biased towards them). Using experience replay is
    # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
    experience = Experience(voc)

    # Information for the logger
    step_score = [[], []]

    print("Model initialized, starting training...")

    for step in range(n_steps):

        # Sample from Agent
        seqs, agent_likelihood, entropy = Agent.sample(batch_size)

        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        # Get prior likelihood and score
        prior_likelihood, _ = Prior.likelihood(Variable(seqs))
        smiles = seq_to_smiles(seqs, voc)
        score = docking_score(smiles)

        # Calculate augmented likelihood
        augmented_likelihood = prior_likelihood + sigma * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Experience Replay
        # First sample
        if experience_replay and len(experience)>4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(4)
            exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        # Then add new experience
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, score, prior_likelihood)
        experience.add_experience(new_experience)

        # Calculate loss
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        loss_p = - (1 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p

        # Calculate gradients and make an update to the network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()

        # Print some information for this step
        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        print("\n       Step {}   Fraction valid SMILES: {:4.1f}  Time elapsed: {:.2f}h Time left: {:.2f}h".format(
                step, fraction_valid_smiles(smiles) * 100, time_elapsed, time_left))
        print("  Agent    Prior   Target   Score             SMILES")
        for i in range(10):
            print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}     {}".format(agent_likelihood[i],
                                                                           prior_likelihood[i],
                                                                           augmented_likelihood[i],
                                                                           score[i],
                                                                           smiles[i]))
            
        # Need this for Vizard plotting
        step_score[0].append(step + 1)
        step_score[1].append(np.mean(score))

    # If the entire training finishes, we create a new folder where we save this python file
    # as well as some sampled sequences and the contents of the experinence (which are the highest
    # scored sequences seen during training)
    if not save_dir:
        save_dir = 'data/results/run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    os.makedirs(save_dir)
    copyfile('reinforcement_learning.py', os.path.join(save_dir, "reinforcement_learning.py"))

    experience.print_memory(os.path.join(save_dir, "memory.smi"))
    torch.save(Agent.rnn.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))
    torch.save(Agent.rnn.state_dict(), 'data/Agent.ckpt')

    seqs, agent_likelihood, entropy = Agent.sample(256)
    prior_likelihood, _ = Prior.likelihood(Variable(seqs))
    prior_likelihood = prior_likelihood.data.cpu().numpy()
    smiles = seq_to_smiles(seqs, voc)
    score = docking_score(smiles)
    with open(os.path.join(save_dir, "sampled"), 'w') as f:
        f.write("SMILES Score PriorLogP\n")
        for smiles, score, prior_likelihood in zip(smiles, score, prior_likelihood):
            f.write("{} {:5.2f} {:6.2f}\n".format(smiles, score, prior_likelihood))


def scoring_function(smiles_list, work_dir='./ledock', k=-10):
    """Docking scores based on Ledock"""
    main_dir = os.path.abspath(os.path.dirname(__file__))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        shutil.copy(os.path.join(main_dir, "ledock/pro.pdb"), work_dir)
        shutil.copy(os.path.join(main_dir, "ledock/dock.in"), work_dir)

    os.chdir(work_dir)
    valid_smiles = [smi for smi in smiles_list if utils.valid_smiles(smi)]
    valid_id = [i for i, smi in enumerate(smiles_list) if utils.valid_smiles(smi)]
    ds.write_smiles_to_file(valid_smiles, './lig.smi')
    os.system('obabel ./lig.smi -omol2 -O ./lig.mol2 --gen3D -m')
    # mol_list = [filename for filename in os.listdir('.') if filename.endswith('mol2')]
    mol_list = ['./lig{}.mol2'.format(i + 1) for i in range(len(valid_smiles))]
    ds.write_smiles_to_file(mol_list, './ligands')
    os.system('ledock ./dock.in')
    score_list = []
    j = 0
    for i in range(len(smiles_list)):
        if i not in valid_id:
            score_list.append(0.0)
            continue
        line_docking_score = linecache.getline('./lig{}.dok'.format(j + 1), 2)
        j = j + 1
        groups = re.search('Score: (.+) kcal/mol', line_docking_score)
        if not groups:
            score_list.append(0.0)
        else:
            dock_score = float(groups.group(1))
            score = max(dock_score, k) / k
            score_list.append(score)
    os.system("rm -f *.mol2 *.dok")
    os.chdir(main_dir)
    return score_list


def docking_score(smiles, num_processes=96):
    scores = []
    smiles_list = np.array_split(smiles, num_processes)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = [executor.submit(scoring_function, smile, "./ledock_{}".format(j)) for j, smile in
                 enumerate(smiles_list)]
        wait(tasks, return_when=ALL_COMPLETED)

        for res in tasks:
            scores.extend(res.result())
    #print(scores)
    return np.array(scores, dtype=np.float32)


if __name__ == "__main__":
    s = time.time()
    train_agent(save_dir='data/result',
                learning_rate=0.0005,
                batch_size=96, n_steps=5,
                sigma=20, experience_replay=0)
    e = time.time()
    print("Use time: {:.4f}s".format(e - s))
