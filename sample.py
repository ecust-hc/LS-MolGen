#!/usr/bin/env python
import torch
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys
from data_structs import MolData, Vocabulary
from model import RNN
import os
import sys
import argparse
import opts
import filter_function as ff

def Sample(opt):
    voc = Vocabulary(init_from_file="data/Voc")  
    Prior = RNN(voc)
    print(opt.model, opt.num)
    # Can restore from a saved RNN
    Prior.rnn.load_state_dict(torch.load(opt.model))
    totalsmiles = set()
    enumerate_number = int(opt.num)
    molecules_total = 0
    for epoch in range(1, opt.epoch):
        seqs, likelihood, _ = Prior.saple(opt.batch_size)
        valid = 0
        for i, seq in enumerate(seqs.cpu().numpy()):
            smile = voc.decode(seq)
            if ff.valid_mol(smile) and ff.PCPF(smile, opt) and ff.MCPF(smile, opt) and ff.StructFliter(smile, opt):
                valid += 1
                totalsmiles.add(smile)
                       
        molecules_total = len(totalsmiles)
        print(("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs))))
        print(valid, molecules_total, epoch)
        if molecules_total > enumerate_number:
            break
    return totalsmiles

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sample.py')
    opts.sample_opts(parser)
    opts.filter_opts(parser)
    opt = parser.parse_args()
    print(opt.model)
    totalsmiles=Sample(opt)
    #f = open('./result/sample_' + os.path.splitext(os.path.split(sys.argv[1])[1])[0] + '_' + str(n) + '.smi', 'w')  
    f = open(opt.save, 'w')
    for smile in totalsmiles:
        f.write(smile + "\n")
    f.close()
    print('Sampling completed')
