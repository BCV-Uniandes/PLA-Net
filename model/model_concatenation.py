import torch
import torch.nn as nn
import torch.nn.functional as F

from gcn_lib.sparse.torch_nn import MLP

from model.model import DeeperGCN

import numpy as np
import logging


class PLANet(torch.nn.Module):
    def __init__(self, args,saliency=False):
        super(PLANet, self).__init__() 

        # Args
        self.args = args
        # Molecule and protein networks
        self.molecule_gcn = DeeperGCN(args)
        self.target_gcn = DeeperGCN(args, is_prot=True)

        # Individual modules' final embbeding size
        output_molecule = args.hidden_channels
        output_protein = args.hidden_channels_prot
        # Concatenated embbeding size
        Final_output = output_molecule + output_protein
        # Overall model's final embbeding size
        hidden_channels = args.hidden_channels

        # Multiplier
        if args.multi_concat:
            self.multiplier_prot = torch.nn.Parameter(torch.zeros(hidden_channels))
            self.multiplier_ligand = torch.nn.Parameter(torch.ones(hidden_channels))
        elif self.args.MLP:
            # MLP
            hidden_channel = 64
            channels_concat = [256, hidden_channel, hidden_channel, 128]
            self.concatenation_gcn = MLP(channels_concat, norm=args.norm, last_lin=True)
            # breakpoint()
            indices = np.diag_indices(hidden_channel)
            tensor_linear_layer = torch.zeros(hidden_channel, Final_output)
            tensor_linear_layer[indices[0], indices[1]] = 1
            self.concatenation_gcn[0].weight = torch.nn.Parameter(tensor_linear_layer)
            self.concatenation_gcn[0].bias = torch.nn.Parameter(
                torch.zeros(hidden_channel)
            )
        else:
            # Concatenation Layer
            self.concatenation_gcn = nn.Linear(Final_output, hidden_channels)
            indices = np.diag_indices(output_molecule)
            tensor_linear_layer = torch.zeros(hidden_channels, Final_output)
            tensor_linear_layer[indices[0], indices[1]] = 1
            self.concatenation_gcn.weight = torch.nn.Parameter(tensor_linear_layer)
            self.concatenation_gcn.bias = torch.nn.Parameter(
                torch.zeros(hidden_channels)
            )

        # Classification Layer
        num_classes = args.nclasses
        self.classification = nn.Linear(hidden_channels, num_classes)

    def forward(self, molecule, target):

        molecule_features = self.molecule_gcn(molecule)
        target_features = self.target_gcn(target)
        # Multiplier
        if self.args.multi_concat:
            All_features = (
                target_features * self.multiplier_prot
                + molecule_features * self.multiplier_ligand
            )
        else:
            # Concatenation of LM and PM modules
            All_features = torch.cat((molecule_features, target_features), dim=1)
            All_features = self.concatenation_gcn(All_features)
        # Classification
        classification = self.classification(All_features)

        return classification

    def print_params(self, epoch=None, final=False):

        logging.info("======= Molecule GCN ========")
        self.molecule_gcn.print_params(epoch)
        logging.info("======= Protein GCN ========")
        self.target_gcn.print_params(epoch)
        if self.args.multi_concat:
            sum_prot_multi = sum(self.multiplier_prot)
            sum_lig_multi = sum(self.multiplier_ligand)
            logging.info("Sumed prot multi: {}".format(sum_prot_multi))
            logging.info("Sumed lig multi: {}".format(sum_lig_multi))
