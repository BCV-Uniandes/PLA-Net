
import pandas as pd
import shutil, os
import os.path as osp
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from torch.autograd import Variable

from rdkit import Chem

from data.features import (
    allowable_features,
    atom_to_feature_vector,
    bond_to_feature_vector,
    atom_feature_vector_to_dict,
    bond_feature_vector_to_dict,
)

from utils.data_util import one_hot_vector_sm, one_hot_vector_am, get_atom_feature_dims


def load_dataset(
    cross_val, binary_task, target, args, use_prot=False, advs=False, test=False
):
    """
    Load data and return data in dataframes format for each split and the loader of each split.
    Args:
        cross_val (int): Data partition being used [1-4].
        binary_tast (boolean): Whether to perform binary classification or multiclass classification.
        target (string): Name of the protein target for binary classification.
        args (parser): Complete arguments (configuration) of the model.
        use_prot (boolean): Whether to use the PM module.
        advs (boolean): Whether to train the LM module with adversarial augmentations.
        test (boolean): Whether the model is being tested or trained.
    Return:
        train (loader): Training loader
        valid (loader): Validation loader
        test (loader): Test loader
        data_train (dataframe): Training data dataframe
        data_valid (dataframe): Validation data dataframe
        data_test (dataframe): Test data dataframe

    """
    # TODO: NO QUEREMOS QUE ESTÉ LA PARTICIÓN DEL MULTICLASE?
    # Read all data files
    if not test:
        # Verify cross validation partition is defined
        assert cross_val in [1, 2, 3, 4], "{} data partition is not defined".format(
            cross_val
        )
        print("Loading data...")
        if binary_task:
            path = "data/datasets/AD/"
            A = pd.read_csv(
                path + "Smiles_AD_1.csv", names=["Smiles", "Target", "Label"]
            )
            B = pd.read_csv(
                path + "Smiles_AD_2.csv", names=["Smiles", "Target", "Label"]
            )
            C = pd.read_csv(
                path + "Smiles_AD_3.csv", names=["Smiles", "Target", "Label"]
            )
            D = pd.read_csv(
                path + "Smiles_AD_4.csv", names=["Smiles", "Target", "Label"]
            )
            data_test = pd.read_csv(
                path + "AD_Test.csv", names=["Smiles", "Target", "Label"]
            )
            if use_prot:
                data_target = pd.read_csv(
                    path + "Targets_Fasta.csv", names=["Fasta", "Target", "Label"]
                )
            else:
                data_target = []
        # Generate train and validation splits according to cross validation number
        if cross_val == 1:
            data_train = pd.concat([A, B, C], ignore_index=True)
            data_val = D
        elif cross_val == 2:
            data_train = pd.concat([A, C, D], ignore_index=True)
            data_val = B
        elif cross_val == 3:
            data_train = pd.concat([A, B, D], ignore_index=True)
            data_val = C
        elif cross_val == 4:
            data_train = pd.concat([B, C, D], ignore_index=True)
            data_val = A
        # If in binary classification select data for the specific target being train
        if binary_task:
            data_train = data_train[data_train.Target == target]
            data_val = data_val[data_val.Target == target]
            data_test = data_test[data_test.Target == target]
            if use_prot:
                data_target = data_target[data_target.Target == target]
        # Get dataset for each split
        train = get_dataset(data_train, use_prot, data_target, args, advs)
        valid = get_dataset(data_val, use_prot, data_target, args)
        test = get_dataset(data_test, use_prot, data_target, args)
    else:
        # Read test data file
        if binary_task:
            path = "data/datasets/AD/"
            data_test = pd.read_csv(
                path + "Smiles_AD_Test.csv", names=["Smiles", "Target", "Label"]
            )
            data_test = data_test[data_test.Target == target]
            if use_prot:
                data_target = pd.read_csv(
                    path + "Targets_Fasta.csv", names=["Fasta", "Target", "Label"]
                )
                data_target = data_target[data_target.Target == target]
            else:
                data_target = []
        test = get_dataset(data_test,target=data_target, use_prot=use_prot, args=args, advs=advs, saliency=args.saliency)
        train = []
        valid = []
        data_train = []
        data_val = []
    print("Done.")
    return train, valid, test, data_train, data_val, data_test


def reload_dataset(cross_val, binary_task, target, args, advs=False):
    print("Reloading data")
    args.edge_dict = {}
    if binary_task:
        path = "data/datasets/AD/"
        A = pd.read_csv(path + "Smiles_AD_1.csv", names=["Smiles", "Target", "Label"])
        B = pd.read_csv(path + "Smiles_AD_2.csv", names=["Smiles", "Target", "Label"])
        C = pd.read_csv(path + "Smiles_AD_3.csv", names=["Smiles", "Target", "Label"])
        D = pd.read_csv(path + "Smiles_AD_4.csv", names=["Smiles", "Target", "Label"])
        data_test = pd.read_csv(
            path + "AD_Test.csv", names=["Smiles", "Target", "Label"]
        )

    if cross_val == 1:
        data_train = pd.concat([A, B, C], ignore_index=True)
    elif cross_val == 2:
        data_train = pd.concat([A, C, D], ignore_index=True)
    elif cross_val == 3:
        data_train = pd.concat([A, B, D], ignore_index=True)
    else:
        data_train = pd.concat([B, C, D], ignore_index=True)

    if binary_task:
        data_train = data_train[data_train.Target == target]

    train = get_dataset(data_train, args=args, advs=advs)
    print("Done.")

    return train, data_train


def smiles_to_graph(smiles_string, is_prot=False, received_mol=False, saliency=False):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    if not is_prot:
        mol = Chem.MolFromSmiles(smiles_string)
    else:
        mol = Chem.MolFromFASTA(smiles_string)
    # atoms
    atom_features_list = []
    atom_feat_dims = get_atom_feature_dims()
    for atom in mol.GetAtoms():
        ftrs = atom_to_feature_vector(atom)
        if saliency:
            ftrs_oh = one_hot_vector_am(ftrs, atom_feat_dims)
            atom_features_list.append(torch.unsqueeze(ftrs_oh, 0))
        else:
            atom_features_list.append(ftrs)

    if saliency:
        x = torch.cat(atom_features_list)
    else:
        x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    return edge_attr, edge_index, x


def smiles_to_graph_advs(
    smiles_string, args, advs=False, received_mol=False, saliency=False
):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    if not received_mol:
        mol = Chem.MolFromSmiles(smiles_string)
    else:
        mol = smiles_string

    # atoms
    atom_features_list = []
    atom_feat_dims = get_atom_feature_dims()

    for atom in mol.GetAtoms():
        ftrs = atom_to_feature_vector(atom)
        if saliency:
            ftrs_oh = one_hot_vector_am(ftrs, atom_feat_dims)
            atom_features_list.append(torch.unsqueeze(ftrs_oh, 0))
        else:
            atom_features_list.append(ftrs)

    if saliency:
        x = torch.cat(atom_features_list)
    else:
        x = np.array(atom_features_list, dtype=np.int64)

    if advs:
        # bonds
        mol_edge_dict = {}

    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    features_dim1 = torch.eye(5)
    features_dim2 = torch.eye(6)
    features_dim3 = torch.eye(2)
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edges_list.append((j, i))

            edge_feature_oh = one_hot_vector_sm(
                edge_feature, features_dim1, features_dim2, features_dim3
            )
            if advs:
                mol_edge_dict[(i, j)] = Variable(
                    torch.tensor([1.0]), requires_grad=True
                )

                # add edges in both directions
                edge_features_list.append(
                    torch.unsqueeze(mol_edge_dict[(i, j)] * edge_feature_oh, 0)
                )
                edge_features_list.append(
                    torch.unsqueeze(mol_edge_dict[(i, j)] * edge_feature_oh, 0)
                )
            else:
                # add edges in both directions
                edge_features_list.append(torch.unsqueeze(edge_feature_oh, 0))
                edge_features_list.append(torch.unsqueeze(edge_feature_oh, 0))
        if advs:
            # Update edge dict
            args.edge_dict[smiles_string] = mol_edge_dict

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]

        edge_attr = torch.cat(edge_features_list)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)
        args.edge_dict[smiles_string] = {}

    return edge_attr, edge_index, x


def get_dataset(
    dataset, use_prot=False, target=None, args=None, advs=False, saliency=False
):
    total_dataset = []
    if use_prot:
        prot_graph = transform_molecule_pg(
            target["Fasta"].item(), label=None, is_prot=use_prot
        )

    for mol, label in tqdm(
        zip(dataset["Smiles"], dataset["Label"]), total=len(dataset["Smiles"])
    ):
        if use_prot:
            total_dataset.append([transform_molecule_pg(mol,label,args, advs, saliency=saliency),prot_graph])
        else:
            total_dataset.append(
                transform_molecule_pg(mol, label, args, advs, saliency=saliency)
            )
    return total_dataset


def get_perturbed_dataset(mols, labels, args):
    total_dataset = []
    for mol, label in zip(mols, labels):
        total_dataset.append(transform_molecule_pg(mol, label, args, received_mol=True))
    return total_dataset


def transform_molecule_pg(
    smiles,
    label,
    args=None,
    advs=False,
    received_mol=False,
    saliency=False,
    is_prot=False,
):

    if is_prot:
        edge_attr_p, edge_index_p, x_p = smiles_to_graph(smiles, is_prot)
        x_p = torch.tensor(x_p)
        edge_index_p = torch.tensor(edge_index_p)
        edge_attr_p = torch.tensor(edge_attr_p)

        return Data(edge_attr=edge_attr_p, edge_index=edge_index_p, x=x_p)

    else:
        if args.advs or received_mol:
            if advs or received_mol:
                edge_attr, edge_index, x = smiles_to_graph_advs(
                    smiles,
                    args,
                    advs=True,
                    received_mol=received_mol,
                    saliency=saliency,
                )
            else:
                edge_attr, edge_index, x = smiles_to_graph_advs(
                    smiles, args, received_mol=received_mol, saliency=saliency
                )
        else:
            edge_attr, edge_index, x = smiles_to_graph(smiles, saliency=saliency)

        if not saliency:
            x = torch.tensor(x)
        y = torch.tensor([label])
        edge_index = torch.tensor(edge_index)
        if not args.advs and not received_mol:
            edge_attr = torch.tensor(edge_attr)

        if received_mol:
            mol = smiles
        else:
            mol = Chem.MolFromSmiles(smiles)

        return Data(
            edge_attr=edge_attr, edge_index=edge_index, x=x, y=y, mol=mol, smiles=smiles
        )
