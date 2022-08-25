import os
import sys
import copy
import time
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader


from utils import metrics_pharma
from utils.args import ArgsInit
from utils.ckpt_util import save_ckpt

from model.model import DeeperGCN
from model.model_concatenation import PLANet

from data.dataset import load_dataset, reload_dataset, get_perturbed_dataset


def robust_augment(model, batch, threshold, device, args):
    """
    Compute augmented molecules based on Rogot Goldberg Similarity between original molecule and augmented molecule.
    Args:
        model:
        batch:
        threshold:
        device:
        args:
    Returns:
        perturbed_batch:
    """
    print("Computing augmented molecules...")

    def Similarity(fps):
        # first generate the distance matrix:
        dists = []
        fps_all = copy.deepcopy(fps)
        fps_all.pop(0)
        sims = DataStructs.BulkRogotGoldbergSimilarity(fps[0], fps[0:2])
        dists.extend([1 - x for x in sims])
        return dists

    cls_criterion = torch.nn.BCELoss()

    # Begin augmentations
    perturbed_mols = []
    perturbed_labels = []
    perturbed_smiles = []
    orig_labels = []

    if args.feature == "full":
        pass
    elif args.feature == "simple":
        # only retain the top two node/edge features
        num_features = args.num_features
        batch.x = batch.x[:, :num_features]
        batch.edge_attr = batch.edge_attr[:, :num_features]
    if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
        pass
    else:
        model.eval()
        pred = model(batch, dropout=False)
        is_labeled = batch.y == batch.y
        labels = torch.unsqueeze(batch.y, 1)

        with torch.enable_grad():
            loss = 0
            class_loss = cls_criterion(
                F.sigmoid(pred[:, 1]).to(torch.float32), batch.y.to(torch.float32)
            )
            loss += class_loss
        loss.backward()

        batch_mol_id = 0
        smiles_used = []
        for mol, smiles in tqdm(zip(batch.mol, batch.smiles), total=len(batch.smiles)):
            if smiles in smiles_used:
                breakpoint()
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            fp_original = AllChem.GetMorganFingerprintAsBitVect(scaffold, 2, 1024)

            find_edge = True
            iteration = 0
            mol_dict = args.edge_dict[smiles].copy()

            num_edges = len(mol_dict.keys())

            while find_edge and iteration < num_edges:
                del_edge = min(mol_dict.keys(), key=lambda k: mol_dict[k].grad)
                grad = mol_dict[del_edge].grad.detach()

                if grad >= 0:
                    break

                atom_1, atom_2 = del_edge

                deg_atom_1 = mol.GetAtomWithIdx(atom_1).GetDegree()
                deg_atom_2 = mol.GetAtomWithIdx(atom_2).GetDegree()
                emol = copy.deepcopy(mol)
                emol = Chem.EditableMol(emol)
                emol.RemoveBond(atom_1, atom_2)
                if deg_atom_1 == 1:
                    emol.RemoveAtom(atom_1)
                if deg_atom_2 == 1:
                    emol.RemoveAtom(atom_2)

                perturbed_mol = emol.GetMol()

                Chem.rdmolops.FastFindRings(perturbed_mol)
                fp_perturbed = AllChem.GetMorganFingerprintAsBitVect(
                    perturbed_mol, 2, 1024
                )
                fps = [fp_original, fp_perturbed, fp_perturbed]
                _, similarity = Similarity(fps)

                if int(labels[batch_mol_id]) == 0:
                    statement = similarity > threshold[1]
                else:
                    statement = similarity < threshold[1]

                if statement:
                    find_edge = False
                    perturbed_mols.append(perturbed_mol)
                    perturbed_labels.append(labels[batch_mol_id].tolist()[0])
                    perturbed_smiles.append(smiles + "-p")
                    break
                else:
                    del mol_dict[del_edge]
                    iteration += 1
            batch_mol_id += 1
            #            del args.edge_dict[smiles]
            smiles_used.append(smiles)
        orig_labels += batch.y.tolist()
    with open(os.path.join(args.save, "adversaries.pickle"), "wb") as file:
        pickle.dump(perturbed_mols, file)
    print("Saved adversaries to" + os.path.join(args.save, "adversaries.pickle"))

    all_labels = orig_labels + perturbed_labels

    weights_train = make_weights_for_balanced_classes(all_labels, args.nclasses)
    weights_train = torch.DoubleTensor(weights_train)
    new_sampler_train = torch.utils.data.sampler.WeightedRandomSampler(
        weights_train, len(weights_train)
    )

    total_mols = copy.deepcopy(batch.mol) + perturbed_mols
    perturbed_set = get_perturbed_dataset(total_mols, all_labels, args)
    new_train_loader = DataLoader(
        perturbed_set,
        batch_size=len(perturbed_set),
        sampler=new_sampler_train,
        num_workers=args.num_workers,
    )
    for perturbed_batch in new_train_loader:  # only runs once
        perturbed_batch = perturbed_batch.to(device)

    print("Done.")
    return perturbed_batch


def train(
    model, device, loader, optimizer, num_classes, args, threshold, trainset=None
):
    """
    Perform training for one epoch.
    Args:
        model:
        device:
        loader (loader): Training loader
        optimizer:
        num_classes (int): Number of classes
        args (parser): Model's configuration
        threshold (dict):
        trainset:
    Return:
        loss:
    """
    loss_list = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        if args.use_prot:
            batch_mol = batch[0].to(device)
            batch_prot = batch[1].to(device)
        else:
            batch_mol = batch.to(device)

        if args.advs:
            perturbed_batch = robust_augment(model, batch_mol, threshold, device, args)

        model.train()
        if args.feature == "full":
            pass
        elif args.feature == "simple":
            num_features = args.num_features
            batch_mol.x = batch_mol.x[:, :num_features]
            batch_mol.edge_attr = batch_mol.edge_attr[:, :num_features]
        if batch_mol.x.shape[0] == 1:
            pass
        else:
            optimizer.zero_grad()
            loss = 0
            if args.advs:
                perturbed_pred = model(perturbed_batch)
                class_loss = cls_criterion(
                    F.sigmoid(perturbed_pred[:, 1]).to(torch.float32),
                    perturbed_batch.y.to(torch.float32),
                )

            elif args.use_prot:
                pred = model(batch_mol, batch_prot)
                class_loss = cls_criterion(
                    F.sigmoid(pred[:, 1]).to(torch.float32),
                    batch_mol.y.to(torch.float32),
                )

            else:
                pred = model(batch_mol)
                class_loss = cls_criterion(
                    F.sigmoid(pred[:, 1]).to(torch.float32),
                    batch_mol.y.to(torch.float32),
                )

            loss += class_loss

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

    return np.mean(loss_list)


@torch.no_grad()
def eval_gcn(model, device, loader, num_classes, args):
    """
    Evaluate the model on the validation set.
    Args:
    Return:
    """
    model.eval()
    loss_list = []
    y_true = []
    y_pred = []
    correct = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        if args.use_prot:
            batch_mol = batch[0].to(device)
            batch_prot = batch[1].to(device)
        else:
            batch_mol = batch.to(device)

        if args.feature == "full":
            pass
        elif args.feature == "simple":
            num_features = args.num_features
            batch_mol.x = batch_mol.x[:, :num_features]
            batch_mol.edge_attr = batch_mol.edge_attr[:, :num_features]
        if batch_mol.x.shape[0] == 1:
            pass
        else:
            with torch.set_grad_enabled(False):
                if args.use_prot:
                    pred = model(batch_mol, batch_prot)
                else:
                    pred = model(batch_mol)

                loss = 0
                class_loss = cls_criterion(
                    F.sigmoid(pred[:, 1]).to(torch.float32),
                    batch_mol.y.to(torch.float32),
                )
                loss += class_loss
                loss_list.append(loss.item())
                pred = F.softmax(pred, dim=1)
                y_true.append(batch_mol.y.view(batch_mol.y.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())
                _, prediction_class = torch.max(pred, 1)
                correct += torch.sum(prediction_class == batch_mol.y)

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    if args.binary:
        nap, f = metrics_pharma.pltmap_bin(y_pred, y_true, num_classes)
        auc = metrics_pharma.plotbinauc(y_pred, y_true)

    acc = correct / len(loader.dataset)

    return acc, auc, f, nap, np.mean(loss_list)


def make_weights_for_balanced_classes(data, nclasses):
    """
    Generate weights for a balance training loader.
    Args:
        data (list): Labels of each molecule
        nclasses (int): number of classes
    Return:
        weight (list): Weights for each class
    """
    count = [0] * nclasses
    for item in data:
        count[item] += 1
    weight_per_class = [0.0] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(data)
    for idx, val in enumerate(data):
        weight[idx] = weight_per_class[val]

    return weight


def main():
    """
    Train a model on the train set and evaluate it on validation set.
    """
    # Init args
    args = ArgsInit().save_exp()

    # Read threshold-files use to select augmented molecules.
    if args.advs:
        args.edge_dict = {}
        if args.binary:
            df = pd.read_csv(
                "./threshold/Binary_Umbral_Maximium_" + args.target + ".csv"
            )
        else:
            df = pd.read_csv("./threshold/Umbral_Molecules_Maximium.csv")
        class_label = np.asarray(df["Class"])
        thresh = np.asarray(df["Umbral"])
        threshold = {k: v for k, v in zip(class_label, thresh)}
    else:
        threshold = {}

    # Set device
    if args.use_gpu:
        device = (
            torch.device("cuda:" + str(args.device))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        device = torch.device("cpu")

    # Set number of classes for binary training
    # TODO: AUTOMATIZAR LOS LR???
    if args.binary:
        args.nclasses = 2

    # Set random seed for numpy, torch and cuda
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    # Print model configuration
    logging.info("%s" % args)

    # Load all data splits
    train_dataset, valid_dataset, test_dataset, data_train, _, _ = load_dataset(
        cross_val=args.cross_val,
        binary_task=args.binary,
        target=args.target,
        args=args,
        use_prot=args.use_prot,
        advs=args.advs,
    )

    # Create a balance traning loader
    if args.balanced_loader:

        weights_train = make_weights_for_balanced_classes(
            list(data_train.Label), args.nclasses
        )
        weights_train = torch.DoubleTensor(weights_train)
        sampler_train = torch.utils.data.sampler.WeightedRandomSampler(
            weights_train, len(weights_train)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler_train,
            num_workers=args.num_workers,
        )
    else:
        # Create an unbalance traning loader

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
    # Create validation and test loaders.
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    # Define the model based on the configuration (With or without PM Module)
    if args.use_prot:
        model = PLANet(args).to(device)
    else:
        model = DeeperGCN(args).to(device)

    # Save model's configuration
    logging.info(model)

    # Set the optimizer and it's parameters
    optimizer = optim.Adamax(model.parameters(), lr=args.lr)

    # Set dictionary that is used to save the best results on every epoch.
    results = {
        "lowest_valid_loss": 100,
        "highest_valid": 0,
        "highest_train": 0,
        "epoch": 0,
    }

    start_time = time.time()

    # Set lists to save overall metrics and loss
    train_epoch_loss = []
    val_epoch_loss = []
    train_epoch_nap = []
    val_epoch_nap = []

    # Load model to resume training
    if args.resume:
        model_name = os.path.join(args.save, "model_ckpt", args.model_load_path)
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        init_epoch = checkpoint["epoch"] + 1
        train_epoch_loss = checkpoint["loss_train"]
        val_epoch_loss = checkpoint["loss_val"]
        train_epoch_nap = checkpoint["nap_train"]
        val_epoch_nap = checkpoint["nap_val"]
        results["highest_valid"] = max(val_epoch_nap)
        results["lowest_valid_loss"] = min(val_epoch_loss)
        results["highest_train"] = max(train_epoch_nap)
        results["epoch"] = init_epoch
        logging.info("Model loaded")
    else:
        init_epoch = 1

    # TODO: Revisar si la quitamos
    if args.init_adv_training:
        model_name = os.path.join(
            args.model_load_init_path, "model_ckpt", args.model_load_path
        )
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        for i in range(args.num_layers):
            checkpoint["model_state_dict"][
                "gcns." + str(i) + ".edge_encoder.bond_embedding_list.0.weight"
            ] = checkpoint["model_state_dict"][
                "gcns." + str(i) + ".edge_encoder.bond_embedding_list.0.weight"
            ].t()
            checkpoint["model_state_dict"][
                "gcns." + str(i) + ".edge_encoder.bond_embedding_list.1.weight"
            ] = checkpoint["model_state_dict"][
                "gcns." + str(i) + ".edge_encoder.bond_embedding_list.1.weight"
            ].t()
            checkpoint["model_state_dict"][
                "gcns." + str(i) + ".edge_encoder.bond_embedding_list.2.weight"
            ] = checkpoint["model_state_dict"][
                "gcns." + str(i) + ".edge_encoder.bond_embedding_list.2.weight"
            ].t()
        model.load_state_dict(checkpoint["model_state_dict"])
        init_epoch = 1
        print("Model loaded")

    # Load pre-trained LM module
    elif args.LMPM:

        best_molecule_path = args.model_load_init_path
        model_path = (
            "BINARY_{}/Fold{}/model_ckpt/BS_2560-NF_full_valid_best.pth".format(
                args.target, args.cross_val
            )
        )
        full_model_path = os.path.join(best_molecule_path, model_path)
        pre_model = torch.load(full_model_path)
        model_weights = {}
        for k, v in pre_model["model_state_dict"].items():
            if args.use_prot:
                if v.shape == model.molecule_gcn.state_dict()[k].shape:
                    model_weights[k] = v
                else:
                    model_weights[k] = torch.transpose(v, 0, 1)
            else:
                if v.shape == model.state_dict()[k].shape:
                    model_weights[k] = v
                else:
                    model_weights[k] = torch.transpose(v, 0, 1)
        model.molecule_gcn.load_state_dict(model_weights)
        dict_clasificacion = {
            "weight": pre_model["model_state_dict"]["graph_pred_linear.weight"],
            "bias": pre_model["model_state_dict"]["graph_pred_linear.bias"],
        }
        model.classification.load_state_dict(dict_clasificacion, strict=False)
        all_params = []
        variable_params = []
        for name, param in model.named_parameters():
            all_params.append(name)
            if param.requires_grad:
                variable_params.append(name)

        if len(variable_params) < len(all_params):
            logging.info(
                "Molecule model loaded and freezed. Experimenting with target {} model only.".format(
                    args.target
                )
            )
    # Load pre-traines LM+Advs and PM models
    elif args.PLANET:
        best_molecule_path = args.model_load_init_path
        model_path = "BINARY_{}/Fold{}/model_ckpt/Checkpoint__Best.pth".format(
            args.target, args.cross_val
        )
        full_model_path = os.path.join(best_molecule_path, model_path)

        pre_model = torch.load(full_model_path)
        model_weights = {}
        for k, v in pre_model["model_state_dict"].items():
            if args.use_prot:
                if v.shape == model.molecule_gcn.state_dict()[k].shape:
                    model_weights[k] = v
                else:
                    model_weights[k] = torch.transpose(v, 0, 1)
            else:
                if v.shape == model.state_dict()[k].shape:
                    model_weights[k] = v
                else:
                    model_weights[k] = torch.transpose(v, 0, 1)
        model.molecule_gcn.load_state_dict(model_weights)
        dict_clasificacion = {
            "weight": pre_model["model_state_dict"]["graph_pred_linear.weight"],
            "bias": pre_model["model_state_dict"]["graph_pred_linear.bias"],
        }
        model.classification.load_state_dict(dict_clasificacion, strict=False)
        # Freeze LM
        for param in model.molecule_gcn.parameters():
            param.requires_grad = False

        # PARA ENTRENAR SOLO LA CAPA LINEAL
        best_molecule_path = args.model_load_prot_init_path
        model_path = "BINARY_{}/Fold{}/model_ckpt/Checkpoint_valid_best.pth".format(
            args.target, args.cross_val
        )
        full_model_path = os.path.join(best_molecule_path, model_path)

        pre_model_prot = torch.load(full_model_path, map_location=device)
        model_weights_prot = {k.replace('target_gcn.', ''):v for k,v in pre_model_prot['model_state_dict'].items() if k.startswith('target_gcn')}

        model.target_gcn.load_state_dict(model_weights_prot)

        logging.info("Protein Loaded")
        # Freeze PM
        for param in model.target_gcn.parameters():
            param.requires_grad = False

        all_params = []
        variable_params = []
        for name, param in model.named_parameters():
            all_params.append(name)
            if param.requires_grad:
                variable_params.append(name)

        if len(variable_params) < len(all_params):
            logging.info(
                "Molecule and protein model loaded and freezed. Experimenting with target {} model only.".format(
                    args.target
                )
            )
    # TODO: Revisar si se puede borrar
    # ESTO ES PARA ENTRENAR SOLO LA CAPA LINEAL
    #      best_molecule_path = args.model_load_prot_init_path
    #      model_path = 'BINARY_{}/Fold{}/model_ckpt/Checkpoint_valid_best.pth'.format(args.target, args.cross_val)
    #      full_model_path = os.path.join(best_molecule_path, model_path)
    #      pre_model = torch.load(full_model_path)
    #      model_weights = {k.replace('target_gcn.', ''):v for k,v in pre_model['model_state_dict'].items() if k.startswith('target_gcn')}
    #      model.target_gcn.load_state_dict(model_weights)
    #      logging.info('Protein Loaded')

    loss_track = 0
    past_loss = 0
    # Training
    for epoch in range(init_epoch, args.epochs + 1):

        logging.info("=====Epoch {}".format(epoch))
        logging.info("Training...")

        if epoch == 1:
            # Evaluate loaded models
            logging.info("Evaluating First Epoch...")
            val_acc, val_auc, val_f, val_nap, val_loss = eval_gcn(
                model, device, valid_loader, args.nclasses, args
            )
            logging.info(
                "Valid:Loss {}, ACC {}, AUC {}, F-Measure {}, AP {}".format(
                    val_loss, val_acc, val_auc, val_f, val_nap
                )
            )
        if args.advs:
            tr_loss = train(
                model,
                device,
                train_loader,
                optimizer,
                args.nclasses,
                args,
                threshold,
                trainset=train_dataset,
            )
        else:
            tr_loss = train(
                model, device, train_loader, optimizer, args.nclasses, args, threshold
            )

        logging.info("Evaluating...")
        tr_acc, tr_auc, tr_f, tr_nap, tr_loss = eval_gcn(
            model, device, train_loader, args.nclasses, args
        )
        val_acc, val_auc, val_f, val_nap, val_loss = eval_gcn(
            model, device, valid_loader, args.nclasses, args
        )

        train_epoch_loss.append(tr_loss)
        val_epoch_loss.append(val_loss)
        train_epoch_nap.append(tr_nap)
        val_epoch_nap.append(val_nap)

        metrics_pharma.plot_loss(
            train_epoch_loss, val_epoch_loss, save_dir=args.save, num_epoch=args.epochs
        )
        metrics_pharma.plot_nap(
            train_epoch_nap, val_epoch_nap, save_dir=args.save, num_epoch=args.epochs
        )

        logging.info(
            "Train:Loss {}, ACC {}, AUC {}, F-Measure {}, AP {}".format(
                tr_loss, tr_acc, tr_auc, tr_f, tr_nap
            )
        )
        logging.info(
            "Valid:Loss {}, ACC {}, AUC {}, F-Measure {}, AP {}".format(
                val_loss, val_acc, val_auc, val_f, val_nap
            )
        )

        logging.info("Learning Rate: {}".format(optimizer.param_groups[0]["lr"]))

        sub_dir = "Checkpoint"
        save_ckpt(
            model,
            optimizer,
            train_epoch_loss,
            val_epoch_loss,
            train_epoch_nap,
            val_epoch_nap,
            epoch,
            args.model_save_path,
            sub_dir,
            name_post="Last_model",
        )

        if tr_nap > results["highest_train"]:

            results["highest_train"] = tr_nap

        if val_loss < results["lowest_valid_loss"]:
            results["lowest_valid_loss"] = val_loss
            results["epoch"] = epoch

            save_ckpt(
                model,
                optimizer,
                train_epoch_loss,
                val_epoch_loss,
                train_epoch_nap,
                val_epoch_nap,
                epoch,
                args.model_save_path,
                sub_dir,
                name_post="valid_best",
            )
        if args.advs or args.PLANET:
            if val_nap > results["highest_valid"]:
                results["highest_valid"] = val_nap
                results["epoch"] = epoch

                save_ckpt(
                    model,
                    optimizer,
                    train_epoch_loss,
                    val_epoch_loss,
                    train_epoch_nap,
                    val_epoch_nap,
                    epoch,
                    args.model_save_path,
                    sub_dir,
                    name_post="valid_best_nap",
                )

        if args.PLANET or args.advs:
            if val_loss >= past_loss:
                loss_track += 1
            else:
                loss_track = 0
            past_loss = val_loss

            if args.PLANET and loss_track >= 5:
                logging.info("Early exit due to overfitting")
                end_time = time.time()
                total_time = end_time - start_time
                logging.info("Best model in epoch: {}".format(results["epoch"]))
                logging.info(
                    "Total time: {}".format(
                        time.strftime("%H:%M:%S", time.gmtime(total_time))
                    )
                )
                sys.exit()
            if args.advs and loss_track >= 15:
                logging.info("Early exit due to overfitting")
                end_time = time.time()
                total_time = end_time - start_time
                logging.info("Best model in epoch: {}".format(results["epoch"]))
                logging.info(
                    "Total time: {}".format(
                        time.strftime("%H:%M:%S", time.gmtime(total_time))
                    )
                )
                sys.exit()
        if args.advs:
            train_dataset, data_train = reload_dataset(
                cross_val=args.cross_val,
                binary_task=args.binary,
                target=args.target,
                args=args,
                advs=args.advs,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
    end_time = time.time()
    total_time = end_time - start_time
    logging.info("Best model in epoch: {}".format(results["epoch"]))
    logging.info(
        "Total time: {}".format(time.strftime("%H:%M:%S", time.gmtime(total_time)))
    )


if __name__ == "__main__":
    cls_criterion = torch.nn.BCELoss()
    reg_criterion = torch.nn.MSELoss()
    main()
