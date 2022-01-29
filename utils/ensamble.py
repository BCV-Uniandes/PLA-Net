import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
from model import DeeperGCN
from model_concatenation import SuperDeeperGCN
from tqdm import tqdm
from args import ArgsInit
from utils.ckpt_util import save_ckpt
import logging
import time
import statistics
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from dataset import load_dataset
import torch.nn.functional as F
import metrics_pharma
import copy
import numpy as np
import datetime
import os
import csv


@torch.no_grad()
def eval(model, device, loader, num_classes, args):
    model.eval()
    y_true = []
    y_pred = []
    correct = 0

    print("------Copying model 1---------")
    prop_predictor1 = copy.deepcopy(model)
    print("------Copying model 2---------")
    prop_predictor2 = copy.deepcopy(model)
    print("------Copying model 3---------")
    prop_predictor3 = copy.deepcopy(model)
    print("------Copying model 4---------")
    prop_predictor4 = copy.deepcopy(model)
    # breakpoint()
    test_model_path = os.path.join(
        args.save,'BINARY_'+args.target
    )
    test_model_path1 = test_model_path + "/Fold1/model_ckpt/Checkpoint_valid_best.pth"
    test_model_path2 = test_model_path + "/Fold2/model_ckpt/Checkpoint_valid_best.pth"
    test_model_path3 = test_model_path + "/Fold3/model_ckpt/Checkpoint_valid_best.pth"
    test_model_path4 = test_model_path + "/Fold4/model_ckpt/Checkpoint_valid_best.pth"
    # LOAD MODELS
    print("------- Loading weights----------")
    prop_predictor1.load_state_dict(torch.load(test_model_path1)["model_state_dict"])
    prop_predictor1.to(device)

    prop_predictor2.load_state_dict(torch.load(test_model_path2)["model_state_dict"])
    prop_predictor2.to(device)

    prop_predictor3.load_state_dict(torch.load(test_model_path3)["model_state_dict"])
    prop_predictor3.to(device)

    prop_predictor4.load_state_dict(torch.load(test_model_path4)["model_state_dict"])
    prop_predictor4.to(device)

    # METHOD.EVAL
    prop_predictor1.eval()
    prop_predictor2.eval()
    prop_predictor3.eval()
    prop_predictor4.eval()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        if args.use_prot:
            batch_mol = batch[0].to(device)
            batch_prot = batch[1].to(device)
        else:
            batch_mol = batch.to(device)
        if args.feature == "full":
            pass
        elif args.feature == "simple":
            # only retain the top two node/edge features
            num_features = args.num_features
            batch_mol.x = batch_mol.x[:, :num_features]
            batch_mol.edge_attr = batch_mol.edge_attr[:, :num_features]
        if batch_mol.x.shape[0] == 1:
            pass
        else:
            with torch.set_grad_enabled(False):
                if args.use_prot:
                    pred1 = F.softmax(prop_predictor1(batch_mol, batch_prot), dim=1)
                    pred2 = F.softmax(prop_predictor2(batch_mol, batch_prot), dim=1)
                    pred3 = F.softmax(prop_predictor3(batch_mol, batch_prot), dim=1)
                    pred4 = F.softmax(prop_predictor4(batch_mol, batch_prot), dim=1)
                else:
                    pred1 = F.softmax(prop_predictor1(batch_mol), dim=1)
                    pred2 = F.softmax(prop_predictor2(batch_mol), dim=1)
                    pred3 = F.softmax(prop_predictor3(batch_mol), dim=1)
                    pred4 = F.softmax(prop_predictor4(batch_mol), dim=1)

                pred = (pred1 + pred2 + pred3 + pred4) / 4
                y_true.append(batch_mol.y.view(batch_mol.y.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())
                _, prediction_class = torch.max(pred, 1)
                correct += torch.sum(prediction_class == batch_mol.y)

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    if args.binary:
        # nap, f = metrics_pharma.norm_ap_binary(y_pred, y_true, num_classes)
        auc = metrics_pharma.plotbinauc(y_pred, y_true)
        nap, f = metrics_pharma.pltmap_bin(y_pred, y_true)
    else:
        nap, f = metrics_pharma.norm_ap(y_pred, y_true, num_classes)
        auc = metrics_pharma.pltauc(y_pred, y_true, num_classes)

    acc = correct / len(loader.dataset)

    return acc, auc, f, nap


def main(target):

    args = ArgsInit().args
    if args.target is None:
        args.target = target

    if args.use_gpu:
        device = (
            torch.device("cuda:" + str(args.device))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        device = torch.device("cpu")

    if args.binary:
        args.nclasses = 2

    # Numpy and torch seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    print(args)

    ( _,_,test_dataset,_,_,_,) = load_dataset(
        cross_val=args.cross_val,
        binary_task=args.binary,
        target=args.target,
        use_prot=args.use_prot,
        args=args,
        test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    if args.use_prot:
        model = SuperDeeperGCN(args).to(device)
    else:
        model = DeeperGCN(args).to(device)

    acc, auc, f, nap = eval(model, device, test_loader, args.nclasses, args)

    save_items = {"Target": [], "NAP": [], "AUC": [], "ACC": [], "F_Med": []}

    save_items["Target"] = args.target
    save_items["NAP"] = nap
    save_items["AUC"] = auc
    save_items["ACC"] = acc.item()
    save_items["F_Med"] = f

    fieldnames = list(save_items.keys())

    csv_file = os.path.join(
        args.save,'Performance.csv'
    )
    if not os.path.exists(csv_file):
        create_header = True
    else:
        create_header = False

    with open(csv_file, "a+") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if create_header:
            writer.writeheader()
        writer.writerow(save_items)

    print({"ACC": acc, "AUC": auc, "F-medida": f, "NAP": nap})

    return nap


if __name__ == "__main__":

    args = ArgsInit().args
    if args.target is None:
            
        targets = ['aa2ar', 'abl1', 'ace', 'aces', 'ada', 'ada17', 'adrb1', 'adrb2',
        'akt1', 'akt2', 'aldr', 'ampc', 'andr', 'aofb', 'bace1', 'braf',
        'cah2', 'casp3', 'cdk2', 'comt', 'cp2c9', 'cp3a4', 'csf1r',
        'cxcr4', 'def', 'dhi1', 'dpp4', 'drd3', 'dyr', 'egfr', 'esr1',
        'esr2', 'fa10', 'fa7', 'fabp4', 'fak1', 'fgfr1', 'fkb1a', 'fnta',
        'fpps', 'gcr', 'glcm', 'gria2', 'grik1', 'hdac2', 'hdac8',
        'hivint', 'hivpr', 'hivrt', 'hmdh', 'hs90a', 'hxk4', 'igf1r',
        'inha', 'ital', 'jak2', 'kif11', 'kit', 'kith', 'kpcb', 'lck',
        'lkha4', 'mapk2', 'mcr', 'met', 'mk01', 'mk10', 'mk14', 'mmp13',
        'mp2k1', 'nos1', 'nram', 'pa2ga', 'parp1', 'pde5a', 'pgh1', 'pgh2',
        'plk1', 'pnph', 'ppara', 'ppard', 'pparg', 'prgr', 'ptn1', 'pur2',
        'pygm', 'pyrd', 'reni', 'rock1', 'rxra', 'sahh', 'src', 'tgfr1',
        'thb', 'thrb', 'try1', 'tryb1', 'tysy', 'urok', 'vgfr2', 'wee1',
        'xiap']
        
        results = {'Target': [], 'Mean_Test': []}
        
        for target in targets:
            nap_result = main(target)
            results['Target'].append(target)
            results['Mean_Test'].append(nap_result)
        
        torch.save(results,os.path.join(args.save,'Overall_test_results.pth'))
        print('Mean Test: {}'.format(np.mean(results['Mean_Test'])))
    else:
        main()
