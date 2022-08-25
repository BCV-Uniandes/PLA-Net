import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
from model import DeeperGCN
from tqdm import tqdm
from args import ArgsInit
from utils.ckpt_util import save_ckpt
from main import robust_augment
import logging
import time
import statistics
from dataset import load_dataset
import metrics_pharma
import copy
import numpy as np 
import datetime
import os 
import csv 
import torch.nn.functional as F
import pandas as pd
import rdkit.Chem.Draw as Draw
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
from model_concatenation import SuperDeeperGCN

def moltosvg(mol, highlightMap, molSize = (300,300), kekulize = True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMoleculeWithHighlights(mc,highlight_atom_map=highlightMap)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace('svg:','')

def compute_saliency_map(model, device, loader, num_classes, args, target,num=0): 
    cls_criterion = torch.nn.BCELoss()

    print('------Copying model 1---------')
    prop_predictor1 = copy.deepcopy(model)

    print('------Copying model 2---------')
    prop_predictor2 = copy.deepcopy(model)

    print('------Copying model 3---------')
    prop_predictor3 = copy.deepcopy(model)

    print('------Copying model 4---------')
    prop_predictor4 = copy.deepcopy(model)
    
#    test_model_path = '/data/lrueda/Molecules-Graphs/deep_gcns_torch-master/examples/ogb/dude_dataset/log/BINARY___'+ target
#    test_model_path = '/data/pruiz/PLA-Net/LM/BINARY_'+ target
    test_model_path = '/data/lrueda/Molecules-Graphs/deep_gcns_torch-master/examples/ogb/dude_dataset/log/PLA-NET_Nature/BINARY_'+ target
#    test_model_path1 = test_model_path+'/Fold1/model_ckpt/BS_2560-NF_full_valid_best.pth'
#    test_model_path2 = test_model_path+'/Fold2/model_ckpt/BS_2560-NF_full_valid_best.pth'
#    test_model_path3 = test_model_path+'/Fold3/model_ckpt/BS_2560-NF_full_valid_best.pth'
#    test_model_path4 = test_model_path+'/Fold4/model_ckpt/BS_2560-NF_full_valid_best.pth'

    test_model_path1 = test_model_path+'/Fold1/model_ckpt_2/Best.pth' #/Checkpoint__Best.pth'
    test_model_path2 = test_model_path+'/Fold2/model_ckpt_2/Best.pth' #/Checkpoint__Best.pth'
    test_model_path3 = test_model_path+'/Fold3/model_ckpt_2/Best.pth' #/Checkpoint__Best.pth'
    test_model_path4 = test_model_path+'/Fold4/model_ckpt_2/Best.pth' #/Checkpoint__Best.pth'

#    import pdb; pdb.set_trace()
    #LOAD MODELS
    print('------- Loading weights----------')
    ckpt1 = torch.load(test_model_path1,
            map_location=lambda storage, loc: storage)
    ckpt1['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.0.weight'] = ckpt1['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.0.weight'].t()
    ckpt1['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.1.weight'] = ckpt1['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.1.weight'].t()
    ckpt1['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.2.weight'] = ckpt1['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.2.weight'].t()
    ckpt1['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.3.weight'] = ckpt1['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.3.weight'].t()
    ckpt1['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.4.weight'] = ckpt1['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.4.weight'].t()
    ckpt1['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.5.weight'] = ckpt1['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.5.weight'].t()
    ckpt1['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.6.weight'] = ckpt1['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.6.weight'].t()
    ckpt1['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.7.weight'] = ckpt1['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.7.weight'].t()
    ckpt1['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.8.weight'] = ckpt1['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.8.weight'].t()
    
    for i in range(args.num_layers):
        ckpt1['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.0.weight'] = ckpt1['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.0.weight'].t()
        ckpt1['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.1.weight'] = ckpt1['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.1.weight'].t()
        ckpt1['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.2.weight'] = ckpt1['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.2.weight'].t()
 
    prop_predictor1.load_state_dict(ckpt1['model_state_dict'])
    prop_predictor1.to(device)

    ckpt2 = torch.load(test_model_path2,
            map_location=lambda storage, loc: storage)
   
    ckpt2['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.0.weight'] = ckpt2['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.0.weight'].t()
    ckpt2['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.1.weight'] = ckpt2['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.1.weight'].t()
    ckpt2['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.2.weight'] = ckpt2['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.2.weight'].t()
    ckpt2['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.3.weight'] = ckpt2['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.3.weight'].t()
    ckpt2['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.4.weight'] = ckpt2['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.4.weight'].t()
    ckpt2['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.5.weight'] = ckpt2['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.5.weight'].t()
    ckpt2['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.6.weight'] = ckpt2['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.6.weight'].t()
    ckpt2['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.7.weight'] = ckpt2['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.7.weight'].t()
    ckpt2['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.8.weight'] = ckpt2['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.8.weight'].t()

    for i in range(args.num_layers):
        ckpt2['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.0.weight'] = ckpt2['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.0.weight'].t()
        ckpt2['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.1.weight'] = ckpt2['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.1.weight'].t()
        ckpt2['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.2.weight'] = ckpt2['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.2.weight'].t()

    prop_predictor2.load_state_dict(ckpt2['model_state_dict'])
    prop_predictor2.to(device)
    
    ckpt3 = torch.load(test_model_path3,
            map_location=lambda storage, loc: storage)
    ckpt3['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.0.weight'] = ckpt3['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.0.weight'].t()
    ckpt3['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.1.weight'] = ckpt3['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.1.weight'].t()
    ckpt3['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.2.weight'] = ckpt3['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.2.weight'].t()
    ckpt3['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.3.weight'] = ckpt3['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.3.weight'].t()
    ckpt3['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.4.weight'] = ckpt3['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.4.weight'].t()
    ckpt3['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.5.weight'] = ckpt3['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.5.weight'].t()
    ckpt3['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.6.weight'] = ckpt3['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.6.weight'].t()
    ckpt3['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.7.weight'] = ckpt3['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.7.weight'].t()
    ckpt3['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.8.weight'] = ckpt3['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.8.weight'].t()

    for i in range(args.num_layers):
        ckpt3['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.0.weight'] = ckpt3['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.0.weight'].t()
        ckpt3['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.1.weight'] = ckpt3['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.1.weight'].t()
        ckpt3['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.2.weight'] = ckpt3['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.2.weight'].t()

    prop_predictor3.load_state_dict(ckpt3['model_state_dict'])
    prop_predictor3.to(device)
    
    ckpt4 = torch.load(test_model_path4,
            map_location=lambda storage, loc: storage)

    ckpt4['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.0.weight'] = ckpt4['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.0.weight'].t()
    ckpt4['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.1.weight'] = ckpt4['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.1.weight'].t()
    ckpt4['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.2.weight'] = ckpt4['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.2.weight'].t()
    ckpt4['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.3.weight'] = ckpt4['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.3.weight'].t()
    ckpt4['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.4.weight'] = ckpt4['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.4.weight'].t()
    ckpt4['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.5.weight'] = ckpt4['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.5.weight'].t()
    ckpt4['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.6.weight'] = ckpt4['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.6.weight'].t()
    ckpt4['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.7.weight'] = ckpt4['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.7.weight'].t()
    ckpt4['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.8.weight'] = ckpt4['model_state_dict']['molecule_gcn.atom_encoder.atom_embedding_list.8.weight'].t()

    for i in range(args.num_layers):
        ckpt4['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.0.weight'] = ckpt4['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.0.weight'].t()
        ckpt4['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.1.weight'] = ckpt4['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.1.weight'].t()
        ckpt4['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.2.weight'] = ckpt4['model_state_dict']['molecule_gcn.gcns.'+ str(i)+'.edge_encoder.bond_embedding_list.2.weight'].t()

    prop_predictor4.load_state_dict(ckpt4['model_state_dict'])
    prop_predictor4.to(device)
    break_loop = False
    count = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        if args.use_prot:
            batch_mol = batch[0].to(device)
            batch_prot = batch[1].to(device)
        else:
            batch_mol = batch.to(device)
        batch_mol.x.requires_grad=True
#        batch.edge_attr.requires_grad=True
        if args.use_prot:
          pred1 = prop_predictor1(batch_mol,batch_prot)
          pred2 = prop_predictor2(batch_mol,batch_prot)
          pred3 = prop_predictor3(batch_mol,batch_prot)
          pred4 = prop_predictor4(batch_mol,batch_prot)
        else:
          pred1 = prop_predictor1(batch_mol)
          pred2 = prop_predictor2(batch_mol)
          pred3 = prop_predictor3(batch_mol)
          pred4 = prop_predictor4(batch_mol)

        pred = (pred1 + pred2 + pred3 + pred4)/4

        is_labeled = batch_mol.y == batch_mol.y
        labels = torch.unsqueeze(batch_mol.y, 1)
        with torch.enable_grad():
           loss = 0
           class_loss = cls_criterion(F.sigmoid(pred[:,1]).to(torch.float32), batch_mol.y.to(torch.float32))
           loss += class_loss
#           for i in range(0,args.nclasses):
#               class_mask = batch.y.clone()
#               class_mask[batch.y == i] = 1
#               class_mask[batch.y != i] = 0
#               class_loss = cls_criterion(F.sigmoid(pred[:,i]).to(torch.float32)[is_labeled], class_mask.to(torch.float32)[is_labeled])
#               loss += class_loss
            
        loss.backward()
        atom_grad = batch_mol.x.grad.detach()
#        edge_grad = batch.edge_attr.grad.detach()
        mol_atom_dict = {}
#        mol_edge_dict = {}
        curr_idb = 0
        curr_ida = 0
        jet = plt.get_cmap('jet')
        for idx, mol in enumerate(batch_mol.mol):
           atom_ids = []
           atom_val = []
           for ida, atom in enumerate(mol.GetAtoms()):
              atom_ids.append(atom.GetIdx())
              atom_val.append(sum(atom_grad[curr_ida+ida,:]).item())
           atom_val =  np.array(atom_val)
           atom_val = (atom_val - np.min(atom_val))/(np.max(atom_val)-np.min(atom_val))
           atom_val = 1-atom_val
           atom_val_norm = [jet(i) for i in atom_val]
           mol_atom_dict[batch_mol.smiles[idx]] = dict(zip(atom_ids, atom_val_norm))
           curr_ida += ida + 1
           img = Draw.MolToImage(mol, highlightMap=mol_atom_dict[batch_mol.smiles[idx]])

           img.save('saliency_maps/'+target+'_PLANET/molecule'+str(idx+count)+'.png')
           count += 1 
           if idx >= len(batch_mol.y[batch_mol.y == 1]):
              break_loop = True
              break
        if break_loop:
           break
def main(target, num=0):

    args = ArgsInit().args

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')
    if args.advs:
        args.edge_dict = {}
    if args.binary:
        args.nclasses = 2
        
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    print(args)

    if not os.path.exists('saliency_maps/'+target+'_PLANET'):
      os.mkdir('saliency_maps/'+target+'_PLANET')

    train_dataset, valid_dataset, test_dataset, data_train, data_val, data_test = load_dataset(cross_val=args.cross_val, binary_task=args.binary,
                                                                                               target=target, args=args, advs=args.advs,use_prot=args.use_prot,test=True)
    
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    if args.use_prot:
        model = SuperDeeperGCN(args, saliency=True).to(device)
    else:
        model = DeeperGCN(args, saliency=True).to(device)

    compute_saliency_map(model, device, test_loader, args.nclasses, args, target, num)

if __name__ == "__main__":
    '''
    targets = ['aa2ar', 'abl1', 'ace', 'aces', 'ada', 'ada17', 'adrb1', 'adrb2',
       'akt1', 'akt2', 'aldr', 'ampc', 'andr', 'aofb', 
       'braf', 'cah2', 'casp3', 'cdk2', 'comt', 'cp2c9', 'cp3a4', 'csf1r',
       'cxcr4', 'def', 'dhi1', 'dpp4', 'drd3', 'dyr', 'egfr', 'esr1',
       'esr2', 'fa10', 'fa7', 'fabp4', 'fak1', 'fgfr1', 'fkb1a', 'fnta',
       'fpps', 'gcr', 'glcm', 'gria2', 'grik1', 'hdac2', 'hivint', 'hivpr', 'hivrt', 'hmdh', 'hs90a', 'hxk4', 'igf1r',
       'inha', 'ital', 'jak2', 'kif11', 'kit', 'kith', 'kpcb', 'lck',
       'lkha4', 'mapk2', 'mcr', 'met', 'mk01', 'mk10', 'mk14', 'mmp13',
       'mp2k1', 'nos1', 'nram', 'pa2ga', 'parp1', 'pde5a', 'pgh1', 'pgh2',
       'plk1', 'pnph', 'ppara', 'ppard', 'pparg', 'prgr', 'ptn1', 'pur2',
       'pygm', 'pyrd', 'reni', 'rock1', 'rxra', 'sahh', 'src', 'tgfr1',
       'thb', 'thrb', 'try1', 'tryb1', 'tysy', 'urok', 'vgfr2', 'wee1',
       'xiap', 'bace1', 'hdac8']
    
    targets_faltan = ['bace1','hdac8']
    
    targets_faltan = []
    '''
 #   nums = [48,28,46,24,21,29,10,15,4,26,11]
#    targets = ['aa2ar', 'ace', 'aces', 'adrb1', 'adrb2','akt1', 'akt2', 'aldr',
#    nums=[]
#    targets=['cah2', 'casp3', 'cdk2', 'comt', 'cp2c9', 'cp3a4', 'csf1r','dhi1', 'dpp4', 'drd3', 'dyr',
#             'bace1', 'esr1', 'esr2', 'fa10', 'fa7', 'fak1', 'fgfr1', 'fkb1a', 'gria2', 'hdac2', 'hdac8', 'hivint', 'hivrt','hmdh', 'hs90a', 'hxk4',
#             'ital', 'jak2', 'kif11', 'kit', 'lck', 'lkha4', 'mapk2', 'met', 'mk01', 'mk14', 'mmp13', 'mp2k1', 'nram', 'parp1', 'pgh1', 'pgh2',
#             'plk1', 'ppara', 'ppard', 'pparg', 'prgr', 'ptn1', 'pygm', 'pyrd', 'thb', 'thrb', 'vgfr2', 'wee1', 'kpcb', 'tgfr1','rock1','nos1',
#             'pde5a','mk10', 'reni' ,'rxra', 'urok'] 
#    targets=['igf1r', 'fnta','inha','src', 
    targets=['esr1']
    for target in targets:
        main(target)
