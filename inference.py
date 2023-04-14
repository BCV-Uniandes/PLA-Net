import torch
from torch_geometric.data import DataLoader
from model.model_concatenation import PLANet
from model.model import DeeperGCN
from tqdm import tqdm
from utils.args import ArgsInit
import logging
import pandas as pd
import time
from data.dataset import load_dataset
import torch.nn.functional as F
import numpy as np 
import os 
import torch.optim as optim

def load_model(model, fold, args):
    model_name = os.path.join(args.inference_path, 'BINARY_'+args.target, f'Fold{fold}','Best_Model.pth')
        
    pre_model = torch.load(model_name,
        map_location=lambda storage, loc: storage)
    model.load_state_dict(pre_model['model_state_dict'])

    # for k, v in pre_model['model_state_dict'].items():
    #         if v.shape == model.molecule_gcn.state_dict()[k].shape:
    #             model_weights[k] = v
    #         else:
    #             model_weights[k] = torch.transpose(v, 0, 1)
    # model.load_state_dict(model_weights)

    return model   

@torch.no_grad()
def test_gcn(model, device, loader,args):
    first = True
    
    for batch in tqdm(loader, desc="Iteration"):
        save_dict = {'Target': [],
                 'Smiles': [],
                 'Probability of Interaction': [],
                 'Class Id': []}
        save_dict_temp = {
                 'Folder 1': [],
                 'Folder 2': [],
                 'Folder 3': [],
                 'Folder 4': []}

        if args.use_prot:
            batch_mol = batch[0].to(device)
            batch_prot = batch[1].to(device)
            smiles = batch_mol['smiles']
            smiles = [smi for smi in smiles]
        else:
            batch_mol = batch[0].to(device)
            smiles = batch_mol['y']
            smiles = [smi for smi in smiles]
            
        if args.feature == 'full':
            pass
        elif args.feature == 'simple':
            # only retain the top two node/edge features
            num_features = args.num_features
            batch_mol.x = batch_mol.x[:, :num_features]
            batch_mol.edge_attr = batch_mol.edge_attr[:, :num_features]
        if batch_mol.x.shape[0] == 1:
            pass
        else:

            target = [args.target]*len(batch[0].y)
            save_dict['Target'].extend(target)
            save_dict['Smiles'].extend(smiles)
            for fold in range(1,5):
                model = load_model(model, fold, args)
                model.eval()

                with torch.set_grad_enabled(False):   
                    if args.use_prot:
                        pred = model(batch_mol,batch_prot)
                    else:
                        pred = model(batch_mol)
                    pred = F.softmax(pred,dim=1)
                    save_dict_temp[f'Folder {fold}'].extend(pred.cpu().tolist()) 
            for fold in range(1,5):
                save_dict_temp[f'Folder {fold}'] = np.array(save_dict_temp[f'Folder {fold}'])

            save_dict['Probability of Interaction'] = np.mean([save_dict_temp['Folder 1'], save_dict_temp['Folder 2'], save_dict_temp['Folder 3'], save_dict_temp['Folder 4']], axis = 0).tolist()
            save_dict['Class Id'] = [int(np.argmax(i)) for i in save_dict['Probability of Interaction']]
            save_dict['Probability of Interaction'] = [x[1] for x in save_dict['Probability of Interaction']]
            for fold in range(1,5):
                save_dict_temp[f'Folder {fold}'] = save_dict_temp[f'Folder {fold}'].tolist()
            
            save_df = pd.DataFrame(save_dict)

            save_path = os.path.join(args.save_path, f'Inference.csv')
            if first == 0:
                save_df.to_csv(save_path, index=False)
                first = False
            else:
                save_df.to_csv(save_path, mode='a', header=False, index= False)
    

def main(args):


    args.save_path = args.inference_path
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')
    
    if args.binary:
        args.nclasses = 2
    
    #Numpy and torch seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    logging.info('%s' % args)   

    ( _,_,test_dataset,_,_,_,) = load_dataset(
        cross_val=args.cross_val,
        binary_task=args.binary,
        target=args.target,
        use_prot=args.use_prot,
        args=args,
        inference=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    
    if args.use_prot:
        model = PLANet(args).to(device)
    else:
        model = DeeperGCN(args).to(device)

    logging.info('Model inference in: {}'.format(args.inference_path))
    start_time = time.time()

    #Load pre-trained molecule model

    logging.info('Evaluating...')
    test_gcn(model, device, test_loader, args)


    end_time = time.time()
    total_time = end_time - start_time
    logging.info('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))


if __name__ == "__main__":
    cls_criterion = torch.nn.BCELoss()
    reg_criterion = torch.nn.MSELoss()

    args = ArgsInit().args

    main(args)
