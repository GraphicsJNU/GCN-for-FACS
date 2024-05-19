import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from Data.facs_dataset import FACSDataset
from Models.gcn_facs import GCNFACS
from Utils.utils import save_model_from_removed_vertex, recon_face_model


#train function    
def train(n_epochs, start_epochs, model, optimizer, scheduler, train_loader, model_save_path):
    
    model.train()
    for i in range(start_epochs, n_epochs+1):
        total_error = 0
        cnt = 0
        progressbar = tqdm(train_loader)
        for data in progressbar:
            data = data.to(device)
            optimizer.zero_grad()
            
            pred = model(data)
            loss = F.mse_loss(pred, data.y.reshape(-1, 53)) 
            
            loss.backward()
            optimizer.step()
            
            total_error = total_error + loss.item()
            cnt = cnt + 1
            progressbar.set_description(f'train {i} loss: {total_error/cnt:.8f}')
            
        scheduler.step()
        
        if i % 10 == 0:
            # torch.save(model.state_dict(), osp.join(model_save_path, f'epoch_{i}.pt'))
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                },
                 osp.join(model_save_path, f'epoch_{i}.pt')
            )
        

if __name__=='__main__':
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", type=int, default=0, help="evaluate the model")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size for training")
    parser.add_argument("--input_id", type=int, default=30, help="input model identity for evaluation")
    parser.add_argument("--input_exp", type=int, default=1, help="input model expression for evaluation")
    parser.add_argument("--output_model", type=int, default=0, help="output model for evaluation")
    args = parser.parse_args()
    
    #device settings
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    #training settings
    is_train = not args.evaluate
    n_epochs = args.epochs
    batch_size = args.batch_size
    lr = 1e-3
    decay_step = 10
    decay_rate = 0.5
    
    #paths
    base_path = osp.dirname(osp.abspath(__file__))
    model_save_path = osp.join(base_path, 'Data/SavedModels')
    dataset_path = osp.join(base_path, 'Data/FACS')
    
    #Load  datasets
    train_dataset = FACSDataset(root=dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    
    #model
    model = GCNFACS(v_cnt=11248, in_dim=3, out_dim=53, K=4, ttype=dtype,
                    normalization="Group Norm", activation="ReLU", bias=True).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_rate)
    
    #load pretrained model
    # start_epoch = 0
    # for epoch in range(n_epochs, 0, -1):
    #     saved_model= osp.join(model_save_path, f'epoch_{epoch}.pt')
    #     if osp.exists(saved_model):
    #         print(f'load pretrained model: {epoch}')
    #         model.load_state_dict(torch.load(saved_model))
    #         start_epoch = epoch
    #         break
        
    start_epoch = 0
    for epoch in range(n_epochs, 0, -1):
        saved_model = osp.join(model_save_path, f'epoch_{epoch}.pt')
        if osp.exists(saved_model):
            print(f'load pretrained model: {epoch}')
            start_epoch = epoch
            saved_data = torch.load(saved_model)
            model.load_state_dict(saved_data['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(saved_data['optimizer_state_dict'])
            if scheduler is not None:
                scheduler.load_state_dict(saved_data['scheduler_state_dict'])
            break
    
    if start_epoch//10 < n_epochs//10 and is_train:
        train(n_epochs, start_epoch, model, optimizer, scheduler, train_loader, model_save_path)
    
    model.eval()
    with torch.no_grad():
        id_tester_num = args.output_model
        exp_tester_num, exp_shape_num = args.input_id, args.input_exp
        id_path = '{}/Data/FACS/raw/sample_identity_{}_0.obj'.format(base_path, id_tester_num)
        exp_path = '{}/Data/FACS/raw/sample_identity_{}_{}.obj'.format(base_path, exp_tester_num, exp_shape_num)
        facs_path = '{}/Data/FACS/raw/sample_coeffs_{}_0.json'.format(base_path, id_tester_num)
        save_path = '{}/Data/results/'.format(base_path)
        save_id_origin_name = '{}_0_origin.obj'.format(id_tester_num)
        save_exp_origin_name = '{}_{}_origin.obj'.format(exp_tester_num, exp_shape_num)
        save_file_name = '{}-{}_{}_{}.obj'.format(id_tester_num, exp_tester_num, exp_shape_num, n_epochs)

        recon_face_model(model, device, train_dataset, exp_tester_num, exp_shape_num, facs_path, save_path, save_file_name)
        save_model_from_removed_vertex(osp.join(save_path, save_file_name), osp.join(save_path, save_file_name))
    
    print('done')