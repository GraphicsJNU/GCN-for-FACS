import os.path as osp
# import openmesh as om
import torch
from torch_geometric.data import Data, Dataset
from pytorch3d.io import load_obj

from Utils.utils import load_facs, load_data, get_vertices_edge

class FACSDataset(Dataset):
    def __init__(self, root):
        super(FACSDataset, self).__init__(root)
        
    
    @property
    def raw_file_names(self):
        raw_file_name = []
        for i in range(100):
            raw_file_name.append(f'sample_identity_{i}_0.obj')
            for j in range(1, 3):
                raw_file_name.append(f'sample_identity_{i}_{j}.obj')
        return raw_file_name
    
    @property
    def processed_file_names(self):
        processed_file_name = []
        
        for i in range(300):
            processed_file_name.append(f'testing{i}.pt')

        return processed_file_name
    
    def process(self):
        idx=0
        for file in self.raw_paths:
            self._process_one_step(file, idx)
            idx = idx + 1
            
            
    def _process_one_step(self, path, idx):
        obj_dict = load_data(osp.join(self.raw_dir, path))
        pos = torch.tensor(obj_dict["verts"], dtype=torch.float)
        edge_index = get_vertices_edge(obj_dict["faces"])
        
        facs_path = path.replace('.obj', '.json')
        facs_path = facs_path.replace('identity', 'coeffs')
        facs = load_facs(facs_path)
        
        data = Data(x=pos, edge_index=edge_index, y=facs["expression_coefficients"]) 
        
        out_path = osp.join(self.processed_dir, f'testing{idx}.pt')
        torch.save(data, out_path)
    
    def len(self):
        return len(self.raw_file_names)
    
    def get(self, idx):
        return torch.load(osp.join(self.processed_dir, self.processed_file_names[idx]))