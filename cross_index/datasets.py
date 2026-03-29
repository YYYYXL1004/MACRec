import numpy as np
import torch
import torch.utils.data as data
import os
from sklearn.preprocessing import normalize as l2_normalize

class EmbDataset(data.Dataset):

    def __init__(self,data_path):

        self.data_path = data_path
        self.embeddings = np.load(data_path)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
    
class EmbDatasetAll(data.Dataset):

    def __init__(self, args):

        self.datasets = args.datasets.split(',')
        embeddings = []
        self.dataset_count = []
        for dataset in self.datasets:
            print(dataset)
            embedding_path = os.path.join(args.data_root, dataset, f'{dataset}{args.embedding_file}')
            embedding = np.load(embedding_path)
            embeddings.append(embedding)
            self.dataset_count.append(embedding.shape[0])
            
        self.embeddings = np.concatenate(embeddings)
        self.dim = self.embeddings.shape[-1]
        
        print(self.dataset_count)
        print(self.embeddings.shape[0])

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
    
class EmbDatasetOne(data.Dataset):

    def __init__(self, args, dataset):


        print(dataset)
        embedding_path = os.path.join(args.data_root, dataset, f'{dataset}{args.embedding_file}')
        self.embedding = np.load(embedding_path)

        self.dim = self.embedding.shape[-1]
        
        self.data_count = self.embedding.shape[0]

        print(self.embedding.shape)

    def __getitem__(self, index):
        emb = self.embedding[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embedding)


class DualEmbDataset(data.Dataset):
    
    def __init__(self, text_data_path, img_data_path):
        self.text_data_path = text_data_path
        self.img_data_path = img_data_path
        
        self.text_embeddings = np.load(text_data_path)
        self._text_dim = self.text_embeddings.shape[-1]
        
        self.img_embeddings = np.load(img_data_path)
        self._img_dim = self.img_embeddings.shape[-1]
        
        assert len(self.text_embeddings) == len(self.img_embeddings), \
            f"Text and image data must have the same length. Text: {len(self.text_embeddings)}, Image: {len(self.img_embeddings)}"
        
        print(f"Loaded dual-modal dataset:")
        print(f"  Text embeddings: {self.text_embeddings.shape} (dim: {self._text_dim})")
        print(f"  Image embeddings: {self.img_embeddings.shape} (dim: {self._img_dim})")
        print(f"  Total samples: {len(self.text_embeddings)}")

    def __getitem__(self, index):
        text_emb = self.text_embeddings[index]
        img_emb = self.img_embeddings[index]
        
        text_tensor = torch.FloatTensor(text_emb)   
        img_tensor = torch.FloatTensor(img_emb)
        
        return text_tensor, img_tensor, index

    def __len__(self):
        return len(self.text_embeddings)
    
    @property
    def text_dim(self):
        return self._text_dim
    
    @property
    def img_dim(self):
        return self._img_dim


class TripleEmbDataset(data.Dataset):
    """
    三路向量数据集: text + image + collab
    
    加载后对每路做 L2 归一化，然后将 collab 分别拼接到 text 和 image，
    返回 (text_concat, image_concat, index)
    """
    
    def __init__(self, text_data_path, img_data_path, collab_data_path, collab_scale=1.0):
        # 加载原始向量
        text_raw = np.load(text_data_path)
        img_raw = np.load(img_data_path)
        collab_raw = np.load(collab_data_path)
        
        assert len(text_raw) == len(img_raw) == len(collab_raw), \
            f"三路向量数量必须一致: text={len(text_raw)}, image={len(img_raw)}, collab={len(collab_raw)}"
        
        # L2 归一化
        text_norm = l2_normalize(text_raw, norm='l2', axis=1)
        img_norm = l2_normalize(img_raw, norm='l2', axis=1)
        collab_norm = l2_normalize(collab_raw, norm='l2', axis=1)
        
        # 缩放 collab 能量：α<1 时降低协同信号对 RQVAE 输入的影响力
        collab_scaled = collab_norm * collab_scale
        
        # 拼接: text+collab, image+collab
        self.text_concat = np.concatenate([text_norm, collab_scaled], axis=1).astype(np.float32)
        self.img_concat = np.concatenate([img_norm, collab_scaled], axis=1).astype(np.float32)
        
        self._text_dim = self.text_concat.shape[-1]
        self._img_dim = self.img_concat.shape[-1]
        
        print(f"TripleEmbDataset 已加载 (collab_scale={collab_scale}):")
        print(f"  text  原始: {text_raw.shape} → 归一化+拼接collab → {self.text_concat.shape}")
        print(f"  image 原始: {img_raw.shape} → 归一化+拼接collab → {self.img_concat.shape}")
        print(f"  collab 原始: {collab_raw.shape}, scale={collab_scale}, 能量占比={collab_scale**2/(1+collab_scale**2)*100:.1f}%")
        print(f"  样本数: {len(self.text_concat)}")

    def __getitem__(self, index):
        text_tensor = torch.FloatTensor(self.text_concat[index])
        img_tensor = torch.FloatTensor(self.img_concat[index])
        return text_tensor, img_tensor, index

    def __len__(self):
        return len(self.text_concat)
    
    @property
    def text_dim(self):
        return self._text_dim
    
    @property
    def img_dim(self):
        return self._img_dim