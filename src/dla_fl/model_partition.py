import torch.nn as nn

class ModelPartition:
    def __init__(self, name, layers):
        self.name = name
        self.layers = layers

    def train_on_device(self, device):
        params = [layer.weight.data.mean().item() for layer in self.layers]
        return {'params': params, 'weight': len(self.layers)}

class ModelPartitioner:
    def __init__(self, cfg):
        self.layers_cfg = cfg['layers']

    def create_partitions(self):
        total = sum(self.layers_cfg)
        dummy = [nn.Linear(10,10) for _ in range(total)]
        parts = []
        idx = 0
        for count in self.layers_cfg:
            parts.append(ModelPartition(f'part_{idx}', dummy[idx:idx+count]))
            idx += count
        return parts