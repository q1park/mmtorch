import torch

def one_hot(y, size):
    y_onehot = torch.FloatTensor(y.size(0), size)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1).type_as(y)
    return y_onehot

class GroupDataset(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = features
        self.label_list = [x.label for x in features]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def get_labels(self):
        return self.label_list

class GroupDataCollator:
    def __init__(self, config):
        self.config = config
        
    def _tensorize(self, name, features, dtype):
        return torch.tensor([f.__dict__[name] for f in features], dtype=dtype).unsqueeze(-1)
        
    def __call__(self, features):
        first = features[0]
        batch = {}

        for k,v in first.__dict__.items():
            if isinstance(v, str) or v is None:
                pass
            elif k=='label' or k=='seq_mask':
                batch[k] = self._tensorize(k, features, torch.long)
            elif self.config[k].vector=='embedding':
                batch[k] = self._tensorize(k, features, torch.long)
            elif self.config[k].vector=='onehot':
                batch[k] = one_hot(self._tensorize(k, features, torch.long), self.config[k].size).unsqueeze(1)
            elif self.config[k].vector=='linear':
                batch[k] = self._tensorize(k, features, torch.float).unsqueeze(-1)
            else:
                raise ValueError("unknown vectorization scheme {}".format(self.config[k].vector))
                
            if type(v)==list:
                batch[k] = batch[k].squeeze(-1)
        return batch