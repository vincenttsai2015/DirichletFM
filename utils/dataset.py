import copy
import pickle
import torch, esm, random, os, json
import numpy as np
from Bio import SeqIO
from urllib.request import urlretrieve
from utils.base import register_dataset

@register_dataset('bmnist')
class BinaryMNIST(torch.utils.data.Dataset):
    """
    Binarized MNIST dataset.
    """
    data_url = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat"

    def __init__(self, root, split, with_labels=True, 
                 labels_root='data', val_from_train=5000):
        super().__init__()
        self.root = root
        self.split = split
        self.alphabet_size = 2
        self.num_cls = 10

        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, f"binarized_mnist_{split}.amat")
        if not os.path.exists(path):
            print(f"Downloading {split} set...")
            urlretrieve(self.data_url.format(split), path)

        # data: float32 0/1, shape (N, 784)
        data = np.loadtxt(path).astype(np.float32)
        # turn into tokens: int64 0/1, shape (N, 784)
        self.seq = torch.from_numpy(data).round().to(torch.long)

        self.targets = None
        if with_labels:
            from torchvision.datasets import MNIST
            labels_root = labels_root or root
            mnist_train = MNIST(labels_root, train=True, download=True)
            mnist_test = MNIST(labels_root, train=False, download=True)

            if split == "train":
                targets = mnist_train.targets[:-val_from_train]
            elif split == "valid":
                targets = mnist_train.targets[-val_from_train:]
            elif split == "test":
                targets = mnist_test.targets
            else:
                raise ValueError(split)

            self.targets = targets.to(torch.long)

            if len(self.targets) != len(self.seq):
                raise RuntimeError(
                    f"Label length mismatch split={split}: data={len(self.seq)} vs labels={len(self.targets)}. "
                    f"Your split alignment assumption may be wrong."
                )

    def __len__(self):
        return self.seq.size(0)

    def __getitem__(self, idx):
        seq = self.seq[idx]            # (784,) long in {0,1}
        if self.targets is None:
            # 如果你真的沒 label，先給 dummy label（見路線2），但這只會讓 loss 沒意義
            cls = torch.tensor(0, dtype=torch.long)
        else:
            cls = self.targets[idx]    # scalar long 0..9
        return seq, cls

class EnhancerDataset(torch.utils.data.Dataset):
    def __init__(self, args, split='train'):
        all_data = pickle.load(open(f'data/the_code/General/data/Deep{"MEL2" if args.mel_enhancer else "FlyBrain"}_data.pkl', 'rb'))
        self.seqs = torch.argmax(torch.from_numpy(copy.deepcopy(all_data[f'{split}_data'])), dim=-1)
        self.clss = torch.argmax(torch.from_numpy(copy.deepcopy(all_data[f'y_{split}'])), dim=-1)
        self.num_cls = all_data[f'y_{split}'].shape[-1]
        self.alphabet_size = 4

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.clss[idx]


class TwoClassOverfitDataset(torch.utils.data.IterableDataset):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.toy_seq_len
        self.alphabet_size = args.toy_simplex_dim
        self.num_cls = 2

        if args.cls_ckpt is not None:
            distribution_dict = torch.load(os.path.join(os.path.dirname(args.cls_ckpt), 'overfit_dataset.pt'))
            self.data_class1 = distribution_dict['data_class1']
            self.data_class2 = distribution_dict['data_class2']
        else:
            self.data_class1 = torch.stack([torch.from_numpy(np.random.choice(np.arange(self.alphabet_size), size=args.toy_seq_len, replace=True)) for _ in range(args.toy_num_seq)])
            self.data_class2 = torch.stack([torch.from_numpy(np.random.choice(np.arange(self.alphabet_size), size=args.toy_seq_len, replace=True)) for _ in range(args.toy_num_seq)])
            distribution_dict = {'data_class1': self.data_class1, 'data_class2': self.data_class2}
        torch.save(distribution_dict, os.path.join(os.environ["MODEL_DIR"], 'overfit_dataset.pt'))

    def __len__(self):
        return 10000000000

    def __iter__(self):
        while True:
            if np.random.rand() < 0.5:
                yield self.data_class1[np.random.choice(np.arange(len(self.data_class1)))], torch.tensor([0])
            else:
                yield self.data_class2[np.random.choice(np.arange(len(self.data_class2)))], torch.tensor([1])

class ToyDataset(torch.utils.data.IterableDataset):
    def __init__(self, args):
        super().__init__()
        self.num_cls = args.toy_num_cls
        self.seq_len = args.toy_seq_len
        self.alphabet_size = args.toy_simplex_dim

        if args.cls_ckpt is not None:
            distribution_dict = torch.load(os.path.join(os.path.dirname(args.cls_ckpt), 'toy_distribution_dict.pt'))
            self.probs = distribution_dict['probs']
            self.class_probs = distribution_dict['class_probs']
        else:
            self.probs = torch.softmax(torch.rand((self.num_cls, self.seq_len, self.alphabet_size)), dim=2)
            self.class_probs = torch.ones(self.num_cls)
            if self.num_cls > 1:
                self.class_probs = self.class_probs * 1 / 2 / (self.num_cls - 1)
                self.class_probs[0] = 1 / 2
            assert self.class_probs.sum() == 1

            distribution_dict = {'probs': self.probs, 'class_probs': self.class_probs}
        torch.save(distribution_dict, os.path.join(os.environ["MODEL_DIR"], 'toy_distribution_dict.pt' ))

    def __len__(self):
        return 10000000000
    def __iter__(self):
        while True:
            cls = np.random.choice(a=self.num_cls,size=1,p=self.class_probs)
            seq = []
            for i in range(self.seq_len):
                seq.append(torch.multinomial(replacement=True,num_samples=1,input=self.probs[cls,i,:]))
            yield torch.tensor(seq), cls

