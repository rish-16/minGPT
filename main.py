import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision

from model import *
from utils import *
from trainer import *

"""
Credits to Andrej Karpathy for his min-GPT implementation. This repo
tests the AFT-Full (from the paper, "An Attention Free Transformer" by Zhai et al.)
implementation used within the Image GPT to see if loss decreases.
"""

def prepare_dataset():
    pluck_rgb = lambda x: torch.from_numpy(np.array(x)).view(32*32, 3)[torch.randperm(32*32)[:5], :]
    px = torch.cat([pluck_rgb(x) for x, y in train_data], dim=0).float()
    print(px.size())

    def kmeans(x, ncluster, niter=10):
        N, D = x.size()
        c = x[torch.randperm(N)[:ncluster]] # init clusters at random
        for i in range(niter):
            # assign all pixels to the closest codebook element
            a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
            # move each codebook element to be the mean of the pixels that assigned to it
            c = torch.stack([x[a==k].mean(0) for k in range(ncluster)])
            # re-assign any poorly positioned codebook elements
            nanix = torch.any(torch.isnan(c), dim=1)
            ndead = nanix.sum().item()
            print('done step %d/%d, re-initialized %d dead clusters' % (i+1, niter, ndead))
            c[nanix] = x[torch.randperm(N)[:ndead]] # re-init dead clusters
        return c

    ncluster = 512
    with torch.no_grad():
        C = kmeans(px, ncluster, niter=8)

    print(C.size())
        
    root = './'
    train_data = torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=True)
    test_data  = torchvision.datasets.CIFAR10(root, train=False, transform=None, target_transform=None, download=True)
    print(len(train_data), len(test_data))    

    train_dataset = ImageDataset(train_data, C)
    test_dataset = ImageDataset(test_data, C)
    
    return train_dataset, train_data, test_dataset, test_data, C

def get_model(train_dataset, train_data, test_dataset, test_data):
    mconf = GPTConfig(
                train_dataset.vocab_size, 
                train_dataset.block_size,
                embd_pdrop=0.0,
                resid_pdrop=0.0, 
                attn_pdrop=0.0,
                n_layer=12, 
                n_head=8, 
                n_embd=256
            )
    
    model = GPT(mconf)

    tconf = TrainerConfig(max_epochs=10, batch_size=256)
    trainer = Trainer(model, train_dataset, test_dataset, tconf)
    trainer.train()

    tokens_per_epoch = len(train_dataset) * train_dataset.block_size
    train_epochs = 20 # todo run a bigger model and longer, this is tiny            

    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(max_epochs=train_epochs, batch_size=16*8, learning_rate=3e-3,
                        betas = (0.9, 0.95), weight_decay=0,
                        lr_decay=True, warmup_tokens=tokens_per_epoch, final_tokens=train_epochs*tokens_per_epoch,
                        ckpt_path='cifar10_model.pt',
                        num_workers=8)
    trainer = Trainer(model, train_dataset, test_dataset, tconf)
    trainer.train()

def test_model(trainer, model, ncluster, train_dataset, test_dataset, C):
    checkpoint = torch.load('cifar10_model.pt')
    model.load_state_dict(checkpoint)

    counts = torch.ones(ncluster) # start counts as 1 not zero, this is called "smoothing"
    rp = torch.randperm(len(train_dataset))
    nest = 5000 # how many images to use for the estimation
    for i in range(nest):
        a, _ = train_dataset[int(rp[i])]
        t = a[0].item() # index of first token in the sequence
        counts[t] += 1
    prob = counts/counts.sum()

    n_samples = 32
    start_pixel = np.random.choice(np.arange(C.size(0)), size=(n_samples, 1), replace=True, p=prob)
    start_pixel = torch.from_numpy(start_pixel).to(trainer.device)
    pixels = sample(model, start_pixel, 32*32-1, temperature=1.0, sample=True, top_k=100)

    iperm = torch.argsort(train_dataset.perm)

    ncol = 8
    nrow = n_samples // ncol
    plt.figure(figsize=(16, 8))

    for i in range(n_samples):
        pxi = pixels[i][iperm] # note: undo the encoding permutation
        
        plt.subplot(nrow, ncol, i+1)
        plt.imshow(C[pxi].view(32, 32, 3).numpy().astype(np.uint8))
        plt.axis('off')
    plt.show()