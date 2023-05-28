import torch
import numpy as np
import umap


def main_nofix():
    f, L = torch.load('esm35m_0.pt')

    print(f.shape)
    #exit()
    fnp = f.numpy()

    fumap = umap.UMAP(verbose = True).fit_transform(fnp)

    torch.save((fumap, L), 'esm35m_umap.pt')

def main_fix():
    f, L = torch.load('esm35m_fixed_0.pt')

    print(f.shape)

    fnp = f.numpy()

    fumap = umap.UMAP(verbose = True).fit_transform(fnp)

    torch.save((fumap, L), 'esm35m_fixed_umap.pt')

if __name__ == '__main__':
    main_fix()