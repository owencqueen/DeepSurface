from torchdrug import datasets, transforms

data_path = '/lustre/isaac/scratch/ababjac/DeepSurface/data/'

truncate_transform = transforms.TruncateProtein(max_length=1024, random=False)
protein_view_transform = transforms.ProteinView(view='residue')
transform = transforms.Compose([truncate_transform, protein_view_transform])

dataset = datasets.BetaLactamase(data_path, atom_feature=None, bond_feature=None, residue_feature='default', transform=transform)
