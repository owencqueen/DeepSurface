import os
import warnings

import torch
from torch import nn
import esm

from torchdrug import core, layers, utils, data
from torchdrug.layers import functional
from torchdrug.core import Registry as R

MODEL_BASE_PATH = '/n/data1/hms/dbmi/zitnik/lab/users/yeh803/PLM/raw_data/PEER_Benchmark/scratch/model-weights'


# class MeanReadout(Readout):
#     """Mean readout operator over graphs with variadic sizes."""

#     def forward(self, graph, input):
#         """
#         Perform readout over the graph(s).

#         Parameters:
#             graph (Graph): graph(s)
#             input (Tensor): node representations

#         Returns:
#             Tensor: graph representations
#         """
#         input2graph = self.get_index2graph(graph)
#         output = scatter_mean(input, input2graph, dim=0, dim_size=graph.batch_size)
#         return output

@R.register("models.CustomModel")
class CustomModel(nn.Module, core.Configurable):
    """
    Parameters:
        path (str): path to store ESM model weights
        model (str, optional): model name. Available model names are ``ESM-1b``, ``ESM-1v`` and ``ESM-1b-regression``.
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """
    urls = {
        "ESM-1b": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt",
        "ESM-1b-regression":
            "https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt",
        "ESM-1v": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S.pt",
        "ESM-2-35m": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt",
        "ESM-2-8m": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt",
        "ESM-2": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
        "ESM-2-regression": "https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt",
        "Tranception_Medium": None, 
        "OntoProtein": None,
        "custom": None,
        
    }
    
    locals = {
        "ESM-1b": "/esm1b_t33_650M_UR50S.pt",
        "ESM-1b-regression": "/esm1b_t33_650M_UR50S-contact-regression.pt",
        "ESM-1v": None,
        "ESM-2-35m": "/esm2_t12_35M_UR50D.pt",
        "ESM-2-8m": "/esm2_t6_8M_UR50D.pt",
        "ESM-2": "/esm2_t33_650M_UR50D.pt",
        "ESM-2-regression": "/esm2_t33_650M_UR50D-contact-regression.pt",
        "Tranception_Medium": "/Tranception_Medium/pytorch_model.bin",  # We can also use Huggingface to load them
        "OntoProtein": "/OntoProtein/pytorch_model.bin",
        "custom": None,
        
    }

    output_dim = {
        "ESM-1b": 1280,
        "ESM-1v": 1280,
        "ESM-2-35m": 480,
        "ESM-2-8m": 320,
        "ESM-2": 1280,
        "Tranception_Medium": 1024,
        "OntoProtein": 1024,
        "custom": None,
    }

    repr_layers = {
        'ESM-2-8m': 6,
        'ESM-2-35m': 12,
    }
    
    md5 = {
        "ESM-1b": "ba8914bc3358cae2254ebc8874ee67f6",
        "ESM-1v": "1f04c2d2636b02b544ecb5fbbef8fefd",
        "ESM-1b-regression": "e7fe626dfd516fb6824bd1d30192bdb1",
        "ESM-2": None,
        "ESM-2-8m": None,
        "ESM-2-35m": None,
        "ESM-2-regression": None,
        "Tranception_Medium": None,
        "OntoProtein": None,
        "custon": None,
    }

    max_input_length = 2048 - 2  # Limit of sequence length

    def __init__(self, path=MODEL_BASE_PATH, model="ESM-2", readout="mean"):
        super(CustomModel, self).__init__()
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        
        self.model_name = model
        _model, alphabet = self.load_weight(path)
        mapping = self.construct_mapping(alphabet)
        self.output_dim = self.output_dim[model]
        self.model = _model
        self.alphabet = alphabet
        self.register_buffer("mapping", mapping)

        if readout == "sum":
            self.readout = layers.SumReadout("residue")
        elif readout == "mean":
            self.readout = layers.MeanReadout("residue")
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def load_weight(self, path):
        if self.model_name not in self.urls and not os.path.exists(path+f'/{self.locals[self.model_name]}'):
            raise ValueError("Unknown model `%s`" % self.model_name)
        elif not os.path.exists(path+f'/{self.locals[self.model_name]}'):
            model_file = utils.download(self.urls[self.model_name], path, md5=self.md5[self.model_name])
        else:
            model_file = path+self.locals[self.model_name]
        model_data = torch.load(model_file, map_location="cpu")
        if self.model_name in {"ESM-1b", "ESM-2", "ESM-2-8m", "ESM-2-35m"}:
            #regression_model = "%s-regression" % self.model_name
            regression_file = utils.download(self.urls[self.model_name], path, md5=self.md5[self.model_name])
            regression_data = torch.load(regression_file, map_location="cpu")
        else:
            regression_data = None
        if self.model_name.startswith('ESM'):
            model_name = os.path.basename(self.urls[self.model_name])
            return esm.pretrained.load_model_and_alphabet_core(model_name, model_data, None)
        else:
            return load_model(self.model_name, model_data)

    def construct_mapping(self, alphabet):
        mapping = [0] * len(data.Protein.id2residue_symbol)
        if self.model_name.startswith('ESM'):
            for i, token in data.Protein.id2residue_symbol.items():
                mapping[i] = alphabet.get_idx(token)
        elif self.model_name.startswith('OntoProtein'):  # TODO
            pass
        elif self.model_name.startswith('Tranception'):  # TODO
            pass
        mapping = torch.tensor(mapping)
        return mapping
    
    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the residue representations and the graph (protein) representation.

        Parameters:
            graph (Protein): :math:`n` protein(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``residue_feature`` and ``graph_feature`` fields:
                residue representations of shape :math:`(|V_{res}|, d)`, graph representations of shape :math:`(n, d)`
        """
        input = graph.residue_type
        input = self.mapping[input]
        size = graph.num_residues
        if (size > self.max_input_length).any():
            warnings.warn("ESM can only encode proteins within %d residues. Truncate the input to fit into ESM."
                          % self.max_input_length)
            starts = size.cumsum(0) - size
            size = size.clamp(max=self.max_input_length)
            ends = starts + size
            mask = functional.multi_slice_mask(starts, ends, graph.num_residue)
            input = input[mask]
            graph = graph.subresidue(mask)
        size_ext = size
        
        if self.alphabet.prepend_bos:
            bos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.alphabet.cls_idx
            input, size_ext = functional._extend(bos, torch.ones_like(size_ext), input, size_ext)
        if self.alphabet.append_eos:
            eos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.alphabet.eos_idx
            input, size_ext = functional._extend(input, size_ext, eos, torch.ones_like(size_ext))
        input = functional.variadic_to_padded(input, size_ext, value=self.alphabet.padding_idx)[0]
        output = self.model(input, repr_layers=[self.repr_layers[self.model_name]])
        residue_feature = output["representations"][self.repr_layers[self.model_name]]

        graph_feature = residue_feature[:,0,:]

        residue_feature = functional.padded_to_variadic(residue_feature, size_ext)
        starts = size_ext.cumsum(0) - size_ext
        if self.alphabet.prepend_bos:
            starts = starts + 1
        ends = starts + size
        mask = functional.multi_slice_mask(starts, ends, len(residue_feature))
        residue_feature = residue_feature[mask]
        #graph_feature = self.readout(graph, residue_feature)

        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature
        }