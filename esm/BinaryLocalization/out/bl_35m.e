
CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
To initialize your shell, run

    $ conda init <SHELL_NAME>

Currently supported shells are:
  - bash
  - fish
  - tcsh
  - xonsh
  - zsh
  - powershell

See 'conda init --help' for more information and options.

IMPORTANT: You may need to close and restart your shell after running 'conda init'.



  warnings.warn("Unknown residue symbol `%s`. Treat as glycine" % residue)

  warnings.warn("Unknown residue symbol `%s`. Treat as glycine" % residue)

  warnings.warn("Unknown residue symbol `%s`. Treat as glycine" % residue)

  warnings.warn("Unknown residue symbol `%s`. Treat as glycine" % residue)

/lustre/isaac/scratch/oqueen/codonbert/lib/python3.8/site-packages/esm/pretrained.py:215: UserWarning: Regression weights not found, predicting contacts will not produce correct results.
  warnings.warn(
Traceback (most recent call last):
  File "ESM-binloc.py", line 73, in <module>
    solver.train(num_epoch=10)
  File "/lustre/isaac/scratch/oqueen/codonbert/lib/python3.8/site-packages/torchdrug/core/engine.py", line 161, in train
    loss, metric = model(batch)
  File "/lustre/isaac/scratch/oqueen/codonbert/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/lustre/isaac/scratch/oqueen/codonbert/lib/python3.8/site-packages/torchdrug/tasks/property_prediction.py", line 102, in forward
    pred = self.predict(batch, all_loss, metric)
  File "/lustre/isaac/scratch/oqueen/codonbert/lib/python3.8/site-packages/torchdrug/tasks/property_prediction.py", line 140, in predict
    output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
  File "/lustre/isaac/scratch/oqueen/codonbert/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/lustre/isaac/scratch/oqueen/DeepSurface/esm/torchdrug_esm.py", line 190, in forward
    output = self.model(input, repr_layers=[self.repr_layers[self.model_name]])
  File "/lustre/isaac/scratch/oqueen/codonbert/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/lustre/isaac/scratch/oqueen/codonbert/lib/python3.8/site-packages/esm/model/esm2.py", line 112, in forward
    x, attn = layer(
  File "/lustre/isaac/scratch/oqueen/codonbert/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/lustre/isaac/scratch/oqueen/codonbert/lib/python3.8/site-packages/esm/modules.py", line 125, in forward
    x, attn = self.self_attn(
  File "/lustre/isaac/scratch/oqueen/codonbert/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/lustre/isaac/scratch/oqueen/codonbert/lib/python3.8/site-packages/esm/multihead_attention.py", line 371, in forward
    attn_weights = attn_weights.masked_fill(
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.39 GiB (GPU 0; 44.43 GiB total capacity; 40.64 GiB already allocated; 1.93 GiB free; 41.38 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF