import torch
import numpy as np
import einops
import torch.nn.functional as F

d_model = 128
p = 113
prange = torch.arange(p)

def modular_sum(a,b,freqs, *, out_mags=None, out_phases=None, inout_norms=(3.19, 0.101), get_tensors=False):
    if out_mags is None: out_mags = torch.randn([d_model, len(freqs)]) / np.sqrt(p/2)
    if out_phases is None: out_phases = torch.rand([d_model, len(freqs)]) * 2 * np.pi - np.pi
    desired = (out_mags[...,None] * torch.cos(prange * freqs[...,None] * 2 * np.pi / p + out_phases[...,None])).sum(1)
    phzrange = torch.arange(d_model) * 2 * np.pi / d_model
    harmonic1 = torch.cos(prange * freqs[...,None,None] * 2 * np.pi / p + phzrange[...,None] - np.pi)
    harmonic2 = torch.cos(prange * 2 * freqs[...,None,None] * 2 * np.pi / p + 2 * phzrange[...,None] - np.pi)
    vec = (harmonic1 + harmonic2/8) # THIS 1/8 MAGNITUDE RATIO HAS BEEN EMPIRICALLY FOUND WITHOUT PRINCIPLED THEORY
    a_vec, b_vec = vec[...,None], vec[...,None,:]
    weights = (torch.cos(out_phases - 2 * phzrange[...,None,None]) * out_mags).permute(2,0,1)
    ## THE sqrt(2) - 1 BIAS TERM BELOW HAS ALSO BEEN EMPIRICALLY FOUND
    output = einops.einsum(F.relu(a_vec + b_vec + np.sqrt(2) - 1), weights, "freqs d1 ..., freqs d1 d2 -> d2 ...")
    output *= inout_norms[0]/output.square().mean(-1).sqrt().mean()
    desired *= inout_norms[1]/desired.square().mean(-1).sqrt().mean()
    logits = einops.einsum(output, desired, "d_model p1 p2, d_model p3 -> p1 p2 p3")
    answers = torch.argmax(logits, -1)
    logits = logits.flatten(0,1)
    c = answers[a,b].item()
    return (c, a_vec, b_vec, weights, desired, output, logits, answers) if get_tensors else c
