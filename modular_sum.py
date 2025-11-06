import torch
import numpy as np
import einops
import torch.nn.functional as F

def modular_sum(a,b,freqs,p=113,N=8,D=128, *, out_mags=None, out_phases=None, inout_norms=(3.19, 0.101), get_tensors=False):
    '''
        This computes modular addition: a + b % p = c and returns c
        freqs: (F,) key frequencies to use that make up the sinusoidal content in the activations
        out_mags/out_phases: (D,F) these represent the spectrum of the desired (~W_U unembed matrix) output
        inout_norms: these normalize the final residual stream and the unembed to Neel Nanda's original norms
        get_tensor: returns all intermediate tensors as well as the answer, c
    '''
    D = out_mags.size(0) if out_mags is not None else out_phases.size(0) if out_phases is not None else D
    if out_mags is None: out_mags = 1 + torch.randn([D, len(freqs)])/ len(freqs)
    if out_phases is None: out_phases = 1 + torch.arange(D)[...,None].expand(D, len(freqs)) * freqs[None] * 2 * np.pi / D
    wk, neuron_phases = torch.arange(p) * 2 * np.pi / p, torch.arange(N) * 2 * np.pi / N
    harmonic1 = torch.cos(freqs[...,None,None] * wk + np.pi + neuron_phases[...,None]) # (F,N,p)
    harmonic2 = torch.cos(2 * freqs[...,None,None] * wk + 2 * neuron_phases[...,None]) # (F,N,p)
    vec = (harmonic1 - harmonic2/8) # (F,N,p) THIS 1/8 MAGNITUDE HAS BEEN EMPIRICALLY FOUND WITHOUT PRINCIPLED THEORY
    a_vec, b_vec = vec[...,None], vec[...,None,:] # (F,N,p,1), (F,N,1,p)
    pre = a_vec + b_vec + np.sqrt(2) - 1 # (F,N,p,p) THE sqrt(2) - 1 BIAS TERM HAS ALSO BEEN EMPIRICALLY FOUND
    weights = (torch.cos(out_phases - 2 * neuron_phases[...,None,None]) * out_mags).permute(2,0,1) # (F,N,D)
    output = einops.einsum(F.relu(pre), weights, "freqs neur a b, freqs neur d_model -> d_model a b") # (D,p,p)
    desired = (out_mags[...,None] * torch.cos(freqs[...,None] * wk + out_phases[...,None])).sum(1) # (D,p)
    output *= inout_norms[0]/output.square().mean(-1).sqrt().mean()
    desired *= inout_norms[1]/desired.square().mean(-1).sqrt().mean()
    logits = einops.einsum(output, desired, "d_model a b, d_model c -> a b c") # (p,p,p)
    answers = torch.argmax(logits, -1) # (p,p)
    c = answers[a,b].item()
    return (c, a_vec, b_vec, weights, desired, output, logits.flatten(0,1), answers) if get_tensors else c
