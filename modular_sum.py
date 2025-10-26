import torch
import numpy as np
import einops
import torch.nn.functional as F

def modular_sum(a,b,freqs,p=113,N=8,D=128, *, mags=None, phases=None, inout_norms=(3.19, 0.101), get_tensors=False):
    '''
        This computes modular addition: a + b % p = c and returns c
        freqs: (F,) key frequencies to use that make up the sinusoidal content in the activations
        mags/phases: (D,F) these represent the spectrum of the desired (~W_U unembed matrix) output
        inout_norms: these normalize the final residual stream and the unembed to Neel Nanda's original norms
        get_tensor: returns all intermediate tensors as well as the answer, c
    '''
    prange = torch.arange(p)
    if mags is not None or phases is not None: D = mags.size(0) if mags is not None else phases.size(0)
    if mags is None: mags = torch.randn([D, len(freqs)]) / np.sqrt(p/2)
    if phases is None: phases = torch.rand([D, len(freqs)]) * 2 * np.pi - np.pi
    n_phases = torch.arange(N) * 2 * np.pi / N
    harmonic1 = torch.cos(prange * freqs[...,None,None] * 2 * np.pi / p + n_phases[...,None] - np.pi)
    harmonic2 = torch.cos(prange * 2 * freqs[...,None,None] * 2 * np.pi / p + 2 * n_phases[...,None] - np.pi)
    vec = (harmonic1 + harmonic2/8) # (F,N,p) THIS 1/8 MAGNITUDE HAS BEEN EMPIRICALLY FOUND WITHOUT PRINCIPLED THEORY
    a_vec, b_vec = vec[...,None], vec[...,None,:] # (F,N,p,1), (F,N,1,p)
    pre = a_vec + b_vec + np.sqrt(2) - 1 # (F,N,p,p) THE sqrt(2) - 1 BIAS TERM BELOW HAS ALSO BEEN EMPIRICALLY FOUND
    weights = (torch.cos(phases - 2 * n_phases[...,None,None]) * mags).permute(2,0,1) # (F,N,D)
    output = einops.einsum(F.relu(pre), weights, "freqs neur a b, freqs neur d_model -> d_model a b")
    desired = (mags[...,None] * torch.cos(prange * freqs[...,None] * 2 * np.pi / p + phases[...,None])).sum(1) # (D,p)
    output *= inout_norms[0]/output.square().mean(-1).sqrt().mean()
    desired *= inout_norms[1]/desired.square().mean(-1).sqrt().mean()
    logits = einops.einsum(output, desired, "d_model a b, d_model c -> a b c")
    answers = torch.argmax(logits, -1)
    c = answers[a,b].item()
    return (c, a_vec, b_vec, weights, desired, output, logits.flatten(0,1), answers) if get_tensors else c
