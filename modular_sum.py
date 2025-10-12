import torch
import numpy as np
import einops
import torch.nn.functional as F

def modular_sum(a,b,freqs,p=113,N=8,D=128, *, mags=None, phases=None, inout_norms=(3.19, 0.101), get_tensors=False):
    '''
        This computes modular addition: a + b % p = c and returns c
        freqs: key frequencies to use that make up the sinusoidal content in the activations
        mags/phases: these represent the spectrum of the desired (~W_U unembed matrix) output
        inout_norms: these normalize the final residual stream and the unembed to Neel Nanda's original norms
        get_tensor: returns all intermediate tensors as well as the answer, c
    '''
    prange = torch.arange(p)
    if mags is not None or phases is not None: D = mags.size(0) if mags is not None else phases.size(0)
    if mags is None: mags = torch.randn([D, len(freqs)]) / np.sqrt(p/2)
    if phases is None: phases = torch.rand([D, len(freqs)]) * 2 * np.pi - np.pi
    phzrange = torch.arange(N) * 2 * np.pi / N
    harmonic1 = torch.cos(prange * freqs[...,None,None] * 2 * np.pi / p + phzrange[...,None] - np.pi)
    harmonic2 = torch.cos(prange * 2 * freqs[...,None,None] * 2 * np.pi / p + 2 * phzrange[...,None] - np.pi)
    vec = (harmonic1 + harmonic2/8) # THIS 1/8 MAGNITUDE RATIO HAS BEEN EMPIRICALLY FOUND WITHOUT PRINCIPLED THEORY
    a_vec, b_vec = vec[...,None], vec[...,None,:]
    weights = (torch.cos(phases - 2 * phzrange[...,None,None]) * mags).permute(2,0,1)
    ## THE sqrt(2) - 1 BIAS TERM BELOW HAS ALSO BEEN EMPIRICALLY FOUND
    output = einops.einsum(F.relu(a_vec + b_vec + np.sqrt(2) - 1), weights, "freqs d1 ..., freqs d1 d2 -> d2 ...")
    output *= inout_norms[0]/output.square().mean(-1).sqrt().mean()
    desired = (mags[...,None] * torch.cos(prange * freqs[...,None] * 2 * np.pi / p + phases[...,None])).sum(1)
    desired *= inout_norms[1]/desired.square().mean(-1).sqrt().mean()
    logits = einops.einsum(output, desired, "d_model p1 p2, d_model p3 -> p1 p2 p3")
    answers = torch.argmax(logits, -1)
    c = answers[a,b].item()
    return (c, a_vec, b_vec, weights, desired, output, logits.flatten(0,1), answers) if get_tensors else c
