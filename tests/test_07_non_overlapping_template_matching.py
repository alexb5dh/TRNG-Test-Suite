import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mp
import scipy.special as ss
from functools import partial
from itertools import repeat
import math

def non_overlapping_template_matching_test(binary, sigLevel=0.01, m=9, advanced = False):
    
    def template_matches(block, template):
        strides = np.packbits(np.lib.stride_tricks.as_strided(block, shape=((block.size - m + 1), m), strides=(1,1)), axis=1).view(np.uint16).reshape(-1)
        inds    = np.where(strides == template)[0]
        dists   = np.diff(inds)

        return len(inds) - np.count_nonzero(dists < m)

    bits = binary.unpacked
    n = len(bits)
    M = n // 8
    N = n // M

    mu = (M - m + 1) / (2 ** m)
    std = M * ((1 / (2 ** m)) - (2 * m - 1) / (2 ** (2 * m)))

    blocks = bits[:N * M].reshape(N, M)

    if not advanced:
        template = np.uint16(2**(m-1))
        matches = non_overlapping_matches(blocks, m, template)
        chisq = np.sum(((matches - mu) ** 2) / std)
        p = ss.gammaincc(N / 2, chisq / 2)
        return [p, (p >= sigLevel)]

    numTemplates = 148
    templateRange = np.arange(2**m,dtype=np.uint16)
    templates = np.random.choice(templateRange,size=numTemplates).reshape(-1,1)

    results = []

    for template in templates:

        template = np.unpackbits(template.view(np.uint8))[16-m:]
    
    # if len(template) > m:
    #     template = template[len(template) - m:]
    # elif len(template) < m:
    #     template = np.concatenate([np.zeros((m - len(template)), dtype=np.uint8), template])

    
        template = np.dot(template, 1 << np.arange(m, dtype=np.uint16)[::-1])

    # if n > 10_000_000:
    #     with ThreadPool(mp.cpu_count()) as p:
    #         matches = np.array([*p.imap(partial(non_overlapping_matches, m=m, template=template), blocks)])
    # else:
    #     matches = np.array([template_matches(block, template) for block in blocks])

        matches = non_overlapping_matches(blocks, m, template)

        chisq = np.sum(((matches - mu)**2) / std)

        p = ss.gammaincc(N/2, chisq/2)
        
        success = (p >= sigLevel)
        results.append([p,success])

        ret = sum([r[1] for r in results])
        ret = [[ret, True],[numTemplates-ret,False]]
        return ret

    return results[0]

def non_overlapping_matches(block, m, template):
    strides = np.lib.stride_tricks.sliding_window_view(block, window_shape=m, axis=1)
    mask = np.array(1 << np.arange(m), dtype=np.uint16)[::-1]

    with ThreadPool(mp.cpu_count()) as p:
        repacked = [*p.starmap(np.matmul, zip(strides, repeat(mask)))]
        inds     = [*p.imap(np.argwhere, repacked == template)]
        dists    = [*p.imap(np.diff, inds)]
        lens     = [*p.imap(len, inds)]
    
    overlaps = np.array([np.count_nonzero(b < m) for b in dists])

    return lens - overlaps