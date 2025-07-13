import numpy as np
import math

def random_excursion_variant_test(binary, sigLevel=0.01):
    bits = 2*binary.unpacked.astype(np.int8)-1

    s = np.add.accumulate(bits, dtype=np.int16)
    J = len(s[s==0]) + 1
    s = s[(s >= -9) & (s <= 9)]

    num, counts = np.unique(s, return_counts=True)
    num = num.astype(np.int32)

    # compute p value for each class (18 total classes)
    ps = np.zeros(18) - 1
    for n, c in zip(num, counts):
        if n != 0:
            ps[n + 9 if n < 0 else n + 8] = math.erfc(abs(c-J)/math.sqrt(2*J*(4*abs(n)-2)))

    for i, p in enumerate(ps):
        if p == -1:
            n = i - 9 if i < 9 else i - 8
            ps[i] = math.erfc(abs(0-J)/math.sqrt(2*J*(4*abs(n)-2)))

    return [(p, p >= sigLevel) for p in ps]
