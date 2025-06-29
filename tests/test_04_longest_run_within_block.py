import numpy as np
import scipy.special as ss

def longest_run_within_block_test(binary):
    """
        Like longest run test, but within blocks of the binary string.
    """
    def longest_run_in_block(x):

        count = 0
        while x:
            x = (x & (x << 1)) 
            count += 1

        return count

    bits = binary.unpacked
    n = binary.n

    if n < 128:
        print("ERROR! Not enough data to run this test. (Longest run within block test)")
        return -1
    elif n < 6272:
        K, M = 3, 8
        vclasses = [1, 2, 3, 4]
        vprobs = [0.21484375, 0.3671875, 0.23046875, 0.1875]
    elif n < 750000:
        K, M = 5, 128
        vclasses = [4, 5, 6, 7, 8, 9]
        vprobs = [0.1174035788, 0.242955959, 0.249363483, 0.17517706, 0.102701071, 0.112398847]
    else:
        K, M = 6, 10000
        vclasses = [10, 11, 12, 13, 14, 15, 16]
        vprobs = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]

    N = n // M

    numBlocks = len(bits) // M

    bits = bits[:numBlocks * M].reshape(numBlocks, M)
    repacked = np.packbits(bits, axis=1)
    blocks = np.array([int.from_bytes(b.tobytes(), 'big') for b in repacked])
    runs = np.array([longest_run_in_block(x) for x in blocks])

    # compute the frequency of lengths based on the monitored classes
    freqs = np.histogram(runs, bins=[-9e10,*vclasses, 9e10])[0]
    freqs = [sum(freqs[:2]), *freqs[2:]]
    vs = np.array(freqs, dtype=np.float32)

    chisq = sum((vs[i] - N*vprobs[i])**2 / (N*vprobs[i]) for i in range(len(vs)))

    p = ss.gammaincc(K/2, chisq/2)

    success = (p >= 0.01)

    return [p, success]