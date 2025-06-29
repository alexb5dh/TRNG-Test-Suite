import numpy as np
import math

def monobit_test(binary):
    """ Tests the proportion of 1s to 0s in the entire binary sequence.

    Returns:
        s (int): test statistic
        p (float) : p-value for the test
        sucecss (bool): test passed/failed
    """

    # binary tricks to compute popcount of u64-bit numpy array
    ones = binary.packed.view(np.uint64)
    ones -= ((ones >> 1) & 0x5555555555555555)
    ones = (ones & 0x3333333333333333) + (ones >> 2 & 0x3333333333333333)
    ones = (((ones + (ones >> 4)) & 0xf0f0f0f0f0f0f0f) * 0x101010101010101 >> 56) & 0xff

    sum2 = 2 * np.sum(ones)
    s = sum2 - binary.n if sum2 > binary.n else binary.n - sum2

    p = math.erfc(s/(math.sqrt(float(binary.n))*math.sqrt(2.0)))
    success = p >= 0.01

    return [p, success]