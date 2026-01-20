import numpy as np

def split_ss(ss, n=2):
    kids = ss.spawn(n)
    return kids[0], kids[1:]

def rng_from_ss(ss):
    return np.random.default_rng(ss)

if __name__ == "__main__":
    ss = np.random.SeedSequence(0)
    ss, (a_ss, b_ss) = split_ss(ss, 2)
    
    a = rng_from_ss(a_ss).normal(size=3)
    b = rng_from_ss(b_ss).normal(size=3)
