try:
    import mkl_random as r

    r.seed(None, 'SFMT19937')
except ModuleNotFoundError:
    import numpy.random as r

rand = r
