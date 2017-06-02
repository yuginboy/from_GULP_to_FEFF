import numpy
cimport numpy
cpdef numpy.ndarray[numpy.double_t, ndim=1] cVectorAdd_f(numpy.ndarray[numpy.double_t, ndim=1] a, numpy.ndarray[numpy.double_t, ndim=1] b):
    cdef int i, n
    n = numpy.size(a)
    cdef numpy.ndarray[numpy.double_t, ndim=1] out
    out = numpy.ndarray(n, dtype=numpy.double)

    for i in range(n):
        out[i] = a[i] + b[i]

    return out

# def cVectorAdd_f (a, b):
#     return a + b

if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')