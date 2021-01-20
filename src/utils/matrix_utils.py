import numpy as np
from scipy.sparse import *

def iter_spmatrix(matrix):
    """ Iterator for iterating the elements in a ``scipy.sparse.*_matrix``

    This will always return:
    >>> (row, column, matrix-element)

    Currently this can iterate `coo`, `csc`, `lil` and `csr`, others may easily be added.

    Parameters
    ----------
    matrix : ``scipy.sparse.sp_matrix``
      the sparse matrix to iterate non-zero elements
    """
    if isspmatrix_coo(matrix):
        for r, c, m in zip(matrix.row, matrix.col, matrix.data):
            yield r, c, m

    elif isspmatrix_csc(matrix):
        for c in range(matrix.shape[1]):
            for ind in range(matrix.indptr[c], matrix.indptr[c+1]):
                yield matrix.indices[ind], c, matrix.data[ind]

    elif isspmatrix_csr(matrix):
        for r in range(matrix.shape[0]):
            for ind in range(matrix.indptr[r], matrix.indptr[r+1]):
                yield r, matrix.indices[ind], matrix.data[ind]

    elif isspmatrix_lil(matrix):
        for r in range(matrix.shape[0]):
            for c, d in zip(matrix.rows[r], matrix.data[r]):
                yield r, c, d

    else: assert False;