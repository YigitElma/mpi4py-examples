Some examples of ``mpi4py``. This is not an extensive list but I will try to extend it as I use them.

An important point to keep in mind is that lower case functions are meant to be used for general Python objects. If you have ``numpy`` arrays or any type of array that exposes buffer and compatible with ``mpi4py``, you can use upper-case functions. For example,

```python
    import numpy as np
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = 5
    M = 10
    x = np.ones((N, M), dtype=np.float32) * rank

    # Allocate receive buffer on root
    if rank == 0:
        recvbuf = np.empty((size * N, M), dtype=np.float32)
    else:
        recvbuf = None

    # Gather arrays to root
    comm.Gather(x, recvbuf, root=0)
```

This will use faster memory transfer. General Python objects are passed by first ``pickle``ing and then un``pickle``ing, hence it is slower. One can use both for convenience. For instance, let's say we don't know the shape of the arrays, hence cannot create a proper receive buffer, you can first send the shape as a tuple via lower-case function. After creating the receiving buffer, send the bigger array. Since the shape information is very small, the overhead wouldn't hurt. 
