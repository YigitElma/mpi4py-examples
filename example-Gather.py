import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI

if __name__ == "__main__":
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

    if rank == 0:
        print("Gathered array shape:", recvbuf.shape)
        print("Contents:")
        print(recvbuf)
