from mpi4py import MPI
import numpy as np
import jax.numpy as jnp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

assert size == 3

# send numpy array
if rank == 0:
    print(f"Process {rank}: Doing nothing")
elif rank == 1:
    data = np.arange(10, dtype=np.float64)
    comm.Send(data, dest=2, tag=17)
    print(f"Process {rank}: Sending data to process 2 ")
elif rank == 2:
    data = np.empty(10, dtype=np.float64)
    comm.Recv(data, source=1, tag=17)
    print(f"Process {rank}: Printing data recovered from process 1 ", data)

comm.Barrier()

# send jnp array casted as numpy array
if rank == 0:
    print(f"== Process {rank}: Doing nothing")
elif rank == 1:
    data = jnp.arange(10, dtype=jnp.int32)
    data = np.array(data, dtype=np.int32)
    comm.Send(data, dest=2, tag=13)
    print(f"== Process {rank}: Sending data to process 2 ")
elif rank == 2:
    data = np.empty(10, dtype=np.int32)
    comm.Recv(data, source=1, tag=13)
    print(f"== Process {rank}: Printing data recovered from process 1 ", data)
