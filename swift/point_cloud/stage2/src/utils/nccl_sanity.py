import os, torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
torch.cuda.set_device(local_rank)

x = torch.ones(1024 * 1024, device="cuda") * (rank + 1)
dist.broadcast(x, src=0)
dist.all_reduce(x)
dist.barrier()
if rank == 0:
    print("NCCL OK, x[0] =", x[0].item())
