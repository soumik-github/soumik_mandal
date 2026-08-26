---
title: SLURM basics
---

## Overview

In this article we describe the basics of SLURM - what is it and how is this useful.

Generally there is a login node, and from the login node we give commands to spin up SLURM-managed clusters. There may be a single node or multiple nodes managed by SLURM. The communication underneath is managed by basic TCP, MPI, etc. SLURM manages the nodes/clusters.

## Terminology

Let's define a few terms:

$$
\boxed{n_{\text{login}}, \quad n_{\text{batch}}, \quad n_{\text{worker}}}
$$

- $n_{\text{login}}$: the **login node** - the node from where one gives the request for getting access to multiple SLURM-managed nodes.
- $n_{\text{batch}}$: the **batch/controller node** - once nodes are assigned, this is the node that acts as the first node / controller of the assignment. All commands in the bash script run here after the runtime environment is set up.
- $n_{\text{worker}}$: the **worker node(s)** - the rest of the assigned nodes.

For example, say Sneha requests 5 nodes with 8 GPUs each. Out of these 5, one becomes $n_{\text{batch}}$ and the remaining 4 become $n_{\text{worker}}$.

<pre class="mermaid">
flowchart LR
    L["Login Node"]:::login --> B["Batch Node (controller)"]:::batch
    B --> W1["Worker Node 1"]:::worker
    B --> W2["Worker Node 2"]:::worker
    B --> W3["Worker Node 3"]:::worker
    B --> W4["Worker Node 4"]:::worker

    classDef login fill:#ede9fe,stroke:#7c3aed,color:#4c1d95,stroke-width:1px;
    classDef batch fill:#dbeafe,stroke:#2563eb,color:#1e3a8a,stroke-width:1px;
    classDef worker fill:#dcfce7,stroke:#16a34a,color:#14532d,stroke-width:1px;
</pre>

## Anatomy of a batch script

In an abstract sense, a bash script contains steps like:

- cluster assignment details
- set the nodes with different settings per requirements
- set the environment, venv, machine-specific library loads, etc.
- manage the storage, start different other services before actual training, etc.
- run the distributed training code
- save the results

Once the 5 nodes are assigned, they are part of the entire assignment. The bash script above can grow arbitrarily big and complex - with multiple jobs running in sequence or in parallel depending on the complexity of the end tasks. But generally one can decrease/release part of the nodes, and may not add back additional nodes.

## Basic SLURM commands

### Job and node status

```bash
squeue -u $USER                            # check if a job is already running
scontrol show job <jobid>                  # get full node details for your job
scontrol show node node_1
scontrol show node node_1,node_2,node_3    # hardware/state details of specific nodes
```

### Running a command across nodes

```bash
# per-node GPU status
srun --jobid=<jobid> -N3 --ntasks-per-node=1 nvidia-smi -L

# run a command on ALL nodes
srun --nodes=3 --ntasks-per-node=1 hostname

# run a command on ONLY gpu-node04 and gpu-node05
srun --nodelist=gpu-node04,gpu-node05 --ntasks=2 --ntasks-per-node=1 nvidia-smi
```

### Requesting nodes in a batch script

```bash
#SBATCH --nodes=3      # SLURM auto-picks 3 free nodes matching constraints
#SBATCH --gres=gpu:8   # auto-assign nodes with 8 GPUs each
```

### Releasing / shrinking a job

```bash
# shrink: release some nodes
scontrol update JobId=<jobid> NodeList=gpu-node03,gpu-node04

# release a job step early
scancel --signal=TERM <jobid>
```

### Scripted health-check auto-management

```bash
#!/bin/bash
for node in $(sinfo -N -h -o "%N"); do
    if ! ssh $node "nvidia-smi" &>/dev/null; then
        scontrol update NodeName=$node State=DRAIN Reason="GPU check failed"
    fi
done
```

## Examples

### Problem 1: Matrix multiplication

Let's say there are 3 nodes with 8 GPUs each - `node_3`/`node_4`/`node_5`. From `node_3` we want to access `node_4` and run a Python script that generates two random matrices ($7\times 9$ and $9\times 7$), multiplies them to get a $7\times 7$ matrix, and sends the result back to `node_3`. Since `node_4` has 8 GPUs, we get 8 such matrices back on `node_3`. Once we have these 8 matrices on `node_3`, we multiply them together sequentially on a single GPU there, and print the final answer.

`matmul_demo.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=matmul_demo
#SBATCH --output=matmul_%j.log
#SBATCH --time=00:10:00

# ---- Component 0: node_3 = collector (1 task, 1 GPU) ----
#SBATCH --nodelist=node_3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

#SBATCH hetjob

# ---- Component 1: node_4 = producers (8 tasks, 8 GPUs) ----
#SBATCH --nodelist=node_4
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=2

source ~/venv/bin/activate     # adjust to your env
export MASTER_PORT=29500

# launch ONE combined 9-rank distributed world across both components
srun --het-group=0,1 python matmul_worker.py
```

`matmul_worker.py`:

```python
import os
import torch
import torch.distributed as dist

def main():
    rank = int(os.environ["SLURM_PROCID"])        # global rank: 0..8
    world_size = int(os.environ["SLURM_NTASKS"])   # 1 + 8 = 9

    os.environ.setdefault("MASTER_ADDR", "node_3")   # rank 0's hostname
    os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    if rank == 0:
        # ---- COLLECTOR on node_3 ----
        collected = []
        for src in range(1, world_size):
            buf = torch.zeros(7, 7)
            dist.recv(tensor=buf, src=src)
            collected.append(buf)
            print(f"[node_3] received matrix from rank {src}")

        device = torch.device("cuda:0")   # node_3's 1 GPU
        result = collected[0].to(device)
        for m in collected[1:]:
            result = result @ m.to(device)   # sequential 7x7 chain multiply

        print("Final 7x7 result on node_3:")
        print(result.cpu())

    else:
        # ---- PRODUCER on node_4 ----
        local_gpu = rank - 1              # ranks 1-8 -> GPUs 0-7
        device = torch.device(f"cuda:{local_gpu}")

        A = torch.rand(7, 9, device=device)
        B = torch.rand(9, 7, device=device)
        C = A @ B                          # 7x9 @ 9x7 = 7x7

        dist.send(tensor=C.cpu(), dst=0)   # send back to node_3
        print(f"[node_4] rank {rank} (GPU {local_gpu}) sent its 7x7 matrix")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```





### Problem 2: Matrix multiplication (modified)

Let's say there are 3 nodes with 8 GPUs each - `node_3`/`node_4`/`node_5`. From `node_3` we want to access `node_4` and run a Python script that generates two random matrices ($7\times 9$ and $9\times 7$), multiplies them to get a $7\times 7$ matrix, and sends the result back to `node_3`. Since `node_4` has 8 GPUs, we get 8 such matrices back on `node_3`. Once we have these 8 matrices on `node_3`, we multiply them together in parallel in 4 gpus of the `node_3` on a single GPU there, and print the final answer.

`matmul.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=matmul_demo
#SBATCH --output=matmul_%j.log
#SBATCH --time=00:10:00

# ---- Component 0: node_3 = 4 collector ranks (1 GPU each) ----
#SBATCH --nodelist=node_3
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2

#SBATCH hetjob

# ---- Component 1: node_4 = 8 producer ranks (1 GPU each) ----
#SBATCH --nodelist=node_4
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=2

source ~/venv/bin/activate
export MASTER_PORT=29500

srun --het-group=0,1 python matmul_worker.py

```

`matmul_worker.py`:

```python
import os
import torch
import torch.distributed as dist

def main():
    rank = int(os.environ["SLURM_PROCID"])        # 0..11
    world_size = int(os.environ["SLURM_NTASKS"])   # 4 + 8 = 12

    os.environ.setdefault("MASTER_ADDR", "node_3")
    os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    if rank < 4:
        # ---- COLLECTOR on node_3, ranks 0-3, one GPU each ----
        device = torch.device(f"cuda:{rank}")
        src1 = 4 + 2 * rank        # each collector pairs with 2 specific producers
        src2 = src1 + 1

        buf1 = torch.zeros(7, 7)
        buf2 = torch.zeros(7, 7)
        dist.recv(tensor=buf1, src=src1)
        dist.recv(tensor=buf2, src=src2)

        a = buf1.to(device)
        b = buf2.to(device)
        result = a @ b
        torch.cuda.synchronize(device)

        print(f"[node_3] rank {rank} (GPU {rank}) result:")
        print(result.cpu())

    else:
        # ---- PRODUCER on node_4, ranks 4-11 ----
        local_gpu = rank - 4                    # 0..7
        device = torch.device(f"cuda:{local_gpu}")

        A = torch.rand(7, 9, device=device)
        B = torch.rand(9, 7, device=device)
        C = A @ B

        dst = local_gpu // 2                    # routes to collector rank 0-3
        dist.send(tensor=C.cpu(), dst=dst)
        print(f"[node_4] rank {rank} (GPU {local_gpu}) sent to collector rank {dst}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

```

Mapping of rank/gpus: 

<table style="border-collapse: collapse; width: 100%;">
<tr>
<th style="border: 1px solid; padding: 6px;">Producer rank (node_4)</th>
<th style="border: 1px solid; padding: 6px;">GPU</th>
<th style="border: 1px solid; padding: 6px;">Sends to collector rank</th>
</tr>
<tr>
<td style="border: 1px solid; padding: 6px;">4, 5</td>
<td style="border: 1px solid; padding: 6px;">0, 1</td>
<td style="border: 1px solid; padding: 6px;">0</td>
</tr>
<tr>
<td style="border: 1px solid; padding: 6px;">6, 7</td>
<td style="border: 1px solid; padding: 6px;">2, 3</td>
<td style="border: 1px solid; padding: 6px;">1</td>
</tr>
<tr>
<td style="border: 1px solid; padding: 6px;">8, 9</td>
<td style="border: 1px solid; padding: 6px;">4, 5</td>
<td style="border: 1px solid; padding: 6px;">2</td>
</tr>
<tr>
<td style="border: 1px solid; padding: 6px;">10, 11</td>
<td style="border: 1px solid; padding: 6px;">6, 7</td>
<td style="border: 1px solid; padding: 6px;">3</td>
</tr>
</table>

Now we have a general recipe for communicating between nodes with multiple-gpus. We can pass arbitrary functions outputs from one node to another, message them and forward to another gpu (in prallel or sequence). Our function can be anything - arithmetic operation, complex API calls, gradient accumulation, back-prop which can only fit in multiple nodes, weather model update, complex RL set up, simulation of physical phenomena etc and many more.



