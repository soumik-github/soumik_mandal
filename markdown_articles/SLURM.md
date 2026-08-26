---
title: SLURM basics
---

# SLURM basics

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

_Coming soon._
