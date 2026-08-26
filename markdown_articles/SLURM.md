---
title: SLURM basics
date: 2026
---

# SLURM basics

## Overview 
In this article we describe the basics of SLURM - what is it and how is this useful.

generally there is a login node and from the login node we give command to spin up slurm mamaged clusters. There may be single node or multi-node managed by slurm. The communication underneath is managed by basic tcp, mpi etc. SLURM manages the nodes/clusters.

Lets define few terminology:
there is login-node (n_{login}), batch-host-node(n_{batch}), worker-node(n_{worker}). Login node is the node from where one give the request for gettign access to multiple SLURM managed nodes. Once the nodes are assigned, there is a n_{batch} which is the node that has been sort of a controller/first node in the assignment. Rest all nodes assigned are n_{worker}. Let's say Sneha have requested for 5 nodes with 8 GPU each. Out of these 5, one will be a n_{batch} and rest 4 will be n_{worker}. n_{batch} is sort of a node where all the commands from the bash script is going to run after all runtime environmnet is set up there. 

In an abstract sense a bash script contains like this :
- cluster assignment details
- set the nodes with different settings per requirements
- set the environment, venv, machine specific library loads etc
- manage the storage, start different other services before actual training  etc
- run the distributed training code
- save the results

Once the 5 nodes are assigned, they are part of the entire assignment. The bash script above can grow arbitrary big and complex - with multiple jobs running in sequence or in parallel depending on the complexity of the end tasks. But generally one can decrease/release the part of nodes, but may not add back additional nodes.

## Basic SLURM commands:

- squeue -u $USER (if job is already running)
- scontrol show job <jobid> (Get full node details for your job)
- scontrol show node node_1
scontrol show node node_1,node_2,node_3
(Get hardware/state details of specific nodes)
- srun --jobid=<jobid> -N3 --ntasks-per-node=1 nvidia-smi -L (per-node GPU status)
- srun --nodes=3 --ntasks-per-node=1 hostname (run a command on ALL nodes)
- srun --nodelist=gpu-node04,gpu-node05 --ntasks=2 --ntasks-per-node=1 nvidia-smi (run a command on ONLY gpu-node04 and gpu-node05)

- #SBATCH --nodes=3          # SLURM auto-picks 3 free nodes matching constraints
#SBATCH --gres=gpu:8 (auto-assign nodes)

- scontrol update JobId=<jobid> NodeList=gpu-node03,gpu-node04 (shrink: release some nodes)
- scancel --signal=TERM <jobid> (release a job step early)

- 
#!/bin/bash
for node in $(sinfo -N -h -o "%N"); do
    if ! ssh $node "nvidia-smi" &>/dev/null; then
        scontrol update NodeName=$node State=DRAIN Reason="GPU check failed"
    fi
done
(Scripted health-check auto-management)

## Examples: 




