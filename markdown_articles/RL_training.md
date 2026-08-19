---
title: RL training setup - Agnetic RAG 
date: 2026-08-17
author: soumik
---

# RL training setup - agentic framework, different stages, understanding the overall flow, open problems

## RL training setup for Agentic RAG

Overall - DPO, vs PPO, vs GRPO.

Bias vs variance
Exploration vs exploitation

GRPO - what are different componenets
How GRPO is formulated in Agentic-RAG set up.

fastAPI vs Uvicorn (running a server vs listening from server and routing)

vLLM/sglang hosting - go nuts on these

Wrap vLLM server on fastAPI and send request/listen through uvicorn

SLURM cluster - basic ideas on SLURM and how they operate. How do we talk to one SLURM cluster with another (basic networking and comm)

Ray - how do we manage SLURM cluster from Ray

how do we feed ray cluster from VERL




VERL training details - batch size, roll out, async/sync call for the APIs, how these requests goes to the APIs and get the repsonse within a timeframe. What happens if vLLM and asyncio server is not set up properly - loss of traiing data and effort vs gpu utilization.


###########

We give high level overview how a training might look like in this Agentic-RAG set up for the previuos article mentioned before.

Assumption : 
- we have a training data, test dataset.
- the training modules are managed by SLURM, RAY
- GRPO is used since this is the most simplest 

The overall flow diagram of this looks like this : 


<pre class="mermaid">
flowchart LR
    Q["Question"]:::question --> A["Subquestion Generation (QG)"]:::gen --> B["Retrieve (KR)"]:::retrieve --> C["Consolidate (FC)"]:::consolidate --> D["Generate (RG)"]:::gen --> E["Update Memory (MU)"]:::memory
    E -.->|"next iteration"| A
    A <--> SQG["vLLM/sglang server"]:::sqg

    classDef question fill:#ede9fe,stroke:#7c3aed,color:#4c1d95,stroke-width:1px;
    classDef gen fill:#dbeafe,stroke:#2563eb,color:#1e3a8a,stroke-width:1px;
    classDef retrieve fill:#dcfce7,stroke:#16a34a,color:#14532d,stroke-width:1px;
    classDef consolidate fill:#fef9c3,stroke:#ca8a04,color:#713f12,stroke-width:1px;
    classDef memory fill:#fce7f3,stroke:#db2777,color:#831843,stroke-width:1px;
    classDef sqg fill:#fce7f3,stroke:#db2777,color:#831843,stroke-width:1px;
</pre>


























