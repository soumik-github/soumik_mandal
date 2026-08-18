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



