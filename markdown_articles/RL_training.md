---
title: RL training setup - agents
---

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
    B <--> SKR["retriver hosting server"]:::skr
    C --> VAFC["verl agent"]:::vafc
    D <--> SRG["vLLM/sglang server"]:::srg
    E --> VAMU["verl agent"]:::vamu

    classDef question fill:#ede9fe,stroke:#7c3aed,color:#4c1d95,stroke-width:1px;
    classDef gen fill:#dbeafe,stroke:#2563eb,color:#1e3a8a,stroke-width:1px;
    classDef retrieve fill:#dcfce7,stroke:#16a34a,color:#14532d,stroke-width:1px;
    classDef consolidate fill:#fef9c3,stroke:#ca8a04,color:#713f12,stroke-width:1px;
    classDef memory fill:#fce7f3,stroke:#db2777,color:#831843,stroke-width:1px;
    classDef sqg fill:#dce7f3,stroke:#db2777,color:#831843,stroke-width:1px;
    classDef skr fill:#dce7f3,stroke:#db2777,color:#831843,stroke-width:1px;
    classDef vafc fill:#dce7f3,stroke:#db2777,color:#831843,stroke-width:1px;
    classDef srg fill:#dce7f3,stroke:#db2777,color:#831843,stroke-width:1px;
    classDef vamu fill:#dce7f3,stroke:#db2777,color:#831843,stroke-width:1px;
</pre>


The servers may be hosted via vLLM/sglang or extrenal API calls with fastAPI/Uvicorn wrapper which will listen to the requests during verl training. The sub-question can be generated using a hosted separate server. Same argument for retriever, and answer generator (RG). Deatils on these on a separate article.

For verl training, we need servers where we may have multiple nodes with GPUs inside them. These nodes can be managed by SLURM(depends on platform). RAY clauster assignment for each component of verl can be done - SLURM clusters can be managed/assigned by RAY.
Below are few ways to them : 

SLURM basics - [SLURM basics](https://soumik-github.github.io/soumik_mandal/html_articles/SLURM.html)

RAY cluster assignment basics - [RAY cluster/concepts basics](https://soumik-github.github.io/soumik_mandal/html_articles/RAY.html)

Once RAY clusters are assigned, we can mention them while spinning up verl RL code.

## One full verl RL-GRPO training step, start to finish, through code

This traces a single outer training step in training phase - everything that happens between "trainer picks up a batch of prompts" and "actor weights are updated and ready for the next batch."

```text
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 1 — rollout generation (data collection)                      │
│  64 prompts × n=4  →  256 trajectories                              │
│  weight sync: FSDP actor shards → vLLM rollout engine (on-policy)   │
│  256 tasks dispatched to 8 Ray AgentLoopWorker processes            │
│  each process runs ~32 StageAwareLoop.run() coroutines concurrently │
└─────────────────────────────────────────────────────────────────────┘
                                   │
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 2 — reward + advantage                                        │
│  256 AgentLoopOutputs collected → KL vs reference policy →          │
│  GRPO group-relative advantage (normalize within each uid's)      │
└─────────────────────────────────────────────────────────────────────┘
                                   │
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 3 — gradient update                                           │
│  256 samples ÷ ppo_mini_batch_size=32 → 8 mini-batches              │
│  each: FSDP forward+backward (seq-parallel=4) → clipped GRPO loss   │
│        + KL loss → optimizer.step()                                 │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                    updated actor weights → next batch's Phase 1
```

### Phase 1 - Rollout generation: 64 prompts → 256 trajectories

**1a. Batch sampling.** The trainer's dataloader pulls `train_batch_size=64` rows from train data. Each row is expanded by `actor_rollout_ref.rollout.n=4` - GRPO's group size - producing 256 `(question, uid, sampling_seed)` tasks. The 4 duplicates of one prompt keep the same `uid`; only the sampling randomness differs between them.

**1b. Weight sync into the rollout engine.** Before generation starts, verl copies the current FSDP-sharded actor weights into the vLLM async rollout engine (`actor_rollout_ref.rollout.name=vllm`, `mode=async`, `tensor_model_parallel_size`). This is what keeps the rollout on-policy: every batch is generated from the exact weights that will be updated at the end of that same step, not a stale copy. (The weight-transfer mechanism itself - sharded broadcast/NCCL update into vLLM's KV-cache-aware engine - is internal to verl)

**1c. Dispatch to Ray workers.** The 256 tasks are distributed across `actor_rollout_ref.rollout.agent.num_workers=8`. Ray actors - 8 separate OS processes, each hosting ~32 of the 256 trajectories as concurrent asyncio coroutines.

**1d. Inside each `run of agent trace` - where trainable and frozen calls diverge.** Up to `max_iterations=10` loop iterations, each a sequential chain of awaits:

| Step | Calls | Model | Route |
|---|---|---|---|
| Subquestion generation | `_subquestion_generation` | Frozen Foundation Model | GeneratorAgent → `run_in_executor` (32-thread pool) → HTTP → SGLang/vLLM server |
| Retrieval | `_retrieve` | N/A | RetrievalAgent → thread pool → HTTP → retriever server (FAISS) |
| Extract/consolidate | `_extract` | Trainable FM | `self.server_manager.generate(...)` → verl's vLLM rollout engine (the one just weight-synced in 1b) |
| Reflection extract (conditional) | `_extract` | Trainable FM | same as above |
| Subanswer generation | `_subanswer_generation` | Frozen FM | GeneratorAgent → SGLang/vLLM server |
| Memory-update extract | `_extract` | Trainable FM | verl's vLLM rollout engine |

The frozen-model calls are where the client-side/server-side concurrency picture applies: each GeneratorAgent/RetrievalAgent call is a single-question HTTP request, offloaded to that process's 32-thread pool; across all 8 Ray processes, up to 8 × 32 = 256 such requests can be simultaneously in flight against the one shared SGLang endpoint launched by the server, which reassembles them into its own GPU batch via continuous batching (`--mem-fraction-static 0.8`, `--dp` auto-replicated across the 8 training-node GPUs). If the vLLM/sglang is not set up properly or GPUs are not enough to handle the incoming request from training script, then they are put into queue of inference server. The asyncio set up in RL expects all the calls to be finshed within a prescribed time, and wothin this if not done, then those calls will be dropped from verl training. Since verl training will proceed after this prescribed time, there will be loss of training quality since traces are missing and there is nothing to train on. Hence it is important to have a sync between the outgoing requests (from RL training) and incoming requests (in inference server).  The trainable-model calls, by contrast, never leave the Ray/verl process boundary - they go straight to the co-located vLLM rollout engine, not through the previous thread pool at all.

Within one trajectory, `sha256(uid)` deterministically picks a single `(extractor_type, iteration)` - the same choice for all 4 GRPO samples of that question. Only the extract call matching that pick has its `prompt_ids`/`response_ids`/`log_probs` kept; every other trainable-model call in the trajectory only shapes memory/context, its tokens discarded.

**1e. Output collection.** Each `run` finishes by scoring itself: `path_reward` (reasoning-trace quality), `outcome_reward` (final-answer correctness), `step_reward` (the pinned sub-QA pair's correctness) - all three via EvaluatorAgent → SGLang, plus `structure_reward` (did the pinned extract call parse as valid JSON). Combined into one scalar `reward = structure_reward + path_reward + outcome_reward + step_reward`, returned with `prompt_ids`, `response_ids`, `response_mask`, `response_logprobs`, `reward_score`. Once all 256 coroutines across all 8 workers resolve, Phase 1 is done. one may add as many different reward signals in each traces depending on the application/research_statement at hand. If lets say its a code quality reward, then one might add how many test cases are passing or not/if one has correct AST, how much is edit distance of generated code AST from the GT one etc. 

More details on what is reward and how depending on the domain it varies is coming soon.

### Phase 2 - Reward shaping and advantage estimation

verl gathers the 256 traces into one batch. Because one generally use KL loss too, a KL penalty between the actor and the frozen reference policy is computed and folded into the reward per response before advantage estimation. Then, the 256 rewards are grouped by `uid` into 64 groups of 4, and each group's `reward_score` is mean-subtracted / std-normalized within the group - GRPO's advantage is purely this group-relative normalization, no learned value function required. 

### Phase 3 - Policy gradient update

The 256-sample batch, now carrying per-trace advantages, is split into 256 / `ppo_mini_batch_size=32` = 8 mini-batches. For each mini-batch:

- Forward + backward pass through the FSDP-sharded actor (if say `ulysses_sequence_parallel_size` - splits long sequences across multiple GPUs), using `use_dynamic_bsz=true` to pack sequences by token budget (`ppo_max_token_len_per_gpu`).
- Loss: the clipped GRPO/PPO surrogate (`clip_ratio_low`, `clip_ratio_high`, `clip_ratio_c`) plus the KL loss term (`use_kl_loss=True`, `kl_loss_coef` - a second, separate KL penalty from the reward-shaping one in Phase 2, applied directly in the loss).
- `optimizer.step()` does gradient update.

Eight such mini-batch updates complete the "gradient update" part of the step (`trainer.balance_batch=true` load-balances token counts across the training GPUs before this pass).

### Phase 4 - Loop closure and checkpointing

The updated actor weights are what get synced into the vLLM rollout engine at the start of the next training step (back to Phase 1b) - this is what makes the algorithm on-policy across steps. Every `trainer.save_freq` steps, actor (and critic) checkpoints are written to `$default_local_dir`. `trainer.total_epochs` means this whole Phase 1→4 cycle repeats across 5 full passes over `train.parquet`. But the rollout also can be made off-policy too where the actor is different than roll-out model in a sense that roll-out model is a delayed version of actor model.

There are different types of training data one can have for RL training. Once type of traininig is we can use a bigger model to score the traces or can have own reward model for scoring. Depending on application such as maths/code domain, if one has final answers, then one can use RLVR (verifiable reward) along with different other typesreward signal such as block code similarity, reading easeness of code, variable naming convention, edge case analysis, static/dynamic analysis etc.

More on vison-languge model and robotics task RL learning will be convered in later articles.



