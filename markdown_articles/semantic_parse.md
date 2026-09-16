---
title: Code and semantic parsing
---

## Overview

In this article we describe semantic parsing with multi-stage training. We first discuss the overall ideas in an abstract sense, and then gradually progress towards the more advanced versions of training. The central theme throughout is reinforcement learning for semantic parsing.

## A short history - from encoder-decoder to decoder-only

We start from the decoder-only architecture. Before going advanced, it helps to look briefly at the history of it.

Prior to GPT, the dominant models were T5 and BART. However, these models were not pre-trained on that much semantic-parsing data at internet scale. So pre-training on this specific domain while still retaining a general understanding of language was always a problem. Researchers tried to gather different parallel corpora of (query, SQL) data to pre-train these models. The overall idea was that once the alignment is done, one can then fine-tune the model on the downstream task.

Moving forward, GPT-style models were pre-trained on internet-scale data. But since this was general pre-training, the semantic-parsing benchmark numbers would be bad here. Such an LLM can, of course, be instruction fine-tuned on (query, SQL) datasets. The problem is that IFT suffers from catastrophic forgetting. To bypass this, one mixes logical reasoning data, NLU tasks, code generation and NL2SQL/NL2KG data into the training data, and performs IFT at different strengths for each of these components. As a result of this, at the end both semantic-parsing and non-semantic-parsing benchmarks give good results.

## Components of a query

Let us see what the different components of a query are - **clauses**, **operators** and **operands**:

- **Clauses** - `SELECT`, `FROM`, etc.
- **Operators** - `AND`, `OR`, etc.
- **Operands** - table names, column names, values, etc.

Historically, mapping natural language to the *operands* has always been the troublesome part, because of the ambiguity of natural language and the non-uniformity of table and column names. Hence models have always struggled. To mitigate this we need large-scale, prolonged, multi-stage reinforcement learning.

Next we deep dive into reinforcement learning basics and how one can adopt RL to do semantic parsing.

## Reinforcement learning basics

Let us first review what the general reinforcement learning techniques are. There are different kinds of RL - model-based, model-free, policy gradient, off-policy, on-policy, etc. Some of the most commonly used ones are VPG (vanilla policy gradient), PPO, REINFORCE, GRPO and DPO. There have been many variations of them, but the general forms remain more or less the same.

We start first with VPG and REINFORCE. These are vanilla policy gradient methods, but they suffer from unstable training and high variance. To handle this, one might choose the PPO method. The idea in PPO is that a baseline is implemented for reward normalization. But to estimate this we need a value function, which is action-agnostic. This reduces the variance of the updates during the policy update, and the critic model is what helps estimate it. Hence in PPO a total of 5 models are used:

$$
\boxed{\ \theta_{\text{old}}, \quad \theta_{\text{policy}}, \quad \theta_{\text{ref}}, \quad \theta_{\text{critic}}, \quad \theta_{\text{reward}}\ }
$$

The algorithm jointly trains both $\theta_{\text{policy}}$ and $\theta_{\text{critic}}$. This is compute heavy and the training is unstable. To mitigate this, one might choose GRPO, where the value is replaced with the statistics of a batch of rollouts for the same prompt. This removes the need for a critic model and frees up the resources, although one might need to do multiple rollouts compared to the single rollout of PPO/REINFORCE, etc. In all of these we have a critic which is providing a stabilizing signal, along with a reward model (later we will discuss how the reward model is obtained in the first place).

DPO is different from the above methods in the way that, through a supervised learning formulation, one can achieve the preference-alignment objective.

## Multi-stage training for semantic parsing

Now that we have established the basics of reinforcement learning, we are going to see how RL is used, and we will gradually make the scenario more complicated as we go through multi-stage training. The assumption here is that we have a good instruction fine-tuned model - **not** fine-tuned on semantic parsing or code. We call this model **M**.


### Stage 1 - instruction fine-tuning (M → M_FT)

We can't do RL on **M**, because it does not have the capacity to handle the instructions and the task format for SP. Hence we first do IFT.

In IFT we take logical reasoning data, NLU data, code generation data and semantic-parsing data. We vary the strength of each of these and do IFT. We call this model **M_FT**. This model can generate only the reasoning and the query. Since this model is now nudged towards the basic SP task, we can do RL on it.

### Stage 2 - RL with verifiable reward (M_FT → M_IFT+RL)

Now we can take this **M_FT** and do RL. Specifically, we can do RLVR - reinforcement learning with verifiable reward. As the verification signal we can use the *execution reward*, which tells us whether applying the final generated query to the database produces a result that matches the answer. There are different ways to match the ground-truth answer against the generated answer; we will talk about this later. We call this model **M_IFT+RL**.

### Stage 3 - going fine-grained with sub-tasks

This model generates only a query and reasoning, given the natural language input. As a result, since only the output is verified, the reasoning itself can be corrupted. Moreover, as we go for more complicated and lengthy NL2SQL tasks, this training fails: as complexity grows, there is a search-space explosion in the final query space.

One sort of mitigation for this is to split the natural language input into sub-tasks, then generate a sub-query for each of them, and finally combine all these sub-queries in a cumulative way to get the final query. But we can't directly start with **M_IFT+RL** and do more RL on it: since it does not generate the intermediate sub-tasks, doing RL on this model will collapse. Hence we again do IFT + RL on **M_IFT+RL**.

But this time the IFT + RL is going to happen at the fine-grained intermediate sub-task level. At a high level, we take **M_IFT+RL** and do IFT on

$$
\boxed{\ \tau \;=\; \Big(x,\ (s_0, a_0),\ (s_1, a_1),\ \ldots,\ (s_n, a_n)\Big)\ }
$$

where the variables mean the following:

| Variable | Meaning |
|---|---|
| $x$ | The input to the whole problem - the natural language question along with the context the model is conditioned on (DB schema, table and column names). It stays the same across every step of the trajectory. |
| $s_i$ | The *state* at step $i$ - everything produced so far: the sub-questions already split off, the sub-queries already generated, and any intermediate result carried forward. $s_0$ is the initial state, where nothing has been decomposed yet. |
| $a_i$ | The *action* taken from state $s_i$ - one decomposition step, i.e. split off this sub-question and emit the sub-query for it. Applying $a_i$ to $s_i$ gives the next state $s_{i+1}$, so that $s_{i+1} = f(s_i, a_i)$. |
| $\tau$ | The full **trajectory** - the ordered chain of $(s, a)$ pairs from the initial state to the final query. |

So the sequence above is one complete path: given the question, this is the full chain of decomposition steps that led to the final query.

Once done, we can do RL training using DPO on the pair

$$
\begin{aligned}
\tau^{w} &= \Big(x,\ (s_0, a_0),\ \ldots,\ (s_i, a_i^{w})\Big) \quad \text{(chosen)} \\[4pt]
\tau^{l} &= \Big(x,\ (s_0, a_0),\ \ldots,\ (s_i, a_i^{l})\Big) \quad \text{(rejected)}
\end{aligned}
$$

Here $a_i^{w}$ (*win*) and $a_i^{l}$ (*lose*) are two candidate actions taken from the **same** state $s_i$ - one correct, one incorrect. $a_i^{w}$ is the sub-task split that leads towards the right final query, and $a_i^{l}$ is one that does not.

This is why both sides of the pair share the prefix $\big(x, (s_0, a_0), \ldots\big)$ and differ only at position $i$:

$$
\tau^{w}_{<i} \;=\; \tau^{l}_{<i}, \qquad a_i^{w} \;\succ\; a_i^{l} \ \big|\ s_i
$$

DPO needs two continuations that are identical up to the point of divergence, so that the preference signal isolates *that single decision* instead of the trajectory as a whole. The model then learns "from this state, prefer this split over that one" - the step-level credit assignment that the outcome-verified RL of Stage 2 could not provide. Since $i$ is just an index, such pairs can be mined at any depth of the trajectory, not only at the first step.

This aligns the model towards the right sub-task splits and away from the wrong ones. There are different ways of getting this correct-path and incorrect-path data, which is the next topic.


## Getting intermediate sub-task data at scale

How do we get scalable data for these intermediate steps / sub-tasks? One cannot gather this manually - such data is very expensive to collect. To mitigate this, one approach can be to automatically synthesize the intermediate sub-task-level data from the $(Q, \mathrm{DB}, Y)$ triple. So the question is, how do we get

$$
\boxed{\ (Q,\ \mathrm{DB}) \;\longrightarrow\; \big\{(q_j,\ y_j)\big\}_{j=1}^{m} \;\longrightarrow\; y\ }
$$

where $Q$ is the natural language question, $\mathrm{DB}$ the database/schema it is asked against, $Y$ the ground-truth answer, and each $(q_j, y_j)$ one synthesized sub-question with its sub-answer. The $m$ intermediate pairs are exactly the supervision that the trajectory $\tau$ of Stage 3 needs, and $y$ is the final answer they compose into - which can then be checked against $Y$.

One way to get this is using MCTS. The overall idea is that we can use other LLMs to generate an MCTS tree, and while we are generating this tree we can prune or approve the branches in it.

More details on the MCTS-based data synthesis are coming soon.
