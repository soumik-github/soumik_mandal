---
title: Agnetic RAG - agentic framework, different stages, open problems
date: 2026-08-17
author: soumik
---

# Agnetic RAG - agentic framework, different stages, open problems

## Agnetic-RAG

The article  formulates Retrieval-Augmented Generation (RAG) as a
**sequential decision-making problem**, rather than a sequence of
independent retrieve-and-generate operations in one-shot. The main issue is that, in
multi-hop reasoning, continuously accumulating retrieved documents can
introduce **irrelevant information, redundancy, contradictions, and
reasoning drift**.
rating the traces for a question we can choose random place in the trace path which we can use for training the "Extractor". We may not choose all possible intermediate steps and final step fro training "Extrcator" since it will be prohibitibly epxensive. A vanilla version of this is where we just generate the traces, dont do any intermediate rewarding, do only final rewarding. But since we want to also improve on the intermeidate steps - specially on the FC and MU stages - we can incorporate the reward signals at these steps.

The training was done on GPU clusters managed by SLURM. Through Ray we managed the servers, and then verl was used for GRPO RL trai
Agnetic RAG addresses this by maintaining a
**persistent, dynamically curated working memory** that evolves
throughout the reasoning process.

### Agnetic-RAG Framework

Let $x$ denote the original question and $K$ the external knowledge corpus.
The reasoning process is represented as a trajectory:

$$
T = \{S_0, a_0, S_1, a_1, \ldots, a_{n-1}, S_n\}
$$

where $S_i$ is the system state at reasoning step $i$, and $a_i$ is the
action that transforms $S_i$ into $S_{i+1}$.

The state at each step is defined as:

$$
\boxed{S_i = (q_i, r_i, M_i)}
$$

where:

- $q_i$: the **current sub-question** being solved.
- $r_i$: the **intermediate response** generated for that sub-question.
- $M_i$: the **global working memory** after step $i$.

The initial state is:

$$
S_0 = (x, \emptyset, \emptyset)
$$

The important aspect is that $M_i$ is a curated knowledge state containing information that is useful for the ongoing reasoning process. It's a semnatic description of the present state of the reasoning to inform the system if the next step to be taken is legit given the present situation. 

This allows the system to:

1. Retain useful information discovered in earlier steps,
2. Resolve conflicting information,
3. Remove redundant information, and
4. Avoid repeatedly retrieving or processing the same evidence.
5. Context aware fragmetation of question into sub-questions.

#### Three-Component building block

In broad term the framework consists of three components:

$$
\boxed{\pi = (R,G,E)}
$$

where:

- $R$: **Retriever** — retrieves relevant documents from the external corpus $K$. This can be KG call or database call or odcuments internal or jira system or google api call etc. Thsi framework gives flexibility towards that.
- $G$: **Generator** — generates intermediate answers for present sub-question.
- $E$: **Extractor** — manages and updates the working memory. 
In later section we will observe how these three modules gets reused in all the stages of the framework's actual implementation.

This work focuses on : **only the Extractor $E$ is trained**,
while the Retriever $R$ and Generator $G$ remain frozen. There are broad scope of work where one can train their own retriever, reranker or question ambiguity resolver. But for this scope of this work, we focus only on training the Extractor.

This makes the framework modular: stronger retrievers or generators can be
used without retraining the memory-management component. The memory-management module is a standalone one which can be plug-and-play with other options that is available out-of-the-box.

#### Information Consolidation

At reasoning step $i$, the Retriever produces a set of documents $D_i$ for
the current sub-question $q_i$. The Extractor consolidates this information
together with the existing memory:

$$
\boxed{I_i = E_{\mathrm{consolidate}}(q_i,D_i,M_{i-1})} \qquad (1)
$$

where $I_i$ is the **distilled information** used for the current
reasoning step.

The consolidation process performs three main functions:

- **Relevance filtering:** retain information useful for $q_i$.
- **Contradiction resolution:** reconcile conflicting information.
- **Redundancy elimination:** remove duplicate or unnecessary information.

Thus, the Generator reasons over the **curated information $I_i$**
rather than directly consuming every retrieved document.

#### Memory Update

After generating the intermediate response, the Extractor updates the
persistent memory:

$$
\boxed{M_i = E_{\mathrm{update}}(x,M_{i-1},I_i,(q_i,r_i))} \qquad (2)
$$

The new memory therefore depends on:

- the original question $x$,
- the previous memory $M_{i-1}$,
- the newly consolidated information $I_i$, and
- the current question-answer pair $(q_i,r_i)$.


### Real-life implementation

Here we describe the stages of implementation using the modules from above. There are total 5 stages. Here is the flow 

$$
\text{question} \rightarrow \text{Subquestion-generation} \rightarrow \text{Retrieve} \rightarrow \text{Consolidate} \rightarrow \text{Generate} \rightarrow \text{Update-Memory}
$$

<pre class="mermaid">
flowchart LR
    Q["Question"]:::question --> A["Subquestion Generation"]:::gen --> B["Retrieve"]:::retrieve --> C["Consolidate"]:::consolidate --> D["Generate"]:::gen --> E["Update Memory"]:::memory
    E -.->|"next iteration"| A

    classDef question fill:#ede9fe,stroke:#7c3aed,color:#4c1d95,stroke-width:1px;
    classDef gen fill:#dbeafe,stroke:#2563eb,color:#1e3a8a,stroke-width:1px;
    classDef retrieve fill:#dcfce7,stroke:#16a34a,color:#14532d,stroke-width:1px;
    classDef consolidate fill:#fef9c3,stroke:#ca8a04,color:#713f12,stroke-width:1px;
    classDef memory fill:#fce7f3,stroke:#db2777,color:#831843,stroke-width:1px;
</pre>

Each stage in braod sense is descrived below :
- **Subquestion-generation:** generates $q_j$ give historical context and orginal question. This is sigle hop in a complex multi-hop question answering.
- **Retrieve:** Retrives appropriate context from IS (information-sources) given the present query to answer.
- **Consolidate:** remove duplicate or unnecessary information and give feedback on the context from retriever stage.
- **Generate:** generates answer of the present stage (sub-question).
- **Update_Memory:** Capitulate on the present state of overall logical flow to achieve the final goal of generating answer. Depending on the all previous stages, generate the feedback which is useful for next iteration.

Lets' formally define the full flow : 

$$
\boxed{q_j = QG(x, prompt, M_{j-1})} \qquad (3)
$$

$$
\boxed{D_j = KR(K, q_j)} \qquad (4)
$$

$$
\boxed{I_j = FC(q_j, D_j, M_{j-1})} \qquad (5)
$$

$$
\boxed{r_j = RG(q_j, I_j)} \qquad (6)
$$

$$
\boxed{M_j = MU(x, M_{j-1}, I_j, prompt, (q_j,r_j ) )} \qquad (7)
$$


$$
T = \{S_0, a_0, S_1, a_1, \ldots, a_{n-1}, S_n\}
$$

Here T is set of steps generated in the full trace of steps.
The question is x.

$$
\boxed{S_j = (q_j, r_j, M_j)}
$$

- $q_j$: the **current sub-question** being solved.
- $r_j$: the **intermediate response** generated for that sub-question.
- $M_j$: the **working memory** after step $i$.

where

$$
S_0 = (x, \emptyset, \emptyset)
$$

$$
\boxed{a_j : S_j \rightarrow S_{j-1} }
$$

Here, QG/RG are "Generation" module, KR is "Retriever" module, and FC/MU are "Extractor" module.


### Training via Path-Outcome Dual Rewards

If we want to train the model now for Extrcator there is a tradeoff to consider : A memory decision made at an early reasoning step may only affect the final
answer several steps later. Therefore, using only a final reward produces very sparse feedback. On the other hand, optimizing only intermediate steps may encourage locally good decisions that ultimately lead to an incorrect final answer. hence we need to optimize for boththe rewards.

The reward options in training are **intermediate reward assignment** and **final reward assignment**. There are multiple ways one can assign intermediate and final reward. For final reward we can use enterprise-FM or locally trained reward model or direct human in loop or data with labels/answers. Depending on stages of the RL training, the end performance expectation, budget etc one may choose different variations of it. For intermediate reward assignment, one mostly can use existing FMs to judge intermediate steps.

#### Intermediate Path Reward

At each intermediate reasoning step, an LLM judge evaluates the quality of
the generated response:

$$
\boxed{R_p(r_j,q_j) = \operatorname{JUDGE}_{\mathrm{PATH}}(r_j,q_j)} \qquad (8)
$$

The path reward measures whether the information extracted for the current step was useful for answering $q_j$.

It evaluates properties such as:
- relevance,
- sufficiency,
- logical coherence, and
- factual consistency.

Therefore, $R_p$ provides **local feedback** on the quality of the
memory decision at step $i$.

#### Final Outcome Reward

After the complete reasoning trajectory, the final answer $y$ is evaluated against the original question $x$ and ground-truth answer $y^*$:

$$
\boxed{R_o(y,x,y^*) = \operatorname{JUDGE}_{\mathrm{OUTCOME}}(y,x,y^*)} \qquad (9)
$$

Here:

- $y$: generated final answer,
- $x$: original question,
- $y^*$: ground-truth answer.

Unlike the path reward, $R_o$ evaluates the **global success of the
complete reasoning process**.

#### Combined Training Objective

The Extractor is trained using both types of rewards:

$$
\boxed{\frac{1}{n-1}\sum_{i=1}^{n-1}R_p(q_i,r_i)+\lambda R_o(y,x,y^*)} \qquad (10)
$$

where:

- $n$: number of reasoning states,
- $R_p$: intermediate/path reward,
- $R_o$: final outcome reward,
- $\lambda$: weight controlling the importance of the global outcome reward.

Conceptually, the objective is:

$$
\text{Training Signal} = \underbrace{\text{Local Path Quality}}_{\text{Is the current memory useful?}} + \lambda \underbrace{\text{Global Outcome Quality}}_{\text{Did the full reasoning process succeed?}}
$$

This combination allows the Extractor to learn memory policies that are both
**locally useful** and **globally beneficial**.

Since we want to maintain a random distribution of reward over local and global rewards so that its effective in training loop, while generating the traces for a question we can choose random place in the trace path which we can use for training the "Extractor". We may not choose all possible intermediate steps and final step fro training "Extrcator" since it will be prohibitibly epxensive. A vanilla version of this is where we just generate the traces, dont do any intermediate rewarding, do only final rewarding. But since we want to also improve on the intermeidate steps - specially on the FC and MU stages - we can incorporate the reward signals at these steps.

The training was done on GPU clusters managed by SLURM. Through Ray we managed the servers, and then verl was used for GRPO RL training. More on the training details is on later article.



