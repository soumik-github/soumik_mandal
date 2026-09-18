---
title: RL algorithms
---

## Overview


## Vanilla policy gradient

A trajectory is the sequence of state-action pairs produced by the policy,

$$
\tau = \big\{(s_i, a_i)\big\}_{i \ge 0}, \qquad \tau \sim \pi_{\theta}
$$

and the objective we want to maximize is the expected return over such trajectories:

$$
\boxed{\ \max_{\theta} \ \mathbb{E}_{\tau \sim \pi_{\theta}}\big[R(\tau)\big]\ }
$$

The return of a full trajectory, and the return-to-go from step $t$ onwards, are

$$
R(\tau) = \sum_{i=0}^{\infty} r_i, \qquad G_t = \sum_{i=t}^{\infty} r_i
$$

(in practice a discount factor $\gamma \in (0,1]$ is folded in, giving $G_t = \sum_{i \ge t} \gamma^{\,i-t} r_i$, which keeps the sum finite). The per-step reward is whatever we decide it to be - in the LLM setting it comes from a reward model $\theta_r$:

$$
r_i = r_{\theta_r}(s_i, a_i)
$$

The VPG / REINFORCE gradient of the objective $J(\theta_{\text{policy}})$ is then

$$
\nabla_{\theta} J(\theta_{\text{policy}}) \;=\; \mathbb{E}_{\tau}\left[ \sum_{t} \nabla_{\theta} \log \pi_{\theta_{\text{policy}}}(a_t \mid s_t) \; G_t \right]
$$

### Value, action-value and advantage

Three quantities built on top of $G_t$ are used everywhere in what follows:

$$
V(s_t) \;=\; \mathbb{E}_{\pi_{\theta}}\big[\,G_t \mid s_t\,\big]
$$

$$
Q(s_t, a_t) \;=\; \mathbb{E}_{\pi_{\theta}}\big[\,G_t \mid s_t,\, a_t\,\big]
$$

$$
\boxed{\ A(s_t, a_t) \;=\; Q(s_t, a_t) - V(s_t)\ }
$$

- $V(s_t)$ is the **state-value function**: the return we expect from state $s_t$ if we keep following the policy. It is action-agnostic, which is exactly what makes it usable as a baseline.
- $Q(s_t, a_t)$ is the **action-value function**: the return we expect after committing to action $a_t$ in state $s_t$ and following the policy thereafter. The two are linked by $V(s_t) = \mathbb{E}_{a \sim \pi_{\theta}}\big[Q(s_t, a)\big]$.

    In the LLM setting this relation simplifies a lot. The state is the prompt plus everything generated so far, the action is the next token, and the environment transition is *deterministic* - once $a_t$ has been sampled, the next state is just the concatenation

    $$
    s_{t+1} = s_t \oplus a_t
    $$

    with no environment stochasticity left to average over. All the randomness sits in the policy's sampling of $a_t$, not in where that token takes us. Hence

    $$
    \boxed{\ Q(s_t, a_t) \;=\; r_t + \gamma\, V(s_{t+1}) \;\xrightarrow[\ \gamma = 1\ ]{\ r_t = 0\ } \; V(s_{t+1})\ }
    $$

    where the reduction on the right holds in the usual LLM reward setting: the reward is terminal (assigned once to the finished response, so the intermediate per-token rewards $r_t$ are zero) and undiscounted. So $Q$ at the current token *is* $V$ at the next state, and the advantage collapses to a difference of two values of the same function:

    $$
    A(s_t, a_t) \;=\; V(s_{t+1}) - V(s_t)
    $$

- $A(s_t, a_t)$ is the **advantage**: how much better action $a_t$ is than the policy's average behavior in that state. By construction $\mathbb{E}_{a \sim \pi_{\theta}}\big[A(s_t, a)\big] = 0$, so replacing $G_t$ with $A(s_t, a_t)$ in the gradient above leaves it unbiased while cutting its variance - this is the baseline trick, and it is what PPO's critic estimates.



### How to estimate $\nabla_{\theta} J(\theta)$

The gradient is an expectation over trajectories, which we never have in closed form - we approximate it with $N$ sampled rollouts:

$$
\nabla_{\theta} J(\theta_{\text{policy}}) \;\approx\; \frac{1}{N} \sum_{n=1}^{N} \sum_{t} \nabla_{\theta} \log \pi_{\theta_{\text{policy}}}\big(a_t^{(n)} \mid s_t^{(n)}\big) \; \hat{G}_t^{(n)}
$$

So there are exactly two things to estimate inside the sum:

- **Estimate $\hat{G}_t$.** The naive estimator is the return-to-go of one single rollout,

    $$
    \hat{G}_t = \sum_{i \ge t} r_i^{(n)}, \qquad \mathbb{E}\big[\hat{G}_t\big] = G_t
    $$

    which is unbiased but has high variance, since $\operatorname{Var}(\hat{G}_t) \propto 1/N$ and in practice $N$ is small. To handle this we later resort to a baseline $b(s_t)$, replacing $\hat{G}_t$ with $\hat{G}_t - b(s_t)$. There are different ways to implement this.

- **Estimate $\nabla_{\theta} \log \pi_{\theta_{\text{policy}}}(a_t \mid s_t)$.** This one is free - the forward pass already produces the log-probability of the sampled token, and autograd gives its gradient:

    $$
    \log \pi_{\theta_{\text{policy}}}(a_t \mid s_t) = \log \operatorname{softmax}\big(z_{\theta}(s_t)\big)_{a_t}
    $$

    In VPG this comes from $\pi_{\theta_{\text{policy}}}$ alone. Later algorithms involve both $\pi_{\theta_{\text{policy}}}$ and $\pi_{\theta_{\text{old}}}$, through the importance ratio

    $$
    \rho_t(\theta) = \frac{\pi_{\theta_{\text{policy}}}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
    $$

### Who is optimizing what

- **RL is different from supervised learning / IFT** in the sense that in SL the data is fixed beforehand. In RL we have a policy from which we generate the data that is then used to optimize that same policy. So the two go in tandem, in a circle. This is what makes RL different, and also hard and unstable.
- **Adam is supposed to be optimizing $J(\theta)$.** Nobody hands us the reward landscape directly. But given $\hat{G}_t$ held fixed, in VPG we can get the reward landscape with respect to $\theta$ at that fixed policy. Even though the policy and the samples keep changing as training progresses, at each step we still have a landscape with respect to $\theta$ at a fixed policy - and we hope Adam solves for it, given that $\hat{G}_t$ has already been reliably estimated. Adam acting through the policy alone will not get us there.
- **But then why do we need another estimator for $G_t$?** Because it takes a separate effort of its own - which is why there are so many variations of $G_t$ estimation. $G_t$ is difficult to estimate from individual traces: each time we cannot afford to have enough traces to estimate it well. Hence we need a model to estimate this.

At this point we are done with VPG, REINFORCE. Next we foray into baselines, PPO, GRPO etc.

### Baseline

The general formulation we want at the end is one where the return splits into an action-dependent term and a state-only term:

$$
\boxed{\ G_t \;=\; m(s_t, a_t) - n(s_t)\ }
$$

This is allowed because subtracting any function of the state alone leaves the gradient unchanged:

$$
\mathbb{E}_{\tau}\left[ \sum_{t} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)\, G_t \right] \;=\; \mathbb{E}_{\tau}\left[ \sum_{t} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)\, \big(G_t - b(s_t)\big) \right]
$$

**Why the baseline term vanishes in expectation.** Split the expectation over the trajectory into an outer one over the state and an inner one over the action taken there (tower property). The inner one is where $b$ lives, and $b(s_t)$ is a constant with respect to $a_t$, so it comes straight out of the sum:

$$
\begin{aligned}
\mathbb{E}_{a_t \sim \pi_{\theta}(\cdot \mid s_t)}\Big[ \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)\, b(s_t) \Big]
&= b(s_t) \sum_{a} \pi_{\theta}(a \mid s_t)\, \nabla_{\theta} \log \pi_{\theta}(a \mid s_t) \\[4pt]
&= b(s_t) \sum_{a} \nabla_{\theta} \pi_{\theta}(a \mid s_t) \\[4pt]
&= b(s_t)\, \nabla_{\theta} \underbrace{\sum_{a} \pi_{\theta}(a \mid s_t)}_{=\,1} \;=\; b(s_t)\, \nabla_{\theta} 1 \;=\; 0
\end{aligned}
$$

The middle step is the log-derivative identity $\pi \nabla \log \pi = \nabla \pi$, and the last one is just the fact that the policy is a normalized distribution over the vocabulary - its total mass is $1$ for every $\theta$, so the gradient of that mass is zero. This is the whole trick: the baseline is invisible to the *mean* of the estimator, no matter what $b$ we pick, as long as it does not depend on $a_t$.

This baseline reduces variance too.

So we use

$$
G_t \;\longleftarrow\; G_t - b(s_t) \qquad \text{(baseline)}
$$


## PPO

### Start with TRPO

The trajectories are sampled from the old policy $\pi_{\theta_{\text{old}}}$ but the objective is evaluated for the policy being updated, so we need importance sampling to correct for that mismatch:

$$
J(\theta) \;=\; \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}}\left[ \sum_{t} \frac{\pi_{\theta_{\text{policy}}}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \; A(s_t, a_t) \right] \;=\; \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}}\left[ \sum_{t} \rho_t(\theta)\, A(s_t, a_t) \right]
$$

with the advantage $A(s_t, a_t)$ also estimated under $\pi_{\theta_{\text{old}}}$. On its own this ratio is unbounded, so the objective is maximized subject to a **trust region** - the constraint that the updated policy stays within a KL ball of the reference policy:

$$
\boxed{\ \max_{\theta} \ \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}}\Big[ \textstyle\sum_{t} \rho_t(\theta)\, A(s_t, a_t) \Big] \quad \text{subject to} \quad D_{\mathrm{KL}}\big(\pi_{\theta_{\text{ref}}} \,\|\, \pi_{\theta_{\text{policy}}}\big) \le \delta\ }
$$

### Trust region reduces to clipping

Solving that constrained problem exactly is expensive. PPO replaces the explicit KL constraint with a clip on the ratio itself - $\delta$ and $\epsilon$ are related, both bounding how far one update is allowed to move the policy. The full PPO objective with the clipping enforced is

$$
\boxed{\ J^{\text{CLIP}}(\theta) \;=\; \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}}\left[ \sum_{t} \min\Big( \rho_t(\theta)\, A_t, \;\; \operatorname{clip}\big(\rho_t(\theta),\, 1-\epsilon,\, 1+\epsilon\big)\, A_t \Big) \right]\ }
$$

where $\rho_t(\theta)$ is the importance ratio defined above and $A_t = A(s_t, a_t)$. The $\min$ of the clipped and unclipped terms is what makes the bound one-sided: the update is free to move the ratio back towards $1$, but gains nothing from pushing it beyond $1 \pm \epsilon$, so once a step would take the policy outside the trust region the gradient through that term goes to zero.

In practice this is optimized together with the critic's value loss:

$$
\mathcal{L}(\theta) \;=\; -\,J^{\text{CLIP}}(\theta) \;+\; c_1 \underbrace{\mathbb{E}\big[(V_{\theta_{\text{critic}}}(s_t) - \hat{G}_t)^2\big]}_{\text{value loss}}
$$

where $c_1$ weights the value loss against the policy objective.

### How the critic is trained

The critic is trained by plain regression - there is no separate reward signal for it, its targets are built out of the reward model's outputs on the rollouts we already have.

For a trajectory of length $T$ sampled from $\pi_{\theta_{\text{old}}}$, the reward model scores each step, and the target for state $s_t$ is the sum of those rewards from that state to the end of the trajectory:

$$
\hat{G}_t \;=\; \sum_{i=t}^{T} \gamma^{\,i-t}\, r_{\theta_r}(s_i, a_i)
$$

This $\hat{G}_t$ is a sampled estimate of exactly what the critic is supposed to output, since $V(s_t) = \mathbb{E}\big[G_t \mid s_t\big]$. So the critic is fit to it by minimizing the squared error:

$$
\boxed{\ \mathcal{L}(\theta_{\text{critic}}) \;=\; \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}}\left[ \sum_{t} \big( V_{\theta_{\text{critic}}}(s_t) - \hat{G}_t \big)^{2} \right]\ }
$$

which is the same value-loss term that appears in the PPO objective above. The fitted $V_{\theta_{\text{critic}}}(s_t)$ is then used as the baseline that turns the raw return into the advantage

$$
A_t \;=\; \hat{G}_t - V_{\theta_{\text{critic}}}(s_t)
$$

so the two models feed each other: the rollouts give the critic its regression targets, and the critic gives the policy gradient its baseline.

## GRPO

GRPO is PPO with $V(s)$ replaced by an average over multiple rollouts - the rest is all the same. For a prompt we sample a group of $G$ rollouts $\{\tau^{(1)}, \ldots, \tau^{(G)}\}$ from $\pi_{\theta_{\text{old}}}$, and the baseline becomes the empirical mean of their returns:

$$
\boxed{\ V(s) \;\longrightarrow\; \frac{1}{G} \sum_{g=1}^{G} \hat{G}^{(g)} \ }
$$

so that the advantage of the $g$-th rollout is its own return, centered (and usually scaled) by the statistics of its own group:

$$
A^{(g)} \;=\; \frac{\hat{G}^{(g)} - \operatorname{mean}\big(\{\hat{G}^{(j)}\}_{j=1}^{G}\big)}{\operatorname{std}\big(\{\hat{G}^{(j)}\}_{j=1}^{G}\big)}
$$

Everything else - the importance ratio, the clipped objective, the trust region - carries over from PPO unchanged.

The models involved are

$$
\pi_{\theta_{\text{policy}}}, \quad \pi_{\theta_{\text{old}}}, \quad \pi_{\theta_{\text{ref}}}, \quad \pi_{\theta_r}, \quad \pi_{\theta_{\text{critic}}}
$$

all five of which are applicable for PPO. For GRPO there is no $\theta_{\text{critic}}$, since the group average has taken over its job.

## DPO

A different flavor is DPO - the supervised training version of preference tuning.

### The Bradley-Terry model

Bradley-Terry is a model of *pairwise preference*. Given a prompt $x$ and two candidate responses, it says the probability that one is preferred over the other depends only on the difference of their rewards, squashed through a sigmoid:

$$
\boxed{\ p^{*}(y_1 \succ y_2 \mid x) \;=\; \sigma\big(r^{*}(x, y_1) - r^{*}(x, y_2)\big)\ }
$$

The consequence is that only reward *differences* are identifiable - shifting $r^{*}$ by any constant leaves every preference probability unchanged. This is what a reward model is normally trained to fit, and it is also the door DPO walks through.

### From the RLHF objective to DPO

The RLHF objective is reward maximization with a KL leash to the reference policy:

$$
\max_{\theta} \ \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_{\theta}(y \mid x)}\big[r(x,y)\big] \;-\; \beta\, \mathbb{D}_{\mathrm{KL}}\big[\pi_{\theta}(y \mid x) \,\Vert\, \pi_{\text{ref}}(y \mid x)\big]
$$

This particular problem has a closed-form optimum - the reference policy reweighted by the exponentiated reward:

$$
\pi^{*}(y \mid x) \;=\; \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, e^{\frac{1}{\beta} r^{*}(x,y)}, \qquad Z(x) = \sum_{y} \pi_{\text{ref}}(y \mid x)\, e^{\frac{1}{\beta} r^{*}(x,y)}
$$

Rearranging it gives the reward *in terms of the policies*, which is the key move - the reward model is no longer a separate network, it is implicit in the ratio between the trained policy and the reference:

$$
r^{*}(x,y) \;=\; \beta \log \frac{\pi^{*}(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \;+\; \beta \log Z(x)
$$

The partition function $Z(x)$ is intractable, but Bradley-Terry only ever sees a *difference* of two rewards for the same prompt - so the $\beta \log Z(x)$ term appears twice with opposite signs and cancels exactly.

### The DPO loss

What is left is a plain binary cross-entropy over preference pairs $(x, y_w, y_l)$, with $y_w$ the chosen and $y_l$ the rejected response:

$$
\boxed{\ \mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) \;=\; -\,\mathbb{E}_{(x,\, y_w,\, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \left( \log \frac{\pi_{\theta}(y_w \mid x)}{\pi_{\theta}(y_l \mid x)} \;-\; \log \frac{\pi_{\text{ref}}(y_w \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right) \right]\ }
$$

So the final alignment of preference learning happens through supervised training over binary cross-entropy. Note which models appear: only $\pi_{\theta_{\text{policy}}}$ and $\pi_{\theta_{\text{ref}}}$ - there is no reward model, no critic, no rollout, no sampling loop at all.

Most industry lab uses GRPO or PPO or some variations of them for LLM/FM training.

## Where the reward comes from

Since we now know the components of each RL algo here, the last piece is how we get the reward $r_{\theta_r}$ in the first place. There are a few avenues:

- **RLHFAI** (RL from AI feedback) - the human annotator is replaced by a strong enterprise/large-open-source model acting as the judge, which either scores a response directly or labels which of two responses is better. Cheap and scalable enough to generate preference data in bulk, but the policy inherits whatever biases and blind spots the judge model has.
- **RLVR** (RL with verifiable reward) - the reward is computed by a program, not a model: execute the generated SQL/code and compare the result, or match the final answer against the ground truth. Objective and essentially impossible to hack, but it only applies where a checkable ground truth exists.
- **RLHF** (RL from human feedback) - human annotators rank pairs of responses and a reward model $\theta_r$ is fit to those preferences through the Bradley-Terry model above, after which it scores every rollout. The highest-quality signal for subjective qualities like tone and helpfulness, but slow and expensive to collect, and the reward model is only as good as its annotation.

Depending upon use case, data quality, task at hand, annotation strength, budget etc, any one of these method can be used. They can be mixed and matched also.

## Training on reasoning traces

For training the reasoning trace itself - the intermediate steps, not just the final answer - PPO, REINFORCE and DPO can be used, but GRPO cannot. The reason is the granularity of the signal each one produces: PPO and REINFORCE attach a value to every intermediate state through

$$
A(s_t, a_t) \quad \text{per step } t
$$

and DPO compares two traces that diverge at one step, whereas GRPO assigns a single scalar per rollout,

$$
A^{(g)} = \frac{\hat{G}^{(g)} - \operatorname{mean}\big(\{\hat{G}^{(j)}\}\big)}{\operatorname{std}\big(\{\hat{G}^{(j)}\}\big)} \quad \text{per trajectory } g
$$

which says nothing about *which* step in the trace was responsible.

So the order matters. First train the multi-hop reasoning with a step-level method:

$$
M \;\xrightarrow{\ \text{PPO / REINFORCE / DPO on traces}\ } \; M_{\text{multi-hop}} \;\xrightarrow{\ \text{GRPO + RLVR}\ }\; M_{\text{final}}
$$

Once the training on multi-hop is done, we can then do GRPO on RLVR etc. to make the training bigger. This works precisely because the multi-hop generation was already good from the previous stage - the traces coming out of $M_{\text{multi-hop}}$ are reliable enough that a single trajectory-level reward is now sufficient to improve on.

post-training is complex - multiple data, multiple complexity, multiple task etc makes the post-training very involved and complex. There are many hacks and approximation. We can use rlhf, rlhfai, rlvr depending on task, data etc + DPO/PPO/GRPO + different aggregation strategy.

## Issues during training

Below are the main failure modes one runs into once the training is actually running.

### Reward hacking (overfitting)

The objective we optimize is not the objective we actually want. We want some true quality $r^{*}$, but we can only ever write down a proxy $\hat{r}$ - a reward model, a judge, a verifier - and the optimizer maximizes the proxy:

$$
\max_{\theta} \ \mathbb{E}\big[\hat{r}(x,y)\big] \quad \text{as a stand-in for} \quad \max_{\theta} \ \mathbb{E}\big[r^{*}(x,y)\big]
$$

The two agree over the data the proxy was built from, but the policy is actively searching for the places where they come apart - and any gap it finds is free reward. This is the RL version of overfitting, and it applies to all three reward sources here:

- **RLHFAI** - the judge model has its own preferences and biases: it can favor longer answers, confident phrasing, a particular format, or text that looks like its own output. The policy learns to produce whatever the judge rates highly, which is not the same as being correct.
- **RLVR** - the verifier is objective, but only about the thing it checks. The policy can find shortcuts that pass the check without solving the task - special-casing the test inputs, exploiting a loose answer-matching rule, or producing a query that returns the right value for the wrong reason.
- **RLHF** - the reward model is a fixed network trained on a finite set of human comparisons, so it is only valid near that distribution. Pushed far enough, the policy drifts off-distribution and finds inputs where the reward model is simply wrong but scores high.

This is why the KL leash to $\pi_{\theta_{\text{ref}}}$ exists at all - it limits how far the policy can wander in search of these gaps. It is a mitigation, not a cure: the more capable the optimizer, the harder it pushes on whatever the proxy fails to capture.

Later covered topics: 
entropy collapse 
free lunch = multiple data, multiple algorithm, multiple reward -> train separate model and average them -> may lead to better result
GRPO, DPO, PPO - adv / disadv based on data set and tasks (where to use what will be covered later articles)
scalability law of RL


