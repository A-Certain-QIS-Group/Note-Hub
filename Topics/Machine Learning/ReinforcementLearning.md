# Notes on Reinforcement Learning
*Aug.29 2023*
*typesetted on Nov.2 2024*

## References
[HF Tutorial](https://huggingface.co/learn/deep-rl-course/unit0/introduction)

## Overview
- Two models to train
    - Actor $π_θ(s,a)$
        - $\text{Target}=∑_\text{i,t}\text{clips}_ϵ(\frac{π_θ(a_t|s_t)}{π_{θ_\text{old}}(a_t|s_t)},A_t)$
        - $\text{clips}(r,A)=\min(rA,\text{clip}(r,1-ϵ,1+ϵ)A)$
        - $ A_t = (r_t+γV_w(s_{t+a}(s_t,a))) - V_w(s_t)$
        - first lock $π_θ$, sample some path 
        - then update $θ$ within the $ϵ$ range for a few substeps
    - Critic $V_w(s)$
        - $δw = βA_tΔ_w(s_t)$
            - > TODO is there a loss function?
- Outline
    - Policy-Based RL --upgrade-> Actor-Critic RL --upgrade-> Proximal Policy Optimization

## TODO
- reread the hf tutorial and colleat the caveats
    - <https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html>
    - random seed
    - hyperparameter tuning
    - normalize the input

### Code
```python
import gymnasium as gym

agent = PolicyModel()
agent.load_state_dict(torch.load(model_path,weights_only=True))

env = gym.make("InvertedPendulum-v5",render_mode="human") 
    # rendered using pygame

obs,info=env.reset()
terminated=truncated=False
while not terminated and not truncated:
    action,logp=agent.sample_action(torch.tensor(obs).float())
    obs,reward,terminated, truncated,info=env.step(action.numpy())
env.close()
```

## Policy-Gradient RL
- Cumulate Reward $R(τ)$
    - $R(τ) = r_{t+1} + γ r_{t+2} + γ^2 r_{t+3} + ... $
    - time decayed summation of future rewards
- Policy $π_θ(a|s)$ 
    - probability distribution over the actions $a$ an agent might perform, given the environment state $s$
    - modeled by a neural network, θ are the weights
- Objective Function $J(θ)=E_{τ∼π}[R(τ)]$
    - $J(θ) = ∑_τ ∏_t P_\text{env}(s_{t+1}|s_t,a_t)π_θ(a_t|s_t)R(τ)$
    - the expectation of Cumulate Reward a policy might achieve
    - $P_\text{env}(s_{t+1}|s_t,a)$ Markovian environment dynamics
    - goal is to maximize $J$ by tuning $θ$
- Training process
    - $θ ← θ + α∇_θ(∑_\text{paths}∑_t\log π_θ(a_t|s_t)R(τ))$
    - it is mathematically equivalent to gradient descendent because Policy Gradient Theorem:
        - $∇_θJ(θ) = E_{π_θ}[∑_t ∇_θ\log π_θ(a_t|s_t)R(τ)]$
        - sketch of proof: $∇(x_1x_2...) = x_1x_2...∑_i∇\log x_i$
    - $E_{π_θ}[⋅]$ is estimated using Monte Carlo
        - vs TimeDifferential
    - Should sample multiple episodes before gradient ascend $θ$
> What is τ

### Code
```python
class PolicyModel(nn.Module):
    def sample_action(self, x):
        mu,sigma=self.forward(x) 
            # the mean and std of the output policy
        dist=Normal(mu,sigma+1e-6)
        action=dist.sample()
        logp=dist.log_prob(action)
        return action,logp

def calculate_cumulative_reward(rewards,gamma):
    cum_reward,cum_rewards=0,[]
    for r in rewards[::-1]:
        cum_reward = r + gamma * cum_reward
        cum_rewards.insert(0,cum_reward)
    return cum_rewards

class ReinforceTrainer:
    def run_episode(self,seed=42):
        logps,rewards=[],[]
        obs,info=self.env.reset(seed=seed)
        terminated=truncated=False
        while not terminated and not truncated:
            action,logp=self.model.sample_action(
                torch.tensor(obs).float())
            obs,reward,terminated, truncated,info=
                self.env.step(action.numpy())
            logps.append(logp)
            rewards.append(reward)
        return logps,rewards
    def update_model(self,logps,rewards):
        cum_rewards=torch.tensor(
            calculate_cumulative_reward(rewards,self.gamma))
        logps=torch.stack(logps)
        loss=-torch.sum(logps*cum_rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

```


## Advantage of Policy-Based RL
- value-based RL:
    - train a value function $Q(s,a)$ or $V(s)$
    - $π(s) = \argmax_a Q(s,a)$
- Policy-based outputs a probability distribution of possible actions, instead of stick to a "standard solution" at the same scenario every time

- advantages of policy based RL:
    - no hard-coded exploration/exploitation trade-off
    - overcomes perceptual aliasing problem
        - when partial information is known, it is bad to have a fixed solution which only depends on the known information
    - compatible with high dimensional/continuous action space
    - good convergence
        - in value-based RL, small change of weights might result in abrupt change of the best policy
- disadvantages of policy based RL:
    - stuck in local minimum
    - slower to train
    - high variance in MC path sampling

- Actor-Critic RL
    - use value function $Q(s,a)-V(s)$ to replace per-path rewards

## Advantage Actor-Critic Method (A2C)
- Two models to train
    - Actor $π_θ(s,a)$
    - Critic $V_w(s)$
- Time Differential
    - $s_t \xrightarrow{π} a_t \xrightarrow{\text{env}} r_t,s_{t+1} → δθ, δw$
    - at each step, after actor's action and then env's feedback, the Actor and Critic model updates by $δθ$ and $δw$
- Advantage $A(s,a)$
    - $A(s,a) = Q(s,a) - V(s)$
    - the advantage of an action when performed at state $s$, compired to empirical state-value $V(s)$
- Time Differential Error of $V$
    - $ A_t = (r_t+γV_w(s_{t+a}(s_t,a))) - V_w(s_t)$
    - used to estimate the Advantage function
    - can use more than 1 steps
- Training Process
    - $δθ = α∇_θ(\log π_θ(s_t,a_t))A_t$
    - $δw = βA_t∇_w V_w(s_t)$
> Question: can we use shared weights between A and C? will it have oscillation problem?

## Proximal Policy Optimization
- <arXiv:1707.06347v2>
- advantage of the action than others
    - > ???
- Problem
    - smaller gradient update ⟹ more likely converge to optimal
    - big step ⟹ falling off the cliffs
    - However, some loss landscape resembles like valleys, where one need to navigate along small gradient on a lower dimensional manifold ("floor"), without being mislead by the large gradient signal on the "cliff" surrounding it
- Naive idea
    - clip the magnitude of $δπ$ in each iteration
- Clipped surrogate function
    - $L_t(θ) = E_t[\min( r_t(θ)A_t, \text{clip}(r_t(θ), 1-ϵ, 1+ϵ)A_t)]$
    - unclipped objectivity function
        - $L_t(θ) = E_t[\log π_θ(a_t|s_t)A_t]$
    - $r_t(θ) = \frac{π_θ(a_t|s_t)}{π_{θ_\text{old}}(a_t|s_t)}$
    - $ϵ∼0.2$
- Behavior of $\min(rA, \text{clip}(r,1-ϵ,1+ϵ)A)$:
    - when $A>0$, the gradient at $r>1+ϵ$ is clipped
        - $θ$ cannot go further to reinforce the behavior which leads to reward
    - when A<0, the gradient at $r<1-ϵ$ is clipped
        - $θ$ cannot go further to unreinforce the behavior which leads to peanlty
    - the two clipped cases:
        - then $θ$ overshoots into such regions (overlearning), it will stuck it and no longer receive gradient information
    - the two unclipped cases:
        - if $θ$ overshoots into such regions (underlearning), there is gradient to bring them back as a safe barrier

- Entropy Bonus
    - $L_S = \frac{1}{β_S} S(π_θ(⋅|s))$
    - encourages exploration at early training
        - > need to vary w.r.t. t?
    - can avoid dirac-delta-like policy in deterministic environment

- Training Process (pseudocode)
    ```python
    for _ in num_iter:
        π_old = π
        for i in num_path_samples:
            # collect path τ_i by running π_old
            # compute A_{i,t}
        for _ in num_grad_steps:
            # optimize ∑_{i,t}L_{i,t} w.r.t. θ
            # averaging over all timeframes
    ```

## Supplemental: Perceptual Aliasing
|   A   |   B   |   C   |   D   |   E   |
|-------|-------|-------|-------|-------|
| avoid | wall  | goal  | wall  | avoid |
| floor | floor | floor | floor | floor |
- perceptual aliasing
    - a robot what can only look forward cannot distinguish B and D
    - determinstic policy:
        - `if obs==wall: go right`
        - will stuck in D
    - probabilistic policy:
        - `if obs==wall: go left or right`

## Supplemental: Q-Learning (Value-Based)
- Cumulated Reward
    - $R(τ) = r_{t+1} + γ r_{t+2} + γ^2 r_{t+3} + ... $
- Q-function $Q(s,a)$
    - estimates the cumulated reward after taken action $a$ under state $s$
    - modeled by a neural network, or a simple look up table
    - One can also use $V(s)$ instead
- Policy
    - $π(s) = \argmax_a Q(s,a)$
    - choose the best known action
- Training Process
    - $Q(s_t,a_t) ← Q(s_t,a_t) + α((r_{t+1} + γ \max_a \hat Q(s_{t+1},a))-Q(s_t,a_t))$
        - $\hat Q$ is a snapshot of $Q$ a few steps ago
- ϵ-greedy policy $π_ϵ$
    - have probability $ϵ$ to explore a random new action
- Time Difference vs Monte Carlo
    - TD = update Q at each step
    - MC = update Q at end of episode
    - be aware of TD error 
        - > ?
- On-policy vs Off-policy
    - use $π_ϵ$ for choosing sample path
    - use $π$ for estimating futher reward
- Fix Q-Target
    - we are using the same parameter to estimate the Q-target $(r_{t+1} + γ \max_a \hat Q(s_{t+1},a))$ and Q-estimation
    - chasing a moving target lead to significant oscillation in training
        - "Cowboy chasing a moving cow"
    - use separate network with fixed parameters for estimating TD target
        - only update them every C steps
- Replay Memory

## Supplemental: Cowboy Problem
