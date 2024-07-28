# MiniCourses

## --- Reinforcement learning ----

### Model-free RL
- Policy optimization
- Q-learning
### Model-based RL
- Learn model
- Given the model

Some mathematical results (from course SF2957 Statictical Machine Learning @ KTH):
### Markov Decision Processes (MDPs)

An MDP is defined by:

- **State Space \( S \)**: A finite set of states.
- **Action Space \( A(s) \)**: A finite set of actions available in each state \( s \).
- **Transition Probability**: \( p(s', r \mid s, a) \) is the probability of transitioning to state \( s' \) and receiving reward \( r \) given the current state \( s \) and action \( a \).
- **Reward Function**: \( r(s, a) = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a] \).
- **Discount Factor**: \( \gamma \), where \( 0 < \gamma < 1 \), determines the importance of future rewards.

### State-Value Function

The state-value function \( v^\pi(s) \) for a policy \( \pi \) is given by:

$$
v^\pi(s) = \mathbb{E}^\pi \left[ \sum_{k=0}^\infty \gamma^k R_{t+k+1} \mid S_t = s \right]
$$

### Action-Value Function

The action-value function \( q^\pi(s, a) \) is:

$$
q^\pi(s, a) = \mathbb{E}^\pi \left[ \sum_{k=0}^\infty \gamma^k R_{t+k+1} \mid S_t = s, A_t = a \right]
$$

### Bellman Equation for State-Value Function

The Bellman equation is:

$$
v^\pi(s) = \mathbb{E}^\pi [R_{t+1} + \gamma v^\pi(S_{t+1}) \mid S_t = s]
$$

### Bellman Optimality Equation

The Bellman optimality equation for \( v^*(s) \) is:

$$
v^*(s) = \sup_{a \in A(s)} \mathbb{E} [R_{t+1} + \gamma v^*(S_{t+1}) \mid S_t = s, A_t = a]
$$

### Bellman Optimality Equation for Action-Value Function
!pip install stable-baselines3[extra]
The Bellman optimality equation for \( q^*(s, a) \) is:

$$
q^*(s, a) = \mathbb{E} [R_{t+1} + \gamma \sup_{a'} q^*(S_{t+1}, a') \mid S_t = s, A_t = a]
$$

"Trade-offs Between Policy Optimization and Q-Learning. The primary strength of policy optimization methods is that they are principled, in the sense that you directly optimize for the thing you want. This tends to make them stable and reliable. By contrast, Q-learning methods only indirectly optimize for agent performance, by training Q_{\theta} to satisfy a self-consistency equation. There are many failure modes for this kind of learning, so it tends to be less stable. [1] But, Q-learning methods gain the advantage of being substantially more sample efficient when they do work, because they can reuse data more effectively than policy optimization techniques"
(https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)