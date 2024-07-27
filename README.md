# MiniCourses

## Reinforcement learning
Some mathematical results (from course SF2957 Statictical Machine Learning @ KTH)
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

The Bellman optimality equation for \( q^*(s, a) \) is:

$$
q^*(s, a) = \mathbb{E} [R_{t+1} + \gamma \sup_{a'} q^*(S_{t+1}, a') \mid S_t = s, A_t = a]
$$
