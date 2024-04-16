# MONTE CARLO CONTROL ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given RL environment using the Monte Carlo algorithm

## PROBLEM STATEMENT
To find the optimal policy for the given RL environnment Frozen Lake using the Monte Carlo algorithm which use both exploration and exploitation for finding the optimal policy.
Exploitation: Agent taking an action based on the current knowledge.
Exploration: Agent taking a new action to learn new information.
The epsilon is used to balance the exploration and exploitation, The value of the epsilon decreases
gradually which indicates sufficient exploration and the agent has gained the new information.



## MONTE CARLO CONTROL ALGORITHM

Initialize the state value function V(s) and the policy π(s) arbitrarily.

Generate an episode using π(s) and store the state, action, and reward sequence.

For each state s appearing in the episode:

G ← return following the first occurrence of s
Append G to Returns(s)
V(s) ← average(Returns(s))
For each state s in the episode:
π(s) ← argmax_a ∑_s' P(s'|s,a)V(s')
Repeat steps 2-4 until the policy converges.

Use the function decay_schedule to decay the value of epsilon and alpha.

Use the function gen_traj to generate a trajectory.

Use the function tqdm to display the progress bar.

After the policy converges, use the function np.argmax to find the optimal policy. The function takes the following arguments:

Q: The Q-table.
axis: The axis along which to find the maximum value.
~~~
Developed by: D.Amarnath Reddy
Reg no:212221240012
~~~
## MONTE CARLO CONTROL FUNCTION
~~~
from tqdm import tqdm
def mc_control(env, gamma = 1.0, init_alpha = 0.5, min_alpha = 0.01,
               alpha_decay_ratio = 0.5, init_epsilon = 1.0, min_epsilon = 0.1,
               epsilon_decay_ratio = 0.9, n_episodes = 3000, max_steps = 200,
               first_visit = True):
  nS, nA = env.observation_space.n, env.action_space.n

  discounts = np.logspace(0,max_steps, num=max_steps,
                          base=gamma, endpoint = False)

  alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)

  epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

  pi_track = []

  Q = np.zeros((nS, nA), dtype = np.float64)
  Q_track = np.zeros((n_episodes, nS, nA), dtype = np.float64)

  select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon_decay_ratio else np.random.randint(len(Q[state]))

  for e in tqdm(range(n_episodes), leave = False):
    trajectory = generate_trajectory(select_action, Q, epsilons[e], env, max_steps)
    visited = np.zeros((nS, nA), dtype = bool)
    for t, (state, action, reward, _, _) in enumerate(trajectory):
      if visited[state][action] and first_visit:
        continue
      visited[state][action] = True
      n_steps = len(trajectory[t:])
      G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
      Q[state][action] += alphas[e] * (G - Q[state][action])
    Q_track[e] = Q
    pi_track.append(np.argmax(Q, axis = 1))
  V = np.max(Q, axis = 1)
  pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis = 1))}[s]
  return Q, V, pi

~~~

## OUTPUT:
Mention the optimal policy, optimal value function , success rate for the optimal policy.

<img width="713" alt="image" src="https://github.com/Dhanireddy-Amarnthreddy/monte-carlo-control/assets/94165103/796ee05b-d4d8-49bd-9613-451ffb76b727">


## RESULT:

Therefore a python program is successfully developed to find the optimal policy for the given RL environment using the Monte Carlo Algorithm.
