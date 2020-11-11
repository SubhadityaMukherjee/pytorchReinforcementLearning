import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

MAXSTATES = 0

# Decide how many states are possible and return possible
def max_dict(d):
    max_v = float('-inf')
    for key, val in d.items():
        if val > max_v:
            max_v = val
            max_key = key
    return max_key, max_v

# Discretize the problem. aka convert continious to small bins to run on
def create_bins():
    # These are defined by the environment itself
    # obs[0] -> cart position --- -4.8 - 4.8
    # obs[1] -> cart velocity --- -inf - inf
    # obs[2] -> pole angle    --- -41.8 - 41.8
    # obs[3] -> pole velocity --- -inf - inf
    bins = np.zeros((4, 10))
    bins[0] = np.linspace(-4.8, 4.8, 10)
    bins[1] = np.linspace(-5, 5, 10)
    bins[2] = np.linspace(-.418, .418, 10)
    bins[3] = np.linspace(-5, 5, 10)
    return bins

# Give bins to the env
def assign_bins(obs, bins):
    return [np.digitize(obs[i], bins[i]) for i in range(4)]

# Convert the states to a string for easier understanding
def get_all_states_str():
    return [str(i).zfill(4) for i in range(MAXSTATES)]

# Convert the current state to a string
def get_state_str(state):
    return ''.join(str(int(e)) for e in state)

# Initialize the table
def init_Q(env):
    Q = {}

    all_states = get_all_states_str()
    # Init by making them 0
    for state in all_states:
        Q[state] = {}
        for action in range(env.action_space.n):
            Q[state][action] = 0
    return Q


# Play once; Either random move or from Q table
def play_once(env, bins, Q, ALPHA, GAMMA, eps=0.5):
    obs = env.reset()
    done = False
    cnt = 0
    state = get_state_str(assign_bins(obs, bins))
    total_reward = 0

    while not done:
        cnt += 1
        # Choose random
        if np.random.uniform() < eps:
            act = env.action_space.sample()
        else:
            # Choose from table
            act = max_dict(Q[state])[0]

        # One step through env with action
        obs, reward, done, _ = env.step(act)

        # Get reward
        total_reward += reward

        if done and cnt < 200:
            reward = -300

        state_new = get_state_str(assign_bins(obs, bins))

        a1, max_q_s1a1 = max_dict(Q[state_new])
        # Calc new Q
        Q[state][act] += ALPHA*(reward + GAMMA*max_q_s1a1 - Q[state][act])
        state, act = state_new, a1

    return total_reward, cnt


def play_multiple(args, env, ALPHA, GAMMA, bins, N=10000):
    Q = init_Q(env)

    length = []
    reward = []

    # Run the above N times
    for n in tqdm(range(N)):
        eps = 1.0 / np.sqrt(n-1)

        episode_reward, episode_length = play_once(
            env, bins, Q, ALPHA, GAMMA, eps)
        env.render()  # Comment this if you are on jupyter
        if n % args.log == 0:
            print(f"{n}:- Eps: {eps}, Rew: {episode_reward}")
        length.append(episode_length)
        reward.append(episode_reward)
    return length, reward


# Plot avg of the rewards
def plot_running_avg(totalrewards):
    N = len(totalrewards)
    runningAvg = np.empty(N)

    for t in range(N):
        runningAvg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])
    plt.plot(runningAvg)
    plt.title("Running Avg")
    plt.savefig("./runningAvg.png")
