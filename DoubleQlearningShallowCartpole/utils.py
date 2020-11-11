import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pysnooper

MAXSTATES = 0

# Choose the max action based on Q vals
def max_action(Q1, Q2, state):
    values = np.array([Q1[state, a] for a in range(2)])
    action = np.argmax(values)
    return action

# Discretize the problem. aka convert continious to small bins to run on
# These are defined by the environment itself
# obs[0] -> cart position
# obs[1] -> cart velocity
# obs[2] -> pole angle
# obs[3] -> pole velocity
bins = np.zeros((4, 10))
bins[0] = np.linspace(-2.4, 2.4, 10)
bins[1] = np.linspace(-4, 4, 10)
bins[2] = np.linspace(-.20943951, .20943951, 10)
bins[3] = np.linspace(-4, 4, 10)

# Give bins to the env
def assign_bins(obs):
    return tuple([int(np.digitize(obs[i], bins[i])) for i in range(4)])

# Generate state space
def get_all_states():
    states = []
    for i in range(len(bins[0])+1):
        for j in range(len(bins[1])+1):
            for k in range(len(bins[2])+1):
                for l in range(len(bins[3])+1):
                    states.append((i,j,k,l))
    return states

# Initialize the table
def init_Q():
    Q1, Q2 = {}, {}

    all_states = get_all_states()
    # Init by making them 0
    for state in all_states:
        for a in range(2):
            Q1[state, a] = 0
            Q2[state, a] = 0
    return Q1,Q2


# Play once; Either random move or from Q table
# @pysnooper.snoop()
def play_once(env, obs, Q1, Q2, done, ALPHA, GAMMA, eps=0.5):
    epRewards = 0
    while not done:
        s = assign_bins(obs)
        Q1, Q2 = init_Q()
        rand = np.random.random()
        # Choose action
        a = max_action(Q1, Q2, s) if rand< (1-eps) else env.action_space.sample()
        # One step through env with action
        obs_, reward, done, info = env.step(a)
        epRewards += reward
        s_ = assign_bins(obs_)
        rand = np.random.random()
        if rand <= .5:
            a_ = max_action(Q1, Q2, s_)
            Q1[s,a] = Q1[s,a] + ALPHA*(reward + GAMMA*Q2[s_,a_] - Q1[s,a])
        elif rand >0.5:
            a_ = max_action(Q1, Q2, s_)
            Q1[s,a] = Q1[s,a] + ALPHA*(reward + GAMMA*Q1[s_,a_] - Q2[s,a])
        obs = obs_

    return epRewards, obs

# Early stopping code
def checkstop(total_reward):
    total_reward = total_reward[::-1]
    if total_reward[0] <= min(total_reward[1:11]):
        #break if current is less than or eq to the last 10
        return True
    else:
        return False

def play_multiple(args, env, ALPHA, GAMMA,EPS, bins, N=10000):
    Q1, Q2 = init_Q()
    done = False
    total_reward = np.zeros(N)

    # Run the above N times
    for n in tqdm(range(N)):

        obs = env.reset()
        episode_reward, obs = play_once(
            env, obs,Q1,Q2,done, ALPHA, GAMMA, EPS)
        EPS -= 2/(N) if EPS >0 else 0
        total_reward[n] = episode_reward
        if args.early == True and n > args.et:
            try:
                if checkstop(total_reward) == True:
                    print(f"Stopping because not improved in {args.et} epochs")
                    break
            except:
                pass
        env.render()  # Comment this if you are on jupyter
        if n % args.log == 0:
            print(f"{n}:- Eps: {EPS}, Rew: {episode_reward}")
    return total_reward


# Plot avg of the rewards
def plot_running_avg(totalrewards):
    print(f"MAX score obtained: {max(totalrewards)}")
    N = len(totalrewards)
    runningAvg = np.empty(N)

    for t in range(N):
        runningAvg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])
    plt.plot(runningAvg)
    plt.title("Running Avg")
    plt.savefig("./runningAvg.png")
