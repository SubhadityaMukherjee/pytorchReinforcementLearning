import gym
import argparse as ap
from utils import *
import utils

# Get all required args
argus = ap.ArgumentParser("Get required")
argus.add_argument("--n", help="No epochs/iterations", type=int, default=10000)
argus.add_argument("--e", help="Environment. Eg: CartPole-v0",
                   type=str, default="CartPole-v0")
argus.add_argument("--a", help="Alpha (learning rate)",
                   type=float, default=.001)
argus.add_argument("--log", help="Log every x epoch", type=int, default=1000)
argus.add_argument("--early", help="Stop if isnt improving in last et steps", type=bool, default=False)
argus.add_argument("--et", help="Choose how many steps to consider for early stopping", type=int, default=40)
args = argus.parse_args()

print(f"Using Environment: {args.e}")
epochs, environ, ALPHA = args.n, args.e, args.a

# Create environment
env = gym.make(environ)
print(env)

# Define params
MAXSTATES = 10**4
utils.MAXSTATES = MAXSTATES
GAMMA = .9 # Exploration vs exploitation term
EPS = 1.0

# Discretize
# bins = assign_bins()

# Run the simulation
episode_rewards = play_multiple(
    args, env, ALPHA, GAMMA, EPS, bins, N=epochs)

# Plot saved to runningAvg.png
plot_running_avg(episode_rewards)
