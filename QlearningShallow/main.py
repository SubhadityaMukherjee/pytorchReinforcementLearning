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

# Discretize
bins = create_bins()

# Run the simulation
episode_lengths, episode_rewards = play_multiple(
    args, env, ALPHA, GAMMA, bins, N=epochs)

# Plot saved to runningAvg.png
plot_running_avg(episode_rewards)
