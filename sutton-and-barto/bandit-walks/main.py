from agents import SampleAverageAgent, SteppingAgent, OptimisticAgent, UpperConfidenceBoundAgent
from bandit_machine import FixedBanditMachine, WalkingBanditMachine

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

EXPLORATION_RATE = 0.1
STEP_SIZE = 0.1
UPPER_CONFIDENCE_BOUND_C = 2

EPOCHS = 5_000_000
NUM_HANDLES = 10

MEAN_PAYOUT_VARIANCE = 1
PAYOUT_VARIANCE = 1
PAYOUT_STEP_VARIANCE = 0.01

ROLLING_AVERAGE_WINDOW_SIZE = 200

def rolling_average(x, window_size):
    """Returns the rolling average of a numpy array x"""
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[window_size:] - c[:-window_size]) / float(window_size)

if __name__ == '__main__':
    agents = [
        SampleAverageAgent(NUM_HANDLES, EXPLORATION_RATE),
        SteppingAgent(NUM_HANDLES, EXPLORATION_RATE, STEP_SIZE),
        OptimisticAgent(NUM_HANDLES, EXPLORATION_RATE, STEP_SIZE),
        UpperConfidenceBoundAgent(NUM_HANDLES, UPPER_CONFIDENCE_BOUND_C, STEP_SIZE),
    ]

    machines = [
        FixedBanditMachine(NUM_HANDLES, MEAN_PAYOUT_VARIANCE, PAYOUT_VARIANCE),
        WalkingBanditMachine(NUM_HANDLES, PAYOUT_VARIANCE, PAYOUT_STEP_VARIANCE),

    ]

    progress = tqdm(total=EPOCHS*len(machines)*len(agents), ncols=100)

    results = np.zeros((len(machines), len(agents), EPOCHS))
    expected = np.zeros((len(machines), EPOCHS))
    for i, machine in enumerate(machines):

        for epoch in range(EPOCHS):
            for j, agent in enumerate(agents):
                results[i,j,epoch] = agent.play(machine)
                progress.update(1)

            expected[i,epoch] = machine.get_best_expected_value()
            machine.update()

    progress.close()

    fig, axes = plt.subplots(len(machines), len(agents), figsize=(18,18))
    for i, name in enumerate(['Fixed Slot Machine', 'Walking Slot Machine']):
        for j, agent in enumerate(['Averaging Agent', 'Stepping Agent', 'Optimistic Agent', 'UCB Agent']):
            axes[i,j].plot(rolling_average(results[i,j], ROLLING_AVERAGE_WINDOW_SIZE), label='Agent Performance')
            axes[i,j].plot(rolling_average(expected[i], 10), label='Expected Performance')
            axes[i,j].set_title(f"{agent} On {name}")

    fig.show()