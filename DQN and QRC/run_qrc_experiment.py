import random
from experiment import Experiment, QRC_AGENT, DQN_Agent

def generate_seeds(n, seed_range=(0, 10000)):
    seeds = [random.randint(seed_range[0], seed_range[1]) for _ in range(n)]
    return seeds


experiment = Experiment(agent_name=QRC_AGENT)
seeds = generate_seeds(100)
print("Running experiment with seeds:", seeds)
# experiment.run_single_visual(seed=seeds[0])
experiment.run_multiple_visual(seeds=seeds)
# print("Running experiment with seeds:", seeds)
# experiment.run_agents_sequential_multiple_seeds(seeds=seeds, qrc_file="qrc_experiment_results.txt", dqn_file="dqn_experiment_results.txt")