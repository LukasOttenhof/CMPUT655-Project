# Hyperparameter Testing Scenarios for Comparing DQN and QRC

We will use the same hyperparameters for both DQN and QRC to compare their performance.

---

## Scenario 1: Default Baseline
- `num_episodes = 1000`
- `max_steps_per_episode = 500`
- `gamma = 0.99`
- `learning_rate = 1e-3`
- `epsilon_start = 1.0`
- `epsilon_decay = 0.99997`
- `epsilon_min = 0.01`
- `batch_size = 64`
- `target_update_freq = 5`
- **QRC only (kept same for comparison):** `alpha = 1.0, sigma_min = 0.0, sigma_max = 1.0`

**Purpose:** Baseline to compare DQN vs QRC under identical hyperparameters.

---

## Scenario 2: Fast Learning
- `learning_rate = 5e-3`
- `gamma = 0.99`
- `epsilon_decay = 0.9999`
- `target_update_freq = 3`

**Purpose:** Test how faster learning rate affects DQN and QRC performance equally.

---

## Scenario 3: More Exploration
- `epsilon_start = 1.0`
- `epsilon_decay = 0.99995`
- `epsilon_min = 0.05`
- `num_episodes = 1500`

**Purpose:** See whether prolonged exploration benefits both algorithms.

---

## Scenario 4: Short-term Memory & Frequent Updates
- `batch_size = 32`
- `target_update_freq = 1`
- `num_episodes = 2000`

**Purpose:** Examine the effect of smaller batches and frequent target updates on both DQN and QRC.

---

## Scenario 5: QRC-Specific Tuning (kept same for comparison)
- `alpha = 0.5`
- `sigma_min = 0.1, sigma_max = 0.5`
- Other hyperparameters same as baseline

**Purpose:** Compare the influence of QRC’s correction term while keeping all other hyperparameters consistent.

