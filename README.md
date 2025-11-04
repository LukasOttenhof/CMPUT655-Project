# CMPUT655-Project

- Setup
```bash
conda create -n rl
conda activate rl
pip3 install chex flax jax numpy gymnax matplotlib tqdm distrax
conda install jupyter
jupyter notebook
```

- DQN Pseudocode
```
Initialize Q-network Q(s, a; θ) with random weights θ
Initialize target network Q_target(s, a; θ_target) with θ_target = θ
Initialize replay memory D
Set hyperparameters: gamma, epsilon, epsilon_min, epsilon_decay, batch_size

For each episode:
    Initialize state s

    While not done:
        # Epsilon-greedy action selection
        With probability epsilon:
            a = random action
        Else:
            a = argmax_a Q(s, a; θ)

        Take action a, observe reward r and next state s'
        Store transition (s, a, r, s', done) in D
        s = s'

        # Train Q-network
        If len(D) >= batch_size:
            Sample random minibatch from D:
                states, actions, rewards, next_states, dones

            # Compute target Q-values using target network
            target_Q = rewards + gamma * max_a Q_target(next_states, a) * (1 - dones)

            # Compute predicted Q-values
            pred_Q = Q(states, actions)

            # Compute loss (MSE)
            loss = mean((target_Q - pred_Q)^2)

            # Gradient descent step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Update target network periodically
    If episode % target_update_freq == 0:
        θ_target = θ
```

- QRC Pseudocode
```
Initialize Q-network Q(s, a; θ) with random weights θ
Initialize target network Q_target(s, a; θ_target) with θ_target = θ
Initialize replay memory D
Set hyperparameters: gamma, epsilon, epsilon_min, epsilon_decay, batch_size
Set QRC parameters: alpha, sigma_min, sigma_max

For each episode:
    Initialize state s

    While not done:
        # Epsilon-greedy action selection
        With probability epsilon:
            a = random action
        Else:
            a = argmax_a Q(s, a; θ)

        Take action a, observe reward r and next state s'
        Store transition (s, a, r, s', done) in D
        s = s'

        # Train Q-network with QRC
        If len(D) >= batch_size:
            Sample random minibatch from D:
                states, actions, rewards, next_states, dones

            # Compute DQN target using target network
            target_DQN = rewards + gamma * max_a Q_target(next_states, a) * (1 - dones)

            # Compute correction term
            Q_next_online = Q(next_states, all actions)
            sigma = std_dev(Q_next_online across actions)
            sigma_clipped = clip(sigma, sigma_min, sigma_max)
            correction = alpha * sigma_clipped

            # QRC target
            target_QRC = target_DQN - correction

            # Compute predicted Q-values
            pred_Q = Q(states, actions)

            # Compute loss (MSE)
            loss = mean((target_QRC - pred_Q)^2)

            # Gradient descent step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Update target network periodically
    If episode % target_update_freq == 0:
        θ_target = θ
```