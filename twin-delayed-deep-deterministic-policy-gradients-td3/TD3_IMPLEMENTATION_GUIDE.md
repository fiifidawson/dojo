# TD3 Implementation Guide — How to Use TD3 in Your Own Projects

This guide walks you through everything you need to know to take this TD3 implementation and apply it to your own reinforcement learning projects.

---

## Table of Contents

1. [Prerequisites — What You Need Before Starting](#1-prerequisites)
2. [When to Use TD3 (and When Not To)](#2-when-to-use-td3)
3. [Step-by-Step: Setting Up a New Project](#3-step-by-step-setting-up-a-new-project)
4. [What You Need to Change for Each New Environment](#4-what-you-need-to-change)
5. [Writing the Training Loop (The Missing Piece)](#5-the-training-loop)
6. [Hyperparameter Guide — What Each One Does and How to Tune It](#6-hyperparameter-guide)
7. [Common Pitfalls and How to Avoid Them](#7-common-pitfalls)
8. [Order of Operations — What Comes First, What Comes Last](#8-order-of-operations)
9. [Debugging Checklist](#9-debugging-checklist)
10. [Extending TD3 — Advanced Modifications](#10-extending-td3)

---

## 1. Prerequisites

Before using TD3, make sure you have:

### Software
- **Python 3.8+**
- **PyTorch** (`pip install torch`)
- **NumPy** (`pip install numpy`)
- **Gymnasium** (the maintained fork of OpenAI Gym): `pip install gymnasium`
- For physics-based environments: `pip install gymnasium[mujoco]` or `pip install gymnasium[box2d]`

### Knowledge
- Basic understanding of neural networks (what layers, activation functions, and backpropagation are)
- Basic understanding of RL concepts (states, actions, rewards, episodes, policies, Q-values)
- Familiarity with PyTorch basics (tensors, `forward()`, optimizers)

### Hardware
- **CPU is fine** for simple environments (CartPole, Pendulum, LunarLander)
- **GPU recommended** for complex environments (MuJoCo humanoids, robotics) or if you want faster training

---

## 2. When to Use TD3 (and When Not To)

### Use TD3 when:
- Your environment has **continuous action spaces** (actions are real numbers, not discrete choices)
  - Examples: controlling a robot's joint angles, steering a car, adjusting a chemical process
- You want an **off-policy** algorithm (can learn from old data, sample-efficient)
- You need **stable training** (TD3 is more stable than DDPG)

### Do NOT use TD3 when:
- Your action space is **discrete** (left, right, jump, shoot) — use DQN, PPO, or A2C instead
- You have a **very high-dimensional action space** (100+ dimensions) — TD3 can struggle; consider SAC
- You want the simplest possible algorithm — PPO is easier to tune, though less sample-efficient
- Your environment is **stochastic** and you need a stochastic policy — consider SAC (Soft Actor-Critic) which learns a stochastic policy

### TD3 vs. Alternatives (Quick Comparison)

| Algorithm | Action Space | On/Off Policy | Key Strength |
|-----------|-------------|---------------|--------------|
| **TD3** | Continuous | Off-policy | Stable, fixes DDPG's overestimation |
| **DDPG** | Continuous | Off-policy | Simpler but less stable than TD3 |
| **SAC** | Continuous | Off-policy | Handles exploration better, stochastic |
| **PPO** | Both | On-policy | Easy to tune, great for beginners |
| **DQN** | Discrete only | Off-policy | The classic for discrete actions |

---

## 3. Step-by-Step: Setting Up a New Project

Here's the exact order of steps to go from zero to a working TD3 project:

### Step 1: Choose and Understand Your Environment

```python
import gymnasium as gym

# Create the environment
env = gym.make('Pendulum-v1')

# Inspect it — you MUST understand these before configuring TD3
print("Observation space:", env.observation_space)         # e.g., Box(-inf, inf, (3,), float32)
print("Observation shape:", env.observation_space.shape)   # e.g., (3,)
print("Action space:", env.action_space)                   # e.g., Box(-2.0, 2.0, (1,), float32)
print("Action shape:", env.action_space.shape)             # e.g., (1,)
print("Action low:", env.action_space.low)                 # e.g., [-2.]
print("Action high:", env.action_space.high)               # e.g., [2.]
```

**You need three pieces of information:**
1. **Observation shape** — This becomes `input_dims`
2. **Number of actions** — This becomes `n_actions`
3. **Action bounds** — The agent reads these from `env.action_space.high` and `env.action_space.low`

### Step 2: Copy `td3.py` Into Your Project

Copy the `td3.py` file as-is. You generally don't need to modify it for standard Gymnasium environments.

### Step 3: Create the Checkpoint Directory

```python
import os
os.makedirs('tmp/td3', exist_ok=True)
```

If you skip this, saving models will crash.

### Step 4: Create the Agent

```python
from td3 import Agent

agent = Agent(
    alpha=0.001,          # Actor learning rate
    beta=0.001,           # Critic learning rate
    input_dims=env.observation_space.shape,
    tau=0.005,            # Soft update rate
    env=env,              # Pass the environment (used for action bounds)
    gamma=0.99,           # Discount factor
    update_actor_interval=2,
    warmup=1000,
    n_actions=env.action_space.shape[0],
    max_size=1000000,
    layer1_size=400,
    layer2_size=300,
    batch_size=100,
    noise=0.1
)
```

### Step 5: Write the Training Loop

See [Section 5](#5-the-training-loop) for a complete training loop.

### Step 6: Run and Monitor

Run the training script and watch the scores. TD3 typically needs thousands of episodes to converge.

---

## 4. What You Need to Change for Each New Environment

### Things You MUST Change

| Parameter | Where | How to Determine It |
|-----------|-------|-------------------|
| `input_dims` | Agent constructor | `env.observation_space.shape` |
| `n_actions` | Agent constructor | `env.action_space.shape[0]` |
| `env` | Agent constructor | Your Gym environment object |

These are the **only** things that are strictly required to change. Everything else is tuning.

### Things You SHOULD Tune

| Parameter | Default | When to Change |
|-----------|---------|---------------|
| `noise` | 0.1 | If the action range is large (e.g., [-100, 100]), you may need more noise. If small, less. A good starting point is ~5-10% of the action range. |
| `warmup` | 1000 | Increase for complex environments (5000-10000). Decrease for simple ones. |
| `batch_size` | 100 | 256 is a common choice. Larger batches = more stable but slower updates. |
| `layer1_size`, `layer2_size` | 400, 300 | Increase for complex environments (e.g., 512, 512 or 1024, 512). Decrease for simple ones (e.g., 256, 128). |
| `gamma` | 0.99 | Lower (0.95-0.98) for short-horizon tasks. Keep at 0.99 for most tasks. |
| `alpha`, `beta` | 0.001 | Standard starting point. Try 0.0003 or 0.0001 if training is unstable. |
| `max_size` | 1,000,000 | Reduce if you're running out of RAM. 100,000 is fine for simple environments. |

### Things You Probably DON'T Need to Change

| Parameter | Default | Why |
|-----------|---------|-----|
| `tau` | 0.005 | This value works well across most environments. |
| `update_actor_interval` | 2 | The original TD3 paper uses 2, and it works well. |

---

## 5. The Training Loop (The Missing Piece)

Your `td3.py` defines the agent but doesn't include the training loop. Here's a complete one:

```python
import gymnasium as gym
import numpy as np
from td3 import Agent
import os

# ──────────────────────────────────────────────
# 1. SETUP
# ──────────────────────────────────────────────
os.makedirs('tmp/td3', exist_ok=True)

env = gym.make('Pendulum-v1')

agent = Agent(
    alpha=0.001,
    beta=0.001,
    input_dims=env.observation_space.shape,
    tau=0.005,
    env=env,
    gamma=0.99,
    update_actor_interval=2,
    warmup=1000,
    n_actions=env.action_space.shape[0],
    max_size=1000000,
    layer1_size=400,
    layer2_size=300,
    batch_size=100,
    noise=0.1
)

# ──────────────────────────────────────────────
# 2. TRAINING LOOP
# ──────────────────────────────────────────────
n_games = 1000           # Total episodes to train
best_score = -np.inf     # Track the best average score
score_history = []        # Store all episode scores

for i in range(n_games):
    observation, info = env.reset()
    done = False
    truncated = False
    score = 0

    while not (done or truncated):
        # Agent picks an action (with exploration noise)
        action = agent.choose_action(observation)

        # Environment responds
        observation_, reward, done, truncated, info = env.step(action)

        # Store the experience
        agent.remember(observation, action, reward, observation_, done or truncated)

        # Learn from a batch of past experiences
        agent.learn()

        # Update score and move to the next state
        score += reward
        observation = observation_

    score_history.append(score)

    # Calculate rolling average over last 100 episodes
    avg_score = np.mean(score_history[-100:])

    # Save the model if we beat the previous best average
    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()

    # Print progress every 10 episodes
    if (i + 1) % 10 == 0:
        print(f'Episode {i+1}, Score: {score:.1f}, '
              f'Avg Score (last 100): {avg_score:.1f}, '
              f'Best Avg: {best_score:.1f}')

# ──────────────────────────────────────────────
# 3. EVALUATION (after training)
# ──────────────────────────────────────────────
# Load the best model
agent.load_models()

# Run 10 episodes without noise to see how well the agent performs
eval_env = gym.make('Pendulum-v1', render_mode='human')
for i in range(10):
    observation, info = eval_env.reset()
    done = False
    truncated = False
    score = 0
    while not (done or truncated):
        # Use the actor directly (no noise) for evaluation
        state = np.array(observation, dtype=np.float32)
        import torch as T
        state_tensor = T.tensor(state).to(agent.actor.device)
        action = agent.actor.forward(state_tensor).cpu().detach().numpy()
        observation, reward, done, truncated, info = eval_env.step(action)
        score += reward
    print(f'Eval Episode {i+1}: Score = {score:.1f}')

eval_env.close()
```

### Anatomy of the Training Loop — What Happens and Why

Here's the flow for each episode:

```
for each episode:
    1. env.reset()              → Get initial state
    2. while not done:
        a. agent.choose_action()  → Pick action (random during warmup, actor + noise after)
        b. env.step(action)       → Execute action, get reward & next state
        c. agent.remember()       → Store transition in replay buffer
        d. agent.learn()          → Sample batch, update critics (& sometimes actor)
        e. score += reward        → Track performance
        f. observation = next     → Move to next state
    3. Log score, save if best
```

---

## 6. Hyperparameter Guide

### The Most Important Hyperparameters (Ranked)

#### 1. Learning Rates (`alpha`, `beta`) — Impact: HIGH

- **What:** How big a step the optimizer takes when updating weights.
- **Too high (e.g., 0.01):** Training is unstable — loss bounces around, agent can't converge.
- **Too low (e.g., 0.00001):** Training is extremely slow — agent barely learns.
- **Good starting point:** `0.001` for both. If unstable, try `0.0003`.
- **Pro tip:** Some practitioners use a slightly higher learning rate for the critic than the actor (e.g., `beta=0.001`, `alpha=0.0003`).

#### 2. Exploration Noise (`noise`) — Impact: HIGH

- **What:** Standard deviation of Gaussian noise added to actions during training.
- **Too high:** Agent takes wild random actions, can't exploit what it learns.
- **Too low:** Agent doesn't explore enough, gets stuck in local optima.
- **How to scale:** Look at your action range. If actions range from [-1, 1], noise=0.1 means ~10% of the range. If actions range from [-100, 100], you might need noise=10.
- **Good starting point:** ~5-10% of the action range.

#### 3. Batch Size (`batch_size`) — Impact: MEDIUM

- **What:** How many transitions to sample per learning step.
- **Small (32-64):** Noisier gradients, but faster per step. Can be good for simple environments.
- **Large (256-512):** More stable gradients, but each step processes more data.
- **Good starting point:** 100 or 256.

#### 4. Discount Factor (`gamma`) — Impact: MEDIUM

- **What:** How much the agent cares about future rewards vs. immediate rewards.
- **0.99:** Agent plans ~100 steps ahead (1/(1-0.99) = 100).
- **0.95:** Agent plans ~20 steps ahead.
- **0.9:** Agent plans ~10 steps ahead.
- **For most tasks:** 0.99 is fine. Only lower it for very short-horizon tasks (< 50 steps per episode).

#### 5. Network Size (`layer1_size`, `layer2_size`) — Impact: MEDIUM

- **What:** Number of neurons in hidden layers.
- **Too small:** Network can't represent the complexity of the task.
- **Too large:** Slower training, potential overfitting, more memory.
- **Simple environments** (Pendulum, CartPole): 256, 128 is enough.
- **Medium environments** (HalfCheetah, Hopper): 400, 300 (the defaults).
- **Complex environments** (Humanoid, multi-robot): 512, 512 or larger.

#### 6. Warmup Steps (`warmup`) — Impact: LOW-MEDIUM

- **What:** How many timesteps of random actions before using the actor.
- **Why it matters:** Ensures the replay buffer has diverse data before learning starts.
- **Simple environments:** 500-1000.
- **Complex environments:** 5000-25000.
- **Rule of thumb:** At least 2-5x the batch size.

#### 7. Soft Update Rate (`tau`) — Impact: LOW

- **What:** How fast target networks track the main networks.
- **Almost always:** Leave at 0.005. This was validated in the original paper and works across most tasks.
- **If training is very unstable:** Try 0.001 (even slower target updates).

#### 8. Replay Buffer Size (`max_size`) — Impact: LOW

- **What:** How many transitions to store.
- **Almost always:** 1,000,000 is fine. Only reduce if you're running out of RAM.
- **Minimum useful size:** Should be at least 10x the batch size.

---

## 7. Common Pitfalls and How to Avoid Them

### Pitfall 1: Forgetting to Create the Checkpoint Directory

**Symptom:** `FileNotFoundError` when saving models.
**Fix:** `os.makedirs('tmp/td3', exist_ok=True)` before creating the agent.

### Pitfall 2: Wrong `n_actions` or `input_dims`

**Symptom:** Shape mismatch errors from PyTorch (e.g., "mat1 and mat2 shapes cannot be multiplied").
**Fix:** Always derive these from the environment:
```python
input_dims = env.observation_space.shape    # NOT a number, must be a tuple like (8,)
n_actions = env.action_space.shape[0]       # A single integer
```

### Pitfall 3: Using TD3 with Discrete Action Spaces

**Symptom:** Errors or nonsensical behavior.
**Fix:** TD3 is **only** for continuous action spaces. Check with:
```python
from gymnasium.spaces import Box
assert isinstance(env.action_space, Box), "TD3 requires continuous (Box) action spaces!"
```

### Pitfall 4: Not Enough Warmup

**Symptom:** Training starts but agent never improves, or performance is erratic early on.
**Fix:** Increase `warmup`. The agent needs a diverse buffer before it can learn useful patterns.

### Pitfall 5: Noise Scale Doesn't Match Action Range

**Symptom:** Agent either never explores (noise too small for the action range) or takes wild actions (noise too large).
**Fix:** Scale noise proportionally to the action range:
```python
action_range = env.action_space.high[0] - env.action_space.low[0]
noise = 0.1 * action_range  # 10% of the range
```

### Pitfall 6: Not Handling `truncated` in Modern Gymnasium

**Symptom:** Episodes that should end early (time limit) don't, or the agent learns incorrectly.
**Context:** Older Gym returned `(obs, reward, done, info)`. Modern Gymnasium returns `(obs, reward, terminated, truncated, info)`.
**Fix:** In the training loop:
```python
observation_, reward, terminated, truncated, info = env.step(action)
# terminated = episode ended naturally (goal reached, fell over, etc.)
# truncated = episode hit the time limit
agent.remember(observation, action, reward, observation_, terminated)
# Note: only pass `terminated` (not truncated) as `done` to the buffer.
# Reason: if truncated, the episode was cut short — future rewards DO exist.
# Setting done=True for truncated states incorrectly tells the agent there's no future.
done = terminated or truncated  # For loop control
```

### Pitfall 7: Training for Too Few Episodes

**Symptom:** Agent doesn't seem to learn.
**Reality:** TD3 is not a fast algorithm. It needs many episodes.
**Expectations:**
- Pendulum: ~200-500 episodes to see improvement
- LunarLanderContinuous: ~500-2000 episodes
- HalfCheetah/Hopper: ~1000-5000 episodes
- Humanoid: ~10,000+ episodes

### Pitfall 8: Evaluating with Exploration Noise

**Symptom:** The trained agent performs worse than expected.
**Fix:** During evaluation, don't add noise. Use the actor directly:
```python
action = agent.actor.forward(state_tensor).cpu().detach().numpy()
```

---

## 8. Order of Operations — What Comes First, What Comes Last

Here is the complete checklist, in order:

### Phase 1: Project Setup
1. Install dependencies (`torch`, `numpy`, `gymnasium`, environment-specific packages)
2. Create the checkpoint directory (`os.makedirs('tmp/td3', exist_ok=True)`)
3. Copy `td3.py` into your project

### Phase 2: Environment Understanding
4. Create the environment (`env = gym.make(...)`)
5. Print and understand the observation space shape
6. Print and understand the action space shape and bounds
7. Verify the action space is continuous (`Box`), not discrete

### Phase 3: Agent Configuration
8. Decide on hyperparameters (start with defaults, tune later)
9. Create the agent with the correct `input_dims`, `n_actions`, and `env`
10. Verify the agent was created without errors (this catches shape mismatches early)

### Phase 4: Training
11. Write the training loop (see Section 5)
12. Run training, monitoring the average score over the last 100 episodes
13. Save the best model (based on rolling average, not single episodes)
14. If performance plateaus, tune hyperparameters (see Section 6)

### Phase 5: Evaluation
15. Load the best saved model
16. Run episodes without exploration noise
17. Optionally render the environment to visually inspect behavior

### Phase 6: Iteration (if needed)
18. If performance is poor, check the debugging checklist (Section 9)
19. Adjust hyperparameters and retrain
20. Consider if TD3 is the right algorithm for your problem

---

## 9. Debugging Checklist

When TD3 isn't working, go through this list in order:

### Level 1: Basic Sanity Checks
- [ ] Did you create the checkpoint directory?
- [ ] Are `input_dims` and `n_actions` correct for your environment?
- [ ] Is the action space continuous (Box)?
- [ ] Are you training for enough episodes?

### Level 2: Learning Checks
- [ ] Is the replay buffer filling up? (Check `agent.memory.mem_cntr`)
- [ ] Is `learn()` actually being called? (It returns early if buffer has fewer samples than `batch_size`)
- [ ] Are the critic losses decreasing over time? (Add logging)
- [ ] Is the actor loss changing? (It only updates every `update_actor_iter` steps)

### Level 3: Hyperparameter Checks
- [ ] Is `noise` proportional to the action range?
- [ ] Is `warmup` sufficient? (Try 10x the batch size)
- [ ] Are learning rates reasonable? (Try 0.0003 if 0.001 is unstable)
- [ ] Is `gamma` appropriate for your episode length?

### Level 4: Environment Checks
- [ ] Does the environment give meaningful rewards? (Not always zero)
- [ ] Are rewards on a reasonable scale? (Very large or small rewards can cause issues)
- [ ] Is the environment deterministic enough to learn from? (Extremely stochastic environments are harder)

### How to Add Logging for Debugging

```python
# Inside the training loop, after agent.learn():
if agent.memory.mem_cntr > agent.batch_size and i % 100 == 0:
    print(f"  Buffer size: {agent.memory.mem_cntr}")
    print(f"  Learn steps: {agent.learn_step_cntr}")
    print(f"  Time step: {agent.time_step}")
```

---

## 10. Extending TD3 — Advanced Modifications

Once you're comfortable with the basics, here are common extensions:

### Reward Normalization

If your environment gives very large or very small rewards, normalize them:
```python
# In the training loop:
reward = reward / 100.0  # Scale down large rewards
# OR use a running mean/std normalization
```

### Learning Rate Scheduling

Decrease the learning rate over time for finer convergence:
```python
from torch.optim.lr_scheduler import StepLR
# After creating the agent:
actor_scheduler = StepLR(agent.actor.optimizer, step_size=500, gamma=0.95)
critic1_scheduler = StepLR(agent.critic_1.optimizer, step_size=500, gamma=0.95)
critic2_scheduler = StepLR(agent.critic_2.optimizer, step_size=500, gamma=0.95)
# At the end of each episode:
actor_scheduler.step()
critic1_scheduler.step()
critic2_scheduler.step()
```

### Prioritized Experience Replay

Instead of sampling uniformly from the buffer, sample transitions that had large prediction errors more frequently. This is a significant modification — the replay buffer needs to be replaced with a priority queue structure.

### Multi-Dimensional Action Noise

The current implementation generates a single noise value for all action dimensions. For environments with different scales per action dimension, generate independent noise per dimension:
```python
# In choose_action, replace:
mu_prime = mu + T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)),
                         dtype=T.float).to(self.actor.device)
```

### Image-Based Observations (Convolutional Networks)

If your environment returns images instead of vectors:
1. Replace the `nn.Linear` first layers with `nn.Conv2d` layers
2. Add a flatten step before the fully connected layers
3. Adjust `input_dims` to be the image shape (e.g., `(3, 84, 84)`)
4. This is a significant change — consider using an established implementation

### Using TD3 for Custom (Non-Gym) Environments

If you're building your own environment (robotics, game, simulation), you need to provide:
1. A `reset()` method that returns the initial state
2. A `step(action)` method that returns `(next_state, reward, done, truncated, info)`
3. An `action_space` object with `.high`, `.low`, and `.shape` attributes
4. An `observation_space` object with a `.shape` attribute

Minimal example:
```python
import numpy as np
from gymnasium.spaces import Box

class MyCustomEnv:
    def __init__(self):
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(4,))
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,))

    def reset(self):
        self.state = np.zeros(4)  # Your initial state logic
        return self.state, {}

    def step(self, action):
        # Your environment dynamics here
        next_state = self.state + action[0]  # Simplified example
        reward = -np.sum(self.state**2)       # Reward function
        done = False                           # Termination condition
        truncated = False
        info = {}
        self.state = next_state
        return next_state, reward, done, truncated, info
```

---

## Quick Reference Card

```
┌──────────────────────────────────────────────────────────┐
│                   TD3 QUICK REFERENCE                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  SETUP:                                                  │
│    1. pip install torch numpy gymnasium                  │
│    2. os.makedirs('tmp/td3', exist_ok=True)              │
│    3. env = gym.make('YourEnv-v1')                       │
│    4. agent = Agent(alpha=0.001, beta=0.001,             │
│         input_dims=env.observation_space.shape,          │
│         tau=0.005, env=env,                              │
│         n_actions=env.action_space.shape[0])             │
│                                                          │
│  TRAINING LOOP:                                          │
│    obs = env.reset()                                     │
│    action = agent.choose_action(obs)                     │
│    obs_, reward, done, trunc, info = env.step(action)    │
│    agent.remember(obs, action, reward, obs_, done)       │
│    agent.learn()                                         │
│                                                          │
│  KEY HYPERPARAMETERS:                                    │
│    alpha, beta = 0.001    (learning rates)               │
│    noise = 0.1            (exploration)                  │
│    gamma = 0.99           (discount factor)              │
│    tau = 0.005            (target update rate)           │
│    warmup = 1000          (random exploration steps)     │
│    batch_size = 100       (learning batch size)          │
│                                                          │
│  ENVIRONMENT REQUIREMENTS:                               │
│    - Continuous action space (gymnasium.spaces.Box)      │
│    - action_space.high, action_space.low, .shape         │
│    - observation_space.shape                             │
│                                                          │
└──────────────────────────────────────────────────────────┘
```
