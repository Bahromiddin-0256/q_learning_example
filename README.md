# Q-Learning Visual Example with FastAPI

An interactive web application that demonstrates Q-Learning reinforcement learning algorithm through a visual grid world environment. Watch as an AI agent learns to navigate from a starting point to a goal while avoiding obstacles!

## 🎯 Features

- **Interactive Grid World**: Visual representation of the environment where the agent learns
- **Real-time Training Visualization**: Watch the agent learn in real-time
- **Performance Charts**: Track rewards and steps per episode
- **Policy Visualization**: See the learned policy with arrows showing optimal actions
- **Configurable Parameters**: Adjust learning rate, discount factor, epsilon, and more
- **FastAPI Backend**: Modern, fast REST API for training and state management
- **Responsive Web UI**: Clean, intuitive interface built with vanilla JavaScript

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Bahromiddin-0256/q_learning_example.git
cd q_learning_example
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

3. Use the web interface to:
   - Initialize the environment
   - Configure learning parameters
   - Train the agent
   - Visualize the learning process

## 📖 How It Works

### Q-Learning Algorithm

Q-Learning is a model-free reinforcement learning algorithm that learns the value of actions in different states. The agent updates its Q-table using the formula:

```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

Where:
- `Q(s,a)`: Q-value for state s and action a
- `α`: Learning rate
- `r`: Reward
- `γ`: Discount factor
- `s'`: Next state
- `a'`: Next action

### Environment

The grid world consists of:
- **Start Position** (🏁): Where the agent begins (top-left)
- **Goal Position** (🎯): Target destination (bottom-right)
- **Obstacles** (🚫): Cells the agent should avoid
- **Empty Cells**: Free spaces for navigation

### Rewards

- Reaching the goal: +10
- Hitting an obstacle: -1
- Hitting a wall: -0.1
- Normal step: -0.04 (encourages shorter paths)

## 🎮 API Endpoints

### `POST /api/initialize`
Initialize the environment and agent with custom parameters.

**Request Body:**
```json
{
  "grid_size": 5,
  "learning_rate": 0.1,
  "discount_factor": 0.95,
  "epsilon": 1.0,
  "epsilon_decay": 0.995
}
```

### `POST /api/train`
Train the agent for a specified number of episodes.

**Request Body:**
```json
{
  "episodes": 100,
  "max_steps": 100,
  "learning_rate": 0.1,
  "discount_factor": 0.95,
  "epsilon": 1.0,
  "epsilon_decay": 0.995,
  "grid_size": 5
}
```

### `GET /api/state`
Get current state of the environment and agent, including Q-table, policy, and performance metrics.

### `POST /api/reset`
Reset the environment to initial state.

### `GET /api/health`
Health check endpoint.

## 🔧 Configuration Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| Grid Size | Dimensions of the grid world | 5 | 3-10 |
| Episodes | Number of training episodes | 100 | 1-1000 |
| Learning Rate (α) | How much new information overrides old | 0.1 | 0.01-1.0 |
| Discount Factor (γ) | Importance of future rewards | 0.95 | 0-1.0 |
| Epsilon (ε) | Initial exploration rate | 1.0 | 0-1.0 |
| Epsilon Decay | Rate at which exploration decreases | 0.995 | 0.9-1.0 |

## 📁 Project Structure

```
q_learning_example/
├── main.py              # FastAPI application and endpoints
├── agent.py             # Q-Learning agent implementation
├── environment.py       # Grid world environment
├── requirements.txt     # Python dependencies
├── static/
│   └── index.html      # Web interface
└── README.md           # This file
```

## 🧪 Testing

You can test the API using curl:

```bash
# Health check
curl http://localhost:8000/api/health

# Initialize environment
curl -X POST http://localhost:8000/api/initialize \
  -H "Content-Type: application/json" \
  -d '{"grid_size": 5, "learning_rate": 0.1, "discount_factor": 0.95, "epsilon": 1.0, "epsilon_decay": 0.995}'

# Train agent
curl -X POST http://localhost:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{"episodes": 50, "max_steps": 100, "grid_size": 5}'

# Get current state
curl http://localhost:8000/api/state
```

## 🎓 Learning Resources

- **Epsilon-Greedy Strategy**: Balances exploration (random actions) vs exploitation (best known actions)
- **Q-Table**: Stores the expected reward for each state-action pair
- **Policy**: The optimal action to take in each state (shown as arrows in the UI)
- **State Values**: Maximum Q-value for each state (shown as numbers in cells)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Charts powered by [Chart.js](https://www.chartjs.org/)
- Inspired by classic reinforcement learning examples