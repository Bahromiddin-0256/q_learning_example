"""
FastAPI application for Q-Learning visualization
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import os

from environment import GridWorld
from agent import QLearningAgent

app = FastAPI(title="Q-Learning Visual Example")

# Global state
env = None
agent = None
training_history = []


class TrainingConfig(BaseModel):
    """Configuration for training."""
    episodes: int = 100
    max_steps: int = 100
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    grid_size: int = 5


class StepRequest(BaseModel):
    """Request for single step."""
    action: str


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r") as f:
            return f.read()
    return """
    <html>
        <head><title>Q-Learning Example</title></head>
        <body>
            <h1>Q-Learning Visual Example</h1>
            <p>API is running. Please ensure static files are properly configured.</p>
        </body>
    </html>
    """


@app.post("/api/initialize")
async def initialize(config: TrainingConfig):
    """Initialize environment and agent."""
    global env, agent, training_history
    
    env = GridWorld(size=config.grid_size)
    agent = QLearningAgent(
        grid_size=config.grid_size,
        learning_rate=config.learning_rate,
        discount_factor=config.discount_factor,
        epsilon=config.epsilon,
        epsilon_decay=config.epsilon_decay
    )
    training_history = []
    
    return {
        "status": "initialized",
        "environment": env.get_state_representation(),
        "config": config.dict()
    }


@app.post("/api/train")
async def train(config: TrainingConfig):
    """Train the agent for specified number of episodes."""
    global env, agent, training_history
    
    if env is None or agent is None:
        # Initialize if not already done
        env = GridWorld(size=config.grid_size)
        agent = QLearningAgent(
            grid_size=config.grid_size,
            learning_rate=config.learning_rate,
            discount_factor=config.discount_factor,
            epsilon=config.epsilon,
            epsilon_decay=config.epsilon_decay
        )
        training_history = []
    
    episode_data = []
    
    for episode in range(config.episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        episode_steps = []
        
        while not done and steps < config.max_steps:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            
            episode_steps.append({
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done
            })
            
            total_reward += reward
            state = next_state
            steps += 1
        
        agent.decay_epsilon()
        agent.episode_rewards.append(total_reward)
        agent.episode_steps.append(steps)
        
        # Store episode data (only last 10 episodes to avoid too much data)
        if episode >= config.episodes - 10:
            episode_data.append({
                "episode": episode,
                "total_reward": total_reward,
                "steps": steps,
                "epsilon": agent.epsilon,
                "episode_steps": episode_steps
            })
    
    training_history.extend(episode_data)
    
    return {
        "status": "training_complete",
        "episodes_trained": config.episodes,
        "episode_data": episode_data,
        "final_epsilon": agent.epsilon,
        "avg_reward_last_10": sum(agent.episode_rewards[-10:]) / min(10, len(agent.episode_rewards))
    }


@app.get("/api/state")
async def get_state():
    """Get current state of environment and agent."""
    if env is None or agent is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Environment not initialized. Call /api/initialize first."}
        )
    
    return {
        "environment": env.get_state_representation(),
        "current_position": env.current_pos,
        "q_table": agent.get_q_table_for_visualization(),
        "policy": agent.get_policy(),
        "state_values": agent.get_state_values(),
        "epsilon": agent.epsilon,
        "episode_rewards": agent.episode_rewards[-50:] if agent.episode_rewards else [],
        "episode_steps": agent.episode_steps[-50:] if agent.episode_steps else []
    }


@app.get("/api/episode_history")
async def get_episode_history():
    """Get training history."""
    if agent is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Agent not initialized."}
        )
    
    return {
        "episode_rewards": agent.episode_rewards,
        "episode_steps": agent.episode_steps,
        "training_history": training_history
    }


@app.post("/api/reset")
async def reset_environment():
    """Reset environment to initial state."""
    if env is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Environment not initialized."}
        )
    
    state = env.reset()
    return {
        "status": "reset",
        "state": state,
        "environment": env.get_state_representation()
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "initialized": env is not None and agent is not None
    }


# Try to mount static files if directory exists
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
