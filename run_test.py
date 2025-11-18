"""Simple test runner to train the Q-learning agent on the GridWorld env.
Run with: python run_test.py
"""
from environment import GridWorld
from agent import QLearningAgent

def run_training(
    grid_size=6,
    obstacle_count=5,
    seed=42,
    randomize_on_reset=False,
    episodes=50,
    max_steps=100,
):
    env = GridWorld(size=grid_size, obstacle_count=obstacle_count, seed=seed, randomize_on_reset=randomize_on_reset)
    agent = QLearningAgent(grid_size=grid_size)

    episode_rewards = []
    episode_steps = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            total_reward += reward
            state = next_state
            steps += 1

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        if (ep + 1) % 10 == 0 or ep == episodes - 1:
            avg_last_10 = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
            print(f"Episode {ep+1}/{episodes} - total_reward={total_reward:.2f}, steps={steps}, epsilon={agent.epsilon:.3f}, avg_last_10={avg_last_10:.3f}")

    print("\nTraining complete")
    print(f"Average reward (last 10): {sum(episode_rewards[-10:]) / min(10, len(episode_rewards)):.3f}")
    print(f"Episodes: {len(episode_rewards)}")
    print(f"Final epsilon: {agent.epsilon:.3f}")

    # print final policy and obstacles for inspection
    print("\nFinal obstacles:", env.obstacles)
    print("Final policy (grid of best actions):")
    policy = agent.get_policy()
    for row in policy:
        print(' '.join(a[0].upper() for a in row))

    return episode_rewards, episode_steps


if __name__ == '__main__':
    run_training()
