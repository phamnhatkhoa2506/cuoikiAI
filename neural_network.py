import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
import random


class MazeNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MazeNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MazeEnvironment:
    def __init__(self, maze: List[List[int]], start_pos: Tuple[int, int], goal_pos: Tuple[int, int]):
        self.maze = maze
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.current_pos = start_pos
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        
    def reset(self) -> np.ndarray:
        self.current_pos = self.start_pos
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        next_pos = (
            self.current_pos[0] + self.actions[action][0],
            self.current_pos[1] + self.actions[action][1]
        )
        
        # Check if move is valid
        if not self._is_valid_move(next_pos):
            return self._get_state(), -1.0, False
        
        self.current_pos = next_pos
        
        # Check if goal reached
        if self.current_pos == self.goal_pos:
            return self._get_state(), 1.0, True
        
        # Small negative reward for each step to encourage finding shortest path
        return self._get_state(), -0.1, False
    
    def _is_valid_move(self, pos: Tuple[int, int]) -> bool:
        if pos[0] < 0 or pos[0] >= len(self.maze):
            return False
        if pos[1] < 0 or pos[1] >= len(self.maze[0]):
            return False
        if self.maze[pos[0]][pos[1]] == 1:
            return False
        return True
    
    def _get_state(self) -> np.ndarray:
        # Create state representation: current position, goal position, and surrounding walls
        state = np.zeros(8)  # 8 features: current_x, current_y, goal_x, goal_y, and 4 surrounding walls
        
        # Current position
        state[0] = self.current_pos[0] / len(self.maze)
        state[1] = self.current_pos[1] / len(self.maze[0])
        
        # Goal position
        state[2] = self.goal_pos[0] / len(self.maze)
        state[3] = self.goal_pos[1] / len(self.maze[0])
        
        # Surrounding walls
        for i, action in enumerate(self.actions):
            next_pos = (
                self.current_pos[0] + action[0],
                self.current_pos[1] + action[1]
            )
            state[4 + i] = 1.0 if not self._is_valid_move(next_pos) else 0.0
        
        return state


class MazeSolver:
    def __init__(self, maze: List[List[int]], start_pos: Tuple[int, int], goal_pos: Tuple[int, int],
                 learning_rate: float = 0.001, gamma: float = 0.99):
        self.env = MazeEnvironment(maze, start_pos, goal_pos)
        self.input_size = 8  # State size
        self.hidden_size = 64
        self.output_size = 4  # Number of possible actions
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = MazeNet(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.gamma = gamma
        
    def train(self, episodes: int = 1000, epsilon: float = 0.1):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = random.randint(0, 3)
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        q_values = self.network(state_tensor)
                    action = q_values.argmax().item()
                
                # Take action
                next_state, reward, done = self.env.step(action)
                total_reward += reward
                
                # Update network
                self._update_network(state, action, reward, next_state, done)
                
                state = next_state
            
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}, Total Reward: {total_reward}")
    
    def _update_network(self, state: np.ndarray, action: int, reward: float, 
                       next_state: np.ndarray, done: bool):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        # Get current Q values
        current_q_values = self.network(state_tensor)
        next_q_values = self.network(next_state_tensor)
        
        # Compute target Q value
        target = current_q_values.clone()
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.gamma * next_q_values.max().item()
        
        # Update network
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(current_q_values, target)
        loss.backward()
        self.optimizer.step()
    
    def solve(self) -> List[Tuple[int, int]]:
        state = self.env.reset()
        path = [self.env.start_pos]
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.network(state_tensor)
            action = q_values.argmax().item()
            
            state, _, done = self.env.step(action)
            path.append(self.env.current_pos)
        
        return path


if __name__ == "__main__":
    # Example maze
    maze = [
        [0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    
    start_pos = (0, 0)
    goal_pos = (4, 4)
    
    # Create and train the solver
    solver = MazeSolver(maze, start_pos, goal_pos)
    print("Training the neural network...")
    solver.train(episodes=1000)
    
    # Solve the maze
    print("\nFinding solution path...")
    solution_path = solver.solve()
    print("Solution path:", solution_path)
