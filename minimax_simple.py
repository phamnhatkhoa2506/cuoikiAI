import pygame as pg
from typing import List, Tuple


class Move(object):
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __copy__(self):
        return Move(self.x, self.y)

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return self.x == other.x and self.y == other.y


class Maze(object):
    def __init__(self, maze: List[List[int]]):
        self.maze = maze
        self.__directions = [
            Move(0, 1),   # right
            Move(1, 0),   # down
            Move(0, -1),  # left
            Move(-1, 0),  # up
        ]

    def get_possible_moves(self, position: Move) -> List[Move]:
        return [
            Move(position.x + move.x, position.y + move.y) 
            for move in self.__directions
            if self.__is_valid_move(Move(position.x + move.x, position.y + move.y))
        ]

    def __is_valid_move(self, move: Move) -> bool:
        if move.x < 0 or move.x >= len(self.maze):
            return False
        if move.y < 0 or move.y >= len(self.maze[0]):
            return False
        if self.maze[move.x][move.y] == 1:
            return False
        return True


class Minimax:
    @staticmethod
    def minimax(maze: Maze, move: Move, goal: Move, depth: int, is_maximizing: bool, 
                visited: set = None) -> Tuple[float, Move]:
        if visited is None:
            visited = set()
        
        # Base cases
        if move == goal:
            return 0, move
        if depth == 0:
            return Minimax.heuristic(move, goal), move
        
        # Clone visited set for each path
        visited = visited.copy()
        visited.add((move.x, move.y))
        
        next_moves = maze.get_possible_moves(move)
        if not next_moves:
            return float('inf') if is_maximizing else float('-inf'), move
        
        # Sort moves by heuristic value for better exploration
        next_moves.sort(key=lambda m: Minimax.heuristic(m, goal))
        
        best_move = next_moves[0]
        
        if is_maximizing:
            best_score = float('-inf')
            for next_move in next_moves:
                # Skip if already visited
                if (next_move.x, next_move.y) in visited:
                    continue
                    
                score, _ = Minimax.minimax(maze, next_move, goal, depth - 1, False, visited)
                if score > best_score:
                    best_score = score
                    best_move = next_move
            return best_score, best_move
        else:
            best_score = float('inf')
            for next_move in next_moves:
                # Skip if already visited
                if (next_move.x, next_move.y) in visited:
                    continue
                    
                score, _ = Minimax.minimax(maze, next_move, goal, depth - 1, True, visited)
                if score < best_score:
                    best_score = score
                    best_move = next_move
            return best_score, best_move

    @staticmethod
    def heuristic(move: Move, goal: Move) -> float:
        # Manhattan distance
        manhattan = abs(move.x - goal.x) + abs(move.y - goal.y)
        
        # Euclidean distance for more precise distance estimation
        euclidean = ((move.x - goal.x) ** 2 + (move.y - goal.y) ** 2) ** 0.5
        
        # Combine both distances with weights
        distance_score = 0.7 * manhattan + 0.3 * euclidean
        
        # Add penalty for moves that don't get closer to the goal
        if manhattan > 0:
            return -distance_score  # Negative because we want to minimize distance
            
        return 0


class Game:
    def __init__(self, maze: Maze, start: Move, goal: Move):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.current_pos = start
        self.cell_size = 30
        self.margin = 1
        self.width = len(maze.maze[0]) * (self.cell_size + self.margin)
        self.height = len(maze.maze) * (self.cell_size + self.margin)
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)  # For visited cells
        
        # Initialize pygame
        pg.init()
        self.screen = pg.display.set_mode((self.width, self.height))
        pg.display.set_caption("Maze Solver with Simple Minimax")
        self.clock = pg.time.Clock()
        
        # Add move delay
        self.move_delay = 300  # milliseconds between moves
        self.last_move_time = 0
        
        # Track visited cells
        self.visited = set()
        
        # Initialize path finding
        self.path = self.find_path()
        if not self.path:
            print("Warning: No valid path found!")
            # Try to find a path with increased depth
            self.path = self.find_path(depth=8)
            if not self.path:
                print("Error: Still no valid path found!")

    def find_path(self, depth=5):
        """Find the complete path from start to goal"""
        path = []
        current = self.start
        visited = set()
        
        while current != self.goal:
            visited.add((current.x, current.y))
            next_moves = self.maze.get_possible_moves(current)
            
            # Filter out visited moves
            next_moves = [move for move in next_moves if (move.x, move.y) not in visited]
            
            if not next_moves:
                print("No valid moves available!")
                return path
                
            # Get the best move using minimax
            _, best_move = Minimax.minimax(self.maze, current, self.goal, depth, True, visited=visited)
            
            if best_move:
                path.append(best_move)
                current = best_move
            else:
                print("No best move found!")
                break
                
        return path

    def draw_maze(self):
        self.screen.fill(self.WHITE)
        
        # Draw maze
        for i in range(len(self.maze.maze)):
            for j in range(len(self.maze.maze[0])):
                x = j * (self.cell_size + self.margin)
                y = i * (self.cell_size + self.margin)
                
                if self.maze.maze[i][j] == 1:  # Wall
                    pg.draw.rect(self.screen, self.BLACK, 
                               (x, y, self.cell_size, self.cell_size))
                else:  # Path
                    if (i, j) in self.visited:
                        pg.draw.rect(self.screen, self.YELLOW, 
                                   (x, y, self.cell_size, self.cell_size))
                    else:
                        pg.draw.rect(self.screen, self.WHITE, 
                                   (x, y, self.cell_size, self.cell_size))
                    pg.draw.rect(self.screen, self.BLACK, 
                               (x, y, self.cell_size, self.cell_size), 1)
        
        # Draw start position
        start_x = self.start.y * (self.cell_size + self.margin)
        start_y = self.start.x * (self.cell_size + self.margin)
        pg.draw.rect(self.screen, self.GREEN, 
                    (start_x, start_y, self.cell_size, self.cell_size))
        
        # Draw goal position
        goal_x = self.goal.y * (self.cell_size + self.margin)
        goal_y = self.goal.x * (self.cell_size + self.margin)
        pg.draw.rect(self.screen, self.RED, 
                    (goal_x, goal_y, self.cell_size, self.cell_size))
        
        # Draw current position
        curr_x = self.current_pos.y * (self.cell_size + self.margin)
        curr_y = self.current_pos.x * (self.cell_size + self.margin)
        pg.draw.rect(self.screen, self.BLUE, 
                    (curr_x, curr_y, self.cell_size, self.cell_size))

    def run(self):
        running = True
        path_index = 0
        
        while running:
            current_time = pg.time.get_ticks()
            
            # Check for quit event
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
            
            # Make move if enough time has passed
            if current_time - self.last_move_time >= self.move_delay:
                if path_index < len(self.path):
                    self.visited.add((self.current_pos.x, self.current_pos.y))
                    self.current_pos = self.path[path_index]
                    path_index += 1
                    
                    if self.current_pos == self.goal:
                        print("Goal reached!")
                        running = False
                else:
                    print("Path completed!")
                    running = False
                self.last_move_time = current_time
            
            # Update display
            self.draw_maze()
            pg.display.flip()
            self.clock.tick(60)
        
        pg.quit()


if __name__ == "__main__":
    # Maze layout (0 = path, 1 = wall)
    maze_data = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    ]
    
    maze = Maze(maze_data)
    start = Move(0, 0)
    goal = Move(9, 9)
    
    game = Game(maze, start, goal)
    game.run() 