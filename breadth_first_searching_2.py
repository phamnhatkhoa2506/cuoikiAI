from treelib import Tree, Node
from collections import deque

tree = Tree()
tree.create_node('A', 'A')
tree.create_node('B', 'B', 'A')
tree.create_node('C', 'C', 'A')
tree.create_node('D', 'D', 'B')
tree.create_node('E', 'E', 'B')
tree.create_node('F', 'F', 'C')
tree.create_node('G', 'G', 'C')
tree.create_node('H', 'H', 'D')
tree.create_node('I', 'I', 'D')
tree.create_node('J', 'J', 'E')
tree.create_node('K', 'K', 'E')
tree.create_node('L', 'L', 'F')
tree.create_node('M', 'M', 'F')
tree.create_node('N', 'N', 'G')
tree.create_node('O', 'O', 'G')

def BFS(
    initial_state: Node,
    goal: str
) -> bool:
    frontier = deque()
    frontier.appendleft(initial_state)
    explored = []

    while len(frontier):
        state = frontier.pop()
        explored.append(state.tag)

        print(explored)
        if goal == state.tag: return explored

        for neighbor in tree.children(state.identifier):
            if neighbor not in (explored and frontier):
                frontier.appendleft(neighbor)

    return False

if __name__ == '__main__':
    result: bool | list = BFS(tree.get_node('A'), 'O')

    if result:
        print(f"Explored: {result}")
    else:
        raise Exception('NOT FOUND')