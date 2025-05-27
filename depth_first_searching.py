class Node:
    def __init__(self, name) -> None:
        self.name = name 
        self.children = []

    def add_children(self, children):
        self.children.extend(children)

def DFS(
    initial_state: Node,
    goal: list[Node]
):
    frontier = [initial_state]
    explored = []

    while frontier:
        state = frontier.pop()
        explored.append(state.name)

        if goal == state.name: return explored

        for child in state.children:
            if child not in (explored and frontier):
                frontier.append(child)

    return False

if __name__ == '__main__':
    list_name = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                 'M', 'N', 'O']
    for name in list_name:
        globals()[name] = Node(name)
    for parent, children in zip(list_name[:7],
                                [[globals()[list_name[i]], globals()[list_name[i + 1]]] for i in range(1, len(list_name), 2)]):
        globals()[parent].add_children(children)

    A: Node
    result = DFS(A, 'H')
    if result:
        print(f'Explored: {result}')
    else:
        raise Exception("NOT FOUND")
