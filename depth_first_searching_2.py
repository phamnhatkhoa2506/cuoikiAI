graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': ['H', 'I'],
    'E': ['J', 'K'],
    'F': ['L', 'M'],
    'G': ['N', 'O'],
    'H': [],
    'I': [],
    'J': [],
    'K': [],
    'L': [],
    'M': [],
    'N': [],
    'O': []
}

def DFS(
    initial_state,
    goal_test
):
    frontier = [initial_state]
    explored = []

    while frontier:
        state = frontier.pop(-1)
        explored.append(state)

        if goal_test == state: return explored

        for neighbor in graph[state]:
            if neighbor not in (frontier and  explored):
                frontier.append(neighbor)

    return False


if __name__ == '__main__':
    result = DFS('A', 'H')
    if result:
        s = f'Explored: {result}'
        print(s)
    else:
        raise Exception("Not Found")
    