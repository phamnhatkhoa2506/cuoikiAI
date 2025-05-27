from collections import deque

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

def breadth_first_search(
    initial_state,
    goal_test
):
    frontier = deque()
    frontier.appendleft(initial_state)
    explored = set()

    while len(frontier):
        state = frontier.pop()
        explored.add(state)

        if goal_test == state: return explored

        for neighbor in graph[state]:
            if neighbor not in (frontier and  explored):
                frontier.appendleft(neighbor)

        print(frontier)

    return False


if __name__ == '__main__':
    result = breadth_first_search('A', 'O')
    if result:
        s = f'Explored: {result}'
        print(s)
    else:
        raise Exception("Not Found")
    