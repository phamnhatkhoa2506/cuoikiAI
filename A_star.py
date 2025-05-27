import heapq as hq

class Node:
    def __init__(self, 
                 name: str, 
                 goal_cost: int) -> None:
        self.__name = name
        self.__goal_cost = goal_cost
        self.__children = []
        self.__cost = 0

    @property
    def name(self):
        return self.__name
    
    @property
    def goal_cost(self):
        return self.__goal_cost
    
    @property
    def cost(self):
        return self.__cost
    
    @cost.setter
    def cost(self, value):
        self.__cost = value
    
    @property
    def children(self):
        return self.__children
    
    def add_children(self, children):
        self.__children.extend(children)

    def __repr__(self) -> str:
        return str({
            "label": self.__name,
            "cost": self.__cost,
            "goal_cost": self.__goal_cost,
        })
    
    def __str__(self) -> str:
        return f'{self.__name}'
    
    def __eq__(self, value: object) -> bool:
        return self.__name == value.name
    
    def __lt__(self, value: object) -> bool:
        return self.__goal_cost + self.__cost  < value.goal_cost + value.cost
    

class Edge:
    def __init__(self, 
                 start: Node, 
                 end: Node, 
                 cost: int) -> None:
        self.__start = start
        self.__end = end
        self.__cost = cost

    @property
    def start_point(self) -> Node:
        return self.__start
    
    @property
    def end_point(self) -> Node:
        return self.__end
    
    @property
    def cost(self) -> int:
        return self.__cost
    
    def __repr__(self) -> str:
        return str(self.__start) + str(self.__end) + f'({self.__cost})'


class Tree:
    def __init__(self) -> None:
        self.__nodes: list[Node] = []
        self.__edges: list[Edge] = []

    @property
    def nodes(self):
        return self.__nodes
    
    def add_nodes(self, *nodes):
        self.__nodes.extend(nodes)

    def add_egdes(self, *edges):
        self.__edges.extend(edges)
        for node in self.__nodes:
            node.add_children([edge.end_point for edge in self.__edges if node == edge.start_point])
            # print([str(n.name) for n in node.children])

    def get_edge(self, start_node: Node, end_node: Node):
        print(str(start_node), str(end_node))
        for edge in self.__edges:
            if start_node == edge.start_point and \
               end_node   == edge.end_point:
                return edge

        return None


def update_cost(
    tree: Tree, 
    parent_node: Node, 
    child_node: Node,
):
    edge: Edge = tree.get_edge(parent_node, child_node)
    # print(edge)
    if edge is not None:
        # if child_node.cost > parent_node.cost + edge.cost:
        child_node.cost = parent_node.cost + edge.cost


def A_star(
    tree: Tree,
    initial_state: Node,
    goal: Node
):
    frontier = [initial_state]
    explored = []

    while frontier:
        state = hq.heappop(frontier)
        explored.append(state)
        print(repr(state))

        if goal == state: return explored

        for neighbor in state.children:
            # print(str(neighbor))
            update_cost(tree, state, neighbor)
            if neighbor.name not in list(set(node.name for node in frontier + explored)):
                hq.heappush(frontier, neighbor)

    return False


if __name__ == '__main__':
    tree = Tree()
    A = Node("A", 6)
    B = Node("B", 3)
    C = Node("C", 4)
    D = Node("D", 5)
    E = Node("E", 3)
    F = Node("F", 1)
    G = Node("G", 6)
    H = Node("H", 2)
    I = Node("I", 5)
    J = Node("J", 4)
    K = Node("K", 2)
    L = Node("L", 0)
    M = Node("M", 4)
    N = Node("N", 0)
    O = Node("O", 4)
    tree.add_nodes(
        A, B, C, D, E, F, G, H, I, J, K, L, M , N, O
    )
    print('Nodes: ', [str(node) for node in tree.nodes])
    tree.add_egdes(
        Edge(A, B, 2),
        Edge(A, C, 1),
        Edge(A, D, 3),
        Edge(B, E, 5),
        Edge(B, F, 4),
        Edge(C, G, 6),
        Edge(C, H, 3),
        Edge(D, I, 2),
        Edge(D, J, 4),
        Edge(F, K, 2),
        Edge(F, L, 1),
        Edge(F, M, 4),
        Edge(H, N, 2),
        Edge(H, O, 4),
    )
    print("Start point: ", tree.nodes[0])
    print("End point: ", tree.nodes[14])

    result = A_star(tree, tree.nodes[0], tree.nodes[14])    
    print(
        f"Eplored: {[str(n) for n in result]}" if result else "NOT FOUND"
    )


# import heapq as hq

# class Node:
#     def __init__(self, label, goal_cost) -> None:
#         self.label = label
#         self.goal_cost = goal_cost
#         self.save_cost = None
#         self.cost = 10000
#         self.pr = []
#         self.chld = []

#     def __repr__(self) -> str:
#         return str({
#             "label": self.label,
#             "cost": self.cost,
#             "goal_cost": self.goal_cost, 
#         })
    
#     def __eq__(self, value: object) -> bool:
#         return self.label == value.label
    
#     def __lt__(self, other):
#         # if self.save_cost == 10000:
#         #     return self.goal_cost + self.cost < other.goal_cost + other.cost
#         # else:
#         return self.cost + self.goal_cost < other.cost + other.goal_cost
        
#     def get_label(self):
#         return self.label
    
#     def neighbors(self):
#         return self.chld + self.pr
    

# class Tree:
#     def __init__(self) -> None:
#         self.nodes = []
#         self.edges = []

#     def add_nodes(self, data):
#         self.nodes.extend(data)

#     def add_node(self, node):
#         self.nodes.append(node)

#     def get_index(self, node):
#         for i, n in enumerate(self.nodes):
#             if n.get_label() == node.get_label():
#                 return i
#         return -1
    
#     def add_edges(self, tuple_edges):
#         for t in tuple_edges:
#             start_label, end_label, w = t

#             index_start_label = self.get_index(Node(start_label, None))
#             index_end_label = self.get_index(Node(end_label, None))

#             self.nodes[index_start_label].chld.append(self.nodes[index_end_label])
#             self.nodes[index_end_label].pr.append(self.nodes[index_start_label])
#             self.edges.append((
#                 self.nodes[index_start_label], 
#                 self.nodes[index_end_label],
#                 w
#             ))

#     def show_nodes(self):
#         return [node.get_label() for node in self.nodes]
    
#     def get_edge(self, start_node, end_node):
#         try:
#             return [edge for edge in self.edges \
#                         if edge[0] == start_node and
#                            edge[1] == end_node][0]
#         except:
#             return None


# def update_cost(tree, current_node, prev_node):
#     edge = tree.get_edge(prev_node, current_node)
#     if edge is not None:
#         if current_node.cost > prev_node.cost + edge[2]:
#             current_node.cost = prev_node.cost + edge[2]


# def A_star(tree, start, end):
#     frontier = [start]
#     hq.heapify(frontier)
#     explored = []
#     while len(frontier) > 0:
#         state = hq.heappop(frontier)
#         explored.append(state)
#         print(state)
#         if state == end:
#             return explored
#         for child in state.neighbors():
#             update_cost(tree, child, state)
#             if child.get_label() not in list(set(node.get_label() for node in frontier + explored)):
#                 hq.heappush(frontier, child)

#     return False


# if __name__ == '__main__':
#     tree = Tree()
#     tree.add_nodes([
#         Node("A", 6),
#         Node("B", 3),
#         Node("C", 4),
#         Node("D", 5),
#         Node("E", 3),
#         Node("F", 1),
#         Node("G", 6),
#         Node("H", 2),
#         Node("I", 5),
#         Node("J", 4),
#         Node("K", 2),
#         Node("L", 0),
#         Node("M", 4),
#         Node("N", 0),
#         Node("O", 4),
#     ])
#     tree.add_edges([
#         ("A", "B", 2),
#         ("A", "C", 1),
#         ("A", "D", 3),
#         ("B", "E", 5),
#         ("B", "F", 4),
#         ("C", "G", 6),
#         ("C", "H", 3),
#         ("D", "I", 2),
#         ("D", "J", 4),
#         ("F", "K", 2),
#         ("F", "L", 1),
#         ("F", "M", 4),
#         ("H", "N", 2),
#         ("H", "O", 4),
#     ])

#     tree.nodes[0].cost = 0
#     result = A_star(tree, tree.nodes[0], tree.nodes[14])
#     if result:
#         s = "Explored: "
#         for i in result:
#             s += i.label + " "
#             print(s)
#     else:
#         print("404 NOT FOUND")