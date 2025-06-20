import matplotlib.pyplot as plt
import networkx as nx
import time

class GraphVisualizer:
    def __init__(self):
        self.undirected_graph = {}  # Adjacency list for undirected graph (BFS/DFS)
        self.directed_graph = nx.DiGraph()  # Directed weighted graph for Dijkstra/Bellman-Ford
        self.setup_example_graphs()

    def setup_example_graphs(self):
        # Setup undirected graph for BFS/DFS
        self.add_undirected_edge("A", "B")
        self.add_undirected_edge("A", "C")
        self.add_undirected_edge("B", "D")
        self.add_undirected_edge("C", "E")
        self.add_undirected_edge("D", "F")
        self.add_undirected_edge("E", "F")

        # Setup directed weighted graph for Dijkstra/Bellman-Ford
        edges = [
            ('A', 'B', 4),
            ('A', 'C', 2),
            ('B', 'C', 5),
            ('B', 'D', 10),
            ('C', 'E', 3),
            ('E', 'D', 4),
            ('D', 'F', 11)
        ]
        self.directed_graph.add_weighted_edges_from(edges)

    def add_undirected_edge(self, u, v):
        self.undirected_graph.setdefault(u, []).append(v)
        self.undirected_graph.setdefault(v, []).append(u)  # Undirected

    def draw_undirected_graph(self, pos, state):
        plt.clf()
        G = nx.Graph()
        for node in self.undirected_graph:
            for neighbor in self.undirected_graph[node]:
                G.add_edge(node, neighbor)

        color_map = {
            "unvisited": "lightgray",
            "visiting": "yellow",
            "visited": "green"
        }

        node_colors = [color_map[state.get(node, "unvisited")] for node in G.nodes]
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=800, font_weight='bold')
        plt.pause(0.8)

    def draw_directed_graph(self, distances, processed, current_node=None, current_edge=None):
        plt.clf()
        pos = nx.spring_layout(self.directed_graph, seed=42)
        labels = {node: f"{node}\n{distances.get(node, '∞')}" for node in self.directed_graph.nodes}
        
        node_colors = []
        for node in self.directed_graph.nodes:
            if node == current_node:
                node_colors.append('orange')
            elif node in processed:
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightblue')

        edge_colors = []
        for edge in self.directed_graph.edges:
            if current_edge and edge == current_edge:
                edge_colors.append('red')
            else:
                edge_colors.append('black')

        edge_labels = nx.get_edge_attributes(self.directed_graph, 'weight')
        nx.draw(self.directed_graph, pos, with_labels=False, node_color=node_colors, node_size=1500)
        nx.draw_networkx_labels(self.directed_graph, pos, labels)
        nx.draw_networkx_edges(self.directed_graph, pos, arrows=True, edge_color=edge_colors)
        nx.draw_networkx_edge_labels(self.directed_graph, pos, edge_labels=edge_labels)
        plt.pause(1)

    def bfs(self, start):
        visited = set()
        queue = [start]
        state = {}
        pos = nx.spring_layout(nx.Graph(self.undirected_graph))

        while queue:
            current = queue.pop(0)

            if current in visited:
                continue

            state[current] = "visiting"
            self.draw_undirected_graph(pos, state)

            visited.add(current)
            state[current] = "visited"
            self.draw_undirected_graph(pos, state)

            for neighbor in self.undirected_graph[current]:
                if neighbor not in visited and neighbor not in queue:
                    queue.append(neighbor)

    def dfs(self, start):
        visited = set()
        stack = [start]
        state = {}
        pos = nx.spring_layout(nx.Graph(self.undirected_graph))

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            state[current] = "visiting"
            self.draw_undirected_graph(pos, state)

            visited.add(current)
            state[current] = "visited"
            self.draw_undirected_graph(pos, state)

            for neighbor in reversed(self.undirected_graph[current]):
                if neighbor not in visited:
                    stack.append(neighbor)

    def dijkstra(self, start):
        distances = {node: float('inf') for node in self.directed_graph.nodes}
        distances[start] = 0
        processed = set()

        while len(processed) < len(self.directed_graph.nodes):
            current = None
            current_dist = float('inf')
            for node in self.directed_graph.nodes:
                if node not in processed and distances[node] < current_dist:
                    current = node
                    current_dist = distances[node]

            if current is None:
                break

            for neighbor in self.directed_graph.successors(current):
                weight = self.directed_graph[current][neighbor]['weight']
                if distances[neighbor] > distances[current] + weight:
                    distances[neighbor] = distances[current] + weight

            processed.add(current)
            self.draw_directed_graph(distances, processed)

    def bellman_ford(self, start):
        distances = {node: float('inf') for node in self.directed_graph.nodes}
        distances[start] = 0

        # Relax all edges |V| - 1 times
        for i in range(len(self.directed_graph.nodes) - 1):
            updated = False
            for u, v, data in self.directed_graph.edges(data=True):
                weight = data['weight']
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    updated = True
                    self.draw_directed_graph(distances, set(), current_node=v, current_edge=(u, v))
                else:
                    self.draw_directed_graph(distances, set(), current_node=v, current_edge=(u, v))
            if not updated:
                break

        # Check for negative-weight cycles
        has_negative_cycle = False
        for u, v, data in self.directed_graph.edges(data=True):
            weight = data['weight']
            if distances[u] + weight < distances[v]:
                has_negative_cycle = True
                print(f"\nNegative cycle detected between {u} and {v}!")
                break

        if has_negative_cycle:
            print("Graph contains a negative weight cycle. Distances may not be accurate.")
        else:
            print("Final shortest path distances:")
            for node, dist in distances.items():
                print(f"{node}: {dist}")

        return not has_negative_cycle

    def display_menu(self):
        print("\nGraph Algorithm Visualizer")
        print("1. Breadth-First Search (BFS)")
        print("2. Depth-First Search (DFS)")
        print("3. Dijkstra's Algorithm")
        print("4. Bellman-Ford Algorithm")
        print("5. Exit")

    def run(self):
        plt.ion()
        
        while True:
            self.display_menu()
            try:
                choice = int(input("Enter your choice (1-5): "))
            except ValueError:
                print("Please enter a valid number.")
                continue
            
            if choice == 5:
                print("Exiting program...")
                break
            elif choice in [1, 2, 3, 4]:
                start_node = input("Enter starting node (e.g., A): ").strip().upper()
                
                if choice == 1:
                    print(f"Running BFS from node {start_node}")
                    self.bfs(start_node)
                elif choice == 2:
                    print(f"Running DFS from node {start_node}")
                    self.dfs(start_node)
                elif choice == 3:
                    print(f"Running Dijkstra's algorithm from node {start_node}")
                    self.dijkstra(start_node)
                elif choice == 4:
                    print(f"Running Bellman-Ford algorithm from node {start_node}")
                    self.bellman_ford(start_node)
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")
        
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    visualizer = GraphVisualizer()
    visualizer.run()