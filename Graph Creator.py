import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

file = open("large_best.gph")
small_graph = []
for line in file:
    small_graph.append(tuple(line.strip().split(',')))
g = nx.DiGraph(small_graph)
nx.draw_networkx(g)
plt.show()

print(1)
