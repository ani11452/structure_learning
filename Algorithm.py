import numpy as np
import pandas as pd
import math
import scipy
import networkx as nx


class Rosetta(object):
    def __init__(self, start_temp, end_temp, anneal, max_parents, filename):
        self.data = pd.read_csv(filename)
        self.variables = list(self.data.columns)
        self.num_vars = len(self.variables)

        self.var_to_instantiations = {}
        self.calculate_var_instantiations()
        self.parent_counter = {}

        self.graph = nx.DiGraph()
        self.initialize_graph()

        self.bayesian_score = 0
        self.r, self.q, self.M, self.a = [], [], [], []
        self.bayesian_score_components = {}
        self.initialize_bayesian_score()

        self.parent = None
        self.child = None
        self.temp = start_temp
        self.end_temp = end_temp
        self.anneal = anneal
        self.max_parents = max_parents

    def initialize_graph(self):
        for i in range(self.num_vars):
            self.graph.add_node(i)
            self.parent_counter[i] = 0

    def calculate_var_instantiations(self):
        for i, variable in enumerate(self.variables):
            self.var_to_instantiations[i] = max(self.data[variable])

    def sub2ind(self, dimensions, x):
        k = np.concatenate(([1], np.cumprod(dimensions[0:-1])))
        return np.dot(k, x - 1)

    def bayesian_score_component(self, M, a):
        p = np.sum(scipy.special.loggamma(np.add(a, M)))
        p -= np.sum(scipy.special.loggamma(a))
        p += np.sum(scipy.special.loggamma(np.sum(a, axis=1)))
        p -= np.sum(scipy.special.loggamma(np.sum(a, axis=1) + np.sum(M, axis=1)))

        return p

    def initialize_bayesian_score(self):
        n = self.num_vars
        self.r = np.array([self.var_to_instantiations[i] for i in range(n)])
        self.q = [int(np.prod([self.r[j] for j in self.graph.predecessors(i)])) for i in range(n)]
        self.M = [np.zeros((self.q[i], self.r[i])) for i in range(n)]
        self.a = [np.ones((self.q[i], self.r[i])) for i in range(n)]

        for i in range(n):
            for index, row in self.data.iterrows():
                row = np.array(row)
                k = row[i] - 1
                parents = list((self.graph.predecessors(i)))

                if parents:
                    j = self.sub2ind(self.r[parents], row[np.array(parents)])
                else:
                    j = 0

                self.M[i][j, k] += 1

            component = self.bayesian_score_component(self.M[i], self.a[i])
            self.bayesian_score_components[i] = component
            self.bayesian_score += component

    def updated_bayesian_score(self, i):
        q = int(np.prod([self.r[j] for j in self.graph.predecessors(i)]))
        M = np.zeros((q, self.r[i]))
        a = np.ones((q, self.r[i]))

        for index, row in self.data.iterrows():
            row = np.array(row)
            k = row[i] - 1
            parents = list(self.graph.predecessors(i))

            if parents:
                j = self.sub2ind(self.r[parents], row[np.array(parents)])
            else:
                j = 0

            M[j, k] += 1

        component = self.bayesian_score_component(M, a)

        return component, q, M, a

    def perturb_graph(self, action):
        if action == "remove":
            self.graph.remove_edge(self.parent, self.child)
        else:
            self.graph.add_edge(self.parent, self.child)

        return nx.simple_cycles(self.graph)

    def metropolis_accept(self, afterE):
        beforeE = self.bayesian_score

        changeE = afterE - beforeE

        if changeE >= 0:
            val = 1
        else:
            val = math.exp(changeE / self.temp)

        return val

    def anneal_temp(self):
        self.temp *= self.anneal

    def undoStep(self, action):
        if action == "remove":
            self.graph.add_edge(self.parent, self.child)
            self.parent_counter[self.child] += 1
        else:
            self.graph.remove_edge(self.parent, self.child)
            self.parent_counter[self.child] -= 1

    def step(self):
        # Check to see if there is child
        if not self.child:
            return

        # Check to see action based on if edge exists or not
        if self.graph.has_edge(self.parent, self.child):
            action = "remove"
            self.parent_counter[self.child] -= 1
        else:
            action = "add"
            if self.parent_counter[self.child] > self.max_parents:
                return
            else:
                self.parent_counter[self.child] += 1

        # Perturb Graph
        cycles = list(self.perturb_graph(action))

        # Check Cycles
        if action == "add" and cycles:
            self.undoStep(action)
            return

        # Find new Bayesian Score Contribution and New Overall Score
        old_component = self.bayesian_score_components[self.child]
        new_component, q, M, a = self.updated_bayesian_score(self.child)
        new_score = self.bayesian_score - old_component + new_component

        # Find metropolis probability
        prob_after = self.metropolis_accept(new_score)

        # Update based on probability
        if np.random.uniform(0, 1) <= prob_after:
            self.bayesian_score = new_score
            self.q[self.child] = q
            self.M[self.child] = M
            self.a[self.child] = a
            self.bayesian_score_components[self.child] = new_component
            self.anneal_temp()
        else:
            self.undoStep(action)

    def simulate(self):
        # Create Nodes List
        nodes = []
        for i in range(self.num_vars):
            nodes.append(i)

        # Randomize Nodes
        np.random.shuffle(nodes)

        # Initialize
        highest_score_graph = self.graph
        highest_score = self.bayesian_score

        # Iterate
        while self.temp >= self.end_temp:

            # Find parent node
            self.parent = np.random.choice(nodes)

            # Find child node
            nodes.remove(self.parent)
            self.child = np.random.choice(nodes)
            nodes.append(self.parent)

            # Conduct Step
            self.step()

            # Update lowest score and lowest score graph
            if (self.bayesian_score > highest_score):
                highest_score_graph = self.graph
                highest_score = self.bayesian_score

            print(highest_score, self.temp)

        return highest_score_graph, self.variables, highest_score



