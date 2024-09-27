import sys
import networkx as nx
import matplotlib.pyplot as plt
from Algorithm import Rosetta
import time


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def compute(infile, outfile):
    calculator = Rosetta(100, 1, 0.999, 5, infile)
    start_time = time.time()
    result = calculator.simulate()
    print(time.time() - start_time)
    nx.draw(result[0])
    plt.show()
    write_gph(result[0], result[1], outfile)



def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
