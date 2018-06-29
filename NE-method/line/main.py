from __future__ import print_function
import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from graph import Graph
import time
import line

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input', required=True,
                        help='Input graph file')
    parser.add_argument('--output', required=True,
                        help='Output representation file')
    parser.add_argument('--number-walks', default=10, type=int,
                        help='Number of random walks to start at each node')
    parser.add_argument('--directed', action='store_true',
                        help='Treat graph as directed.')
    parser.add_argument('--walk-length', default=80, type=int,
                        help='Length of the random walk started at each node')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of parallel processes.')
    parser.add_argument('--representation-size', default=64, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of skipgram model.')
    parser.add_argument('--epochs', default=5, type=int,
                        help='The training epochs of LINE and GCN')
    parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
                        help='Input graph format')
    parser.add_argument('--negative-ratio', default=5, type=int,
                        help='the negative ratio of LINE')
    parser.add_argument('--weighted', action='store_true',
                        help='Treat graph as weighted')
    parser.add_argument('--order', default=3, type=int,
                        help='Choose the order of LINE, 1 means first order, 2 means second order, 3 means first order + second order')
    parser.add_argument('--dropout', default=0.5, type=float, 
                        help='Dropout rate (1 - keep probability)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight for L2 loss on embedding matrix')
    parser.add_argument('--hidden', default=16, type=int,
                        help='Number of units in hidden layer 1')
    parser.add_argument('--kstep', default=4, type=int,
                        help='Use k-step transition probability matrix')

    args = parser.parse_args()
    return args


def main(args):
    t1 = time.time()
    g = Graph()
    print("Reading...")
    if args.graph_format == 'adjlist':
        g.read_adjlist(filename=args.input)
    elif args.graph_format == 'edgelist':
        g.read_edgelist(filename=args.input, weighted=args.weighted, directed=args.directed)
    model = line.LINE(g, epoch = args.epochs, rep_size=args.representation_size, order=args.order)
    model.save_embeddings(args.output)

if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    main(parse_args())
