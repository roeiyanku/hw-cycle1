"""
Tests for has_cycle1: does a weighted directed graph contain a cycle
whose product of edge weights is smaller than 1?

Run with:  python cycle1_test.py
"""

import math
import random
import time
import unittest

import networkx as nx

from cycle1 import WeightedDiGraph, has_cycle1


def brute_force(graph: nx.DiGraph) -> bool:
    """
    Reference implementation: enumerate all simple cycles and check products.
    (If any cycle has product < 1, some simple cycle within it does too,
    so checking simple cycles is enough.)  Only usable on small graphs.
    """
    for cycle in nx.simple_cycles(graph):
        log_sum = sum(
            math.log(graph.edges[u, v]["weight"])
            for u, v in zip(cycle, cycle[1:] + cycle[:1])
        )
        if log_sum < 0:
            return True
    return False


def random_graph(n: int, m: int, low: float, high: float, seed=None) -> nx.DiGraph:
    """Random directed graph with n nodes, ~m edges, weights uniform in [low, high]."""
    rng = random.Random(seed)
    edges = set()
    while len(edges) < m:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            edges.add((u, v))
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from((u, v, {"weight": rng.uniform(low, high)}) for u, v in edges)
    return G


class TestSmallGraphs(unittest.TestCase):
    def test_empty_graph(self):
        self.assertFalse(has_cycle1(WeightedDiGraph()))

    def test_single_edge_no_cycle(self):
        self.assertFalse(has_cycle1(WeightedDiGraph([0, 1, 0.5])))

    def test_path_no_cycle(self):
        # small weights but no cycle at all
        self.assertFalse(has_cycle1(WeightedDiGraph([0, 1, 0.1], [1, 2, 0.1], [2, 3, 0.1])))

    def test_triangle_product_above_one(self):
        self.assertFalse(has_cycle1(WeightedDiGraph([0, 1, 55], [1, 2, 66], [2, 0, 77])))

    def test_triangle_product_below_one(self):
        self.assertTrue(has_cycle1(WeightedDiGraph([0, 1, 0.55], [1, 2, 0.66], [2, 0, 0.77])))

    def test_two_cycle(self):
        self.assertTrue(has_cycle1(WeightedDiGraph([0, 1, 0.5], [1, 0, 1.5])))   # 0.75 < 1
        self.assertFalse(has_cycle1(WeightedDiGraph([0, 1, 0.5], [1, 0, 3.0])))  # 1.5 > 1

    def test_product_exactly_one(self):
        # 2 * 0.5 == 1 exactly: NOT smaller than 1
        self.assertFalse(has_cycle1(WeightedDiGraph([0, 1, 2.0], [1, 0, 0.5])))

    def test_self_loop(self):
        self.assertTrue(has_cycle1(WeightedDiGraph([0, 0, 0.9])))
        self.assertFalse(has_cycle1(WeightedDiGraph([0, 0, 1.1])))

    def test_mixed_weights_cycle_above_one(self):
        # cycle contains weights < 1 but the product is still > 1
        self.assertFalse(has_cycle1(WeightedDiGraph([0, 1, 0.5], [1, 2, 0.5], [2, 0, 5.0])))

    def test_good_cycle_hidden_among_big_weights(self):
        # one bad cycle far from a big-weight cycle
        G = WeightedDiGraph(
            [0, 1, 10], [1, 2, 10], [2, 0, 10],   # product 1000
            [3, 4, 0.5], [4, 3, 0.5],             # product 0.25  <- the culprit
            [2, 3, 100],                          # bridge (not on any cycle back)
        )
        self.assertTrue(has_cycle1(G))

    def test_disconnected_components(self):
        G = WeightedDiGraph([0, 1, 2], [1, 0, 2], [10, 11, 0.5], [11, 10, 0.5])
        self.assertTrue(has_cycle1(G))
        G2 = WeightedDiGraph([0, 1, 2], [1, 0, 2], [10, 11, 3], [11, 10, 3])
        self.assertFalse(has_cycle1(G2))

    def test_non_integer_node_labels(self):
        G = WeightedDiGraph(["a", "b", 0.5], ["b", "a", 0.5])
        self.assertTrue(has_cycle1(G))


class TestRandomGraphsVsBruteForce(unittest.TestCase):
    """Compare against simple-cycle enumeration on many small random graphs."""

    def test_random_small_graphs(self):
        rng = random.Random(42)
        for trial in range(200):
            n = rng.randint(2, 7)
            m = rng.randint(1, n * (n - 1))
            G = random_graph(n, m, low=0.3, high=3.0, seed=rng.random())
            self.assertEqual(
                has_cycle1(G), brute_force(G),
                msg=f"trial {trial}: disagreement on {list(G.edges(data=True))}",
            )

    def test_random_small_graphs_weights_near_one(self):
        # weights close to 1 make the decision numerically delicate
        rng = random.Random(7)
        for trial in range(100):
            n = rng.randint(2, 6)
            m = rng.randint(1, n * (n - 1))
            G = random_graph(n, m, low=0.9, high=1.1, seed=rng.random())
            self.assertEqual(has_cycle1(G), brute_force(G), msg=f"trial {trial}")


class TestLargeGraphs(unittest.TestCase):
    def test_large_all_weights_above_one(self):
        # no product can be < 1
        G = random_graph(1000, 100_000, low=1.0, high=5.0, seed=1)
        self.assertFalse(has_cycle1(G))

    def test_large_all_weights_below_one(self):
        # dense graph full of cycles, all weights < 1
        G = random_graph(1000, 100_000, low=0.1, high=0.9, seed=2)
        self.assertTrue(has_cycle1(G))

    def test_large_with_planted_cheap_cycle(self):
        # all weights >= 1 except one tiny planted 3-cycle
        G = random_graph(1000, 100_000, low=1.0, high=5.0, seed=3)
        G.add_edge(101, 202, weight=0.01)
        G.add_edge(202, 303, weight=0.01)
        G.add_edge(303, 101, weight=0.01)
        self.assertTrue(has_cycle1(G))

    def test_large_no_cycle_at_all(self):
        # DAG: edges only from lower to higher index, tiny weights everywhere
        rng = random.Random(4)
        G = nx.DiGraph()
        G.add_nodes_from(range(1000))
        edges = set()
        while len(edges) < 100_000:
            u, v = rng.randrange(1000), rng.randrange(1000)
            if u < v:
                edges.add((u, v))
        G.add_edges_from((u, v, {"weight": rng.uniform(0.01, 0.5)}) for u, v in edges)
        self.assertFalse(has_cycle1(G))

    def test_speed_1000_vertices_100000_edges(self):
        # hardest case for Bellman-Ford: cycles everywhere, none below 1
        G = random_graph(1000, 100_000, low=1.0, high=1.001, seed=5)
        start = time.perf_counter()
        result = has_cycle1(G)
        elapsed = time.perf_counter() - start
        self.assertFalse(result)
        self.assertLess(elapsed, 1.0, f"took {elapsed:.3f}s, required < 1s")


if __name__ == "__main__":
    unittest.main()
