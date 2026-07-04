import pytest
from cycle1 import has_cycle1, WeightedDiGraph
from testcases import parse_testcases

testcases = parse_testcases("testcases.txt")

def run_testcase(input:str):
    graph = WeightedDiGraph(*input)
    return has_cycle1(graph)


@pytest.mark.parametrize("testcase", testcases, ids=[testcase["name"] for testcase in testcases])
def test_cases(testcase):
    actual_output = run_testcase(testcase["input"])
    assert actual_output == testcase["output"], f"Expected {testcase['output']}, got {actual_output}"


def test_new_cases():
    # empty graph: no cycle at all
    assert not has_cycle1(WeightedDiGraph())

    # self-loop below / exactly at / above 1
    assert has_cycle1(WeightedDiGraph([0, 0, 0.9]))
    assert not has_cycle1(WeightedDiGraph([0, 0, 1.0]))   # product is 1, NOT smaller than 1
    assert not has_cycle1(WeightedDiGraph([0, 0, 1.1]))

    # 2-cycle whose product is exactly 1 (2 * 0.5): not smaller than 1
    assert not has_cycle1(WeightedDiGraph([0, 1, 2.0], [1, 0, 0.5]))

    # long cycle of weights just below 1: each edge looks harmless, 0.99**10 ~= 0.904 < 1
    assert has_cycle1(WeightedDiGraph(*[[i, (i + 1) % 10, 0.99] for i in range(10)]))

    # same shape just above 1: 1.01**10 ~= 1.105 > 1
    assert not has_cycle1(WeightedDiGraph(*[[i, (i + 1) % 10, 1.01] for i in range(10)]))

    # undirected cycle 0-1-3-2-0 but no *directed* cycle: tiny weights must not fool it
    assert not has_cycle1(WeightedDiGraph([0, 1, 0.1], [0, 2, 0.1], [1, 3, 0.1], [2, 3, 0.1]))

    # two cycles sharing node 1: expensive 0<->1, cheap 1<->2
    assert has_cycle1(WeightedDiGraph([0, 1, 10], [1, 0, 10], [1, 2, 0.5], [2, 1, 0.5]))

    # cheap edges exist but hang off the cycle; the only cycle has product 25 > 1
    assert not has_cycle1(WeightedDiGraph([0, 1, 5], [1, 0, 5], [1, 2, 0.001], [2, 3, 0.001]))

    # cycle mixing sub-1 and above-1 weights: 0.1 * 8 = 0.8 < 1
    assert has_cycle1(WeightedDiGraph([0, 1, 0.1], [1, 0, 8.0]))

    # extreme magnitudes: 1e6 * 1e-7 = 0.1 < 1, but 1e6 * 1e-5 = 10 > 1
    assert has_cycle1(WeightedDiGraph([0, 1, 1e6], [1, 0, 1e-7]))
    assert not has_cycle1(WeightedDiGraph([0, 1, 1e6], [1, 0, 1e-5]))

    # cheap cycle in a separate component, unreachable from the expensive one
    assert has_cycle1(WeightedDiGraph([0, 1, 9], [1, 2, 9], [2, 0, 9], ["x", "y", 0.9], ["y", "x", 0.9]))

    # same shape but both components' cycles have product > 1
    assert not has_cycle1(WeightedDiGraph([0, 1, 9], [1, 2, 9], [2, 0, 9], ["x", "y", 1.5], ["y", "x", 1.5]))

    # ---------- random graphs, checked against brute-force cycle enumeration ----------

    import math
    import random
    import time
    import networkx as nx

    def brute_force(graph: nx.DiGraph) -> bool:
        # enumerate all simple cycles; if any cycle has product < 1,
        # some simple cycle within it does too, so this is a valid reference
        for cycle in nx.simple_cycles(graph):
            log_sum = sum(
                math.log(graph.edges[u, v]["weight"])
                for u, v in zip(cycle, cycle[1:] + cycle[:1])
            )
            if log_sum < 0:
                return True
        return False

    def random_graph(n: int, m: int, low: float, high: float, seed=None) -> nx.DiGraph:
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

    rng = random.Random(42)
    for trial in range(200):
        n = rng.randint(2, 7)
        m = rng.randint(1, n * (n - 1))
        G = random_graph(n, m, low=0.3, high=3.0, seed=rng.random())
        assert has_cycle1(G) == brute_force(G), \
            f"trial {trial}: disagreement on {list(G.edges(data=True))}"

    # weights close to 1 make the decision numerically delicate
    rng = random.Random(7)
    for trial in range(100):
        n = rng.randint(2, 6)
        m = rng.randint(1, n * (n - 1))
        G = random_graph(n, m, low=0.9, high=1.1, seed=rng.random())
        assert has_cycle1(G) == brute_force(G), f"near-one trial {trial}"

    # ---------- large graphs: 1000 vertices, 100000 edges ----------

    # all weights >= 1: no product can be < 1
    assert not has_cycle1(random_graph(1000, 100_000, low=1.0, high=5.0, seed=1))

    # dense graph full of cycles, all weights < 1
    assert has_cycle1(random_graph(1000, 100_000, low=0.1, high=0.9, seed=2))

    # all weights >= 1 except one tiny planted 3-cycle
    G = random_graph(1000, 100_000, low=1.0, high=5.0, seed=3)
    G.add_edge(101, 202, weight=0.01)
    G.add_edge(202, 303, weight=0.01)
    G.add_edge(303, 101, weight=0.01)
    assert has_cycle1(G)

    # large DAG (edges only low index -> high index), tiny weights but no cycle at all
    rng = random.Random(4)
    G = nx.DiGraph()
    G.add_nodes_from(range(1000))
    edges = set()
    while len(edges) < 100_000:
        u, v = rng.randrange(1000), rng.randrange(1000)
        if u < v:
            edges.add((u, v))
    G.add_edges_from((u, v, {"weight": rng.uniform(0.01, 0.5)}) for u, v in edges)
    assert not has_cycle1(G)

    # ---------- speed: must finish within 1 second on 1000 vertices / 100000 edges ----------

    # hardest case for negative-cycle search: cycles everywhere, none with product < 1
    G = random_graph(1000, 100_000, low=1.0, high=1.001, seed=5)
    start = time.perf_counter()
    result = has_cycle1(G)
    elapsed = time.perf_counter() - start
    assert not result
    assert elapsed < 1.0, f"took {elapsed:.3f}s, required < 1s"
