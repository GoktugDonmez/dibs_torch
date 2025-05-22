import random

import igraph as ig
import jax
import jax.numpy as jnp


def scalefree_dag_gmat(key, d, n_edges_per_node=2):
    random.seed(int(jnp.sum(key)))
    perm = jax.random.permutation(key, d).tolist()
    g = ig.Graph.Barabasi(n=d, m=n_edges_per_node, directed=True).permute_vertices(perm)
    dag_mat = graph_to_mat(g)
    return dag_mat


def graph_to_mat(g):
    # ig.Graph to jnp.array representing adjacency matrix
    return jnp.array(g.get_adjacency().data)


def mat_to_graph(mat):
    # jnp.array representing adjacency matrix to ig.Graph object
    assert len(mat.shape) == 2, "igraph can only handle d x d matrix inputs"
    assert mat.shape[0] == mat.shape[1], "igraph can only handle d x d matrix inputs"
    return ig.Graph.Weighted_Adjacency(mat.tolist())


@jax.jit
def topological_sort_jit(adj_matrix):
    # Assumes no cycles - goes to infinite loop if there's a cycle
    # As a result, we process any learnt graphs with process_dag() to make
    # it an empty graph if it has any cycles
    n = adj_matrix.shape[0]

    @jax.jit
    def find_candidate(in_degree, visited):
        candidates = (in_degree == 0) & (~visited)
        return jnp.argmax(candidates), jnp.any(candidates)

    @jax.jit
    def loop_body(state):
        visited, in_degree, order, i = state
        node, valid = find_candidate(in_degree, visited)
        new_visited = jax.lax.cond(
            valid, lambda: visited.at[node].set(True), lambda: visited
        )
        new_in_degree = jax.lax.cond(
            valid, lambda: in_degree - adj_matrix[node], lambda: in_degree
        )
        new_order = jax.lax.cond(valid, lambda: order.at[i].set(node), lambda: order)
        return new_visited, new_in_degree, new_order, i + jnp.int32(valid)

    # Initialize state
    visited = jnp.zeros(n, dtype=bool)
    in_degree = jnp.sum(adj_matrix, axis=0)
    order = jnp.full(n, -1, dtype=jnp.int32)
    i = 0

    # Main loop
    def cond_fun(state):
        visited, _, _, i = state
        return (jnp.sum(visited) < n) & (i < n)

    final_state = jax.lax.while_loop(
        cond_fun, loop_body, (visited, in_degree, order, i)
    )

    return final_state[2]  # Return the order


def adjacency_to_igraph_directed(adj_matrix):
    g = ig.Graph(directed=True)
    num_vertices = len(adj_matrix)
    g.add_vertices(num_vertices)  # Add vertices

    # Add edges and weights
    edges = []
    weights = []
    for i in range(num_vertices):
        for j in range(num_vertices):  # Consider all entries for directed edges
            if adj_matrix[i, j] != 0:
                edges.append((i, j))
                weights.append(jnp.round(adj_matrix[i, j], 3))

    g.add_edges(edges)
    g.es["weight"] = weights

    return g
