import networkx as nx


def calculate_pagerank(similarity_matrix):
    """Xây dựng đồ thị và tính điểm PageRank."""

    graph = nx.from_numpy_array(
        similarity_matrix
    )

    # Loại bỏ self-loop
    graph.remove_edges_from(
        nx.selfloop_edges(graph)
    )

    scores = nx.pagerank(
        graph,
        alpha=0.85,
        weight="weight"
    )

    return scores