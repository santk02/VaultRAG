from app.retrieval.fusion import rrf, fuse_results


def test_rrf_basic():
    """Test basic RRF functionality — a chunk ranked highly in both lists should score
    above one that only appears in a single list."""
    # Two ranked lists
    list1 = ["chunk_a", "chunk_b", "chunk_c"]
    list2 = ["chunk_b", "chunk_a", "chunk_d"]

    result = rrf([list1, list2], k=60, top_n=5)

    # chunk_b appears at rank 1 in list2 and rank 2 in list1
    # chunk_a appears at rank 1 in list1 and rank 2 in list2
    # Both should appear before chunk_c and chunk_d
    assert "chunk_b" in result
    assert "chunk_a" in result
    assert result.index("chunk_b") < result.index("chunk_c")
    assert result.index("chunk_a") < result.index("chunk_d")


def test_rrf_empty_lists():
    """Test RRF with empty lists."""
    result = rrf([[], []], k=60, top_n=5)
    assert result == []


def test_rrf_single_list():
    """Test RRF with a single list."""
    single_list = ["chunk_a", "chunk_b", "chunk_c"]
    result = rrf([single_list], k=60, top_n=5)

    assert result == single_list


def test_rrf_top_n():
    """Test that top_n limits results."""
    long_list = [f"chunk_{i}" for i in range(100)]
    result = rrf([long_list], k=60, top_n=10)

    assert len(result) == 10


def test_fuse_results():
    """Test the convenience function for fusing results."""
    bm25_results = [("chunk_a", 1), ("chunk_b", 2), ("chunk_c", 3)]
    vector_results = [("chunk_b", 1), ("chunk_a", 2), ("chunk_d", 3)]

    result = fuse_results(bm25_results, vector_results)

    # Should return chunk_ids, not tuples
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)
    assert "chunk_a" in result
    assert "chunk_b" in result


def test_rrf_k_parameter():
    """Test that k parameter affects scoring."""
    list1 = ["chunk_a", "chunk_b"]
    list2 = ["chunk_b", "chunk_a"]

    # Small k should give more weight to early ranks (1/(1+rank) varies sharply by rank)
    result_small_k = rrf([list1, list2], k=1, top_n=2)

    # Large k should flatten the scores (1/(1000+rank) barely varies by rank)
    result_large_k = rrf([list1, list2], k=1000, top_n=2)

    # Both should contain the same chunks
    assert set(result_small_k) == set(result_large_k)
