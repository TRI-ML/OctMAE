import torch as th
from ocnn.octree import Octree


def get_xyz_from_octree(octree, depth, nempty=False):
    scale = 2 ** (1 - depth)
    x, y, z, _ = octree.xyzb(depth=depth, nempty=nempty)
    xyz = (th.stack([x, y, z], dim=1) + 0.5) * scale - 1.0
    return xyz


def search_value(key: th.Tensor, query: th.Tensor, value: th.Tensor = None):
    r''' Searches values according to sorted shuffled keys.

    Args:
      value (th.Tensor): The input tensor with shape (N, C).
      key (th.Tensor): The key tensor corresponds to :attr:`value` with shape
          (N,), which contains sorted shuffled keys of an octree.
      query (th.Tensor): The query tensor, which also contains shuffled keys.
    '''

    # deal with out-of-bound queries, the indices of these queries
    # returned by th.searchsorted equal to `key.shape[0]`
    out_of_bound = query > key[-1]

    # search
    idx = th.searchsorted(key, query)
    idx[out_of_bound] = -1   # to avoid overflow when executing the following line
    found = key[idx] == query

    if value is not None:
        # assign the found value to the output
        out = th.zeros(query.shape[0], value.shape[1], device=value.device)
        out[found] = value[idx[found]]
        return out, found
    else:
        return found


def octree_align(value: th.Tensor, octree: Octree, octree_query: Octree,
                 depth: int, nempty: bool = False):
    r''' Wraps :func:`octree_align` to take octrees as input for convenience.
    '''

    key = octree.key(depth, nempty)
    query = octree_query.key(depth, nempty)
    assert key.shape[0] == value.shape[0]
    return search_value(key, query, value)


def octree_search(octree: Octree, octree_query: Octree,
                  depth: int, nempty: bool = False):
    r''' Wraps :func:`octree_align` to take octrees as input for convenience.
    '''
    key = octree.key(depth, nempty)
    query = octree_query.key(depth, nempty)
    return search_value(key, query)
