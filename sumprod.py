import numpy as np

def inference_sum_product(unary_potentials, pairwise_potentials, edges):
    """Sum-product inference on trees.

    Parameters
    ----------
    unary_potentials : nd-array
        Unary potentials of energy function.

    pairwise_potentials : nd-array
        Pairwise potentials of energy function.

    edges : nd-array
        Edges of energy function.
    """
    n_vertices, n_states = unary_potentials.shape
    n_edges = edges.shape[0]
    parents = -np.ones(n_vertices, dtype=np.int)
    visited = np.zeros(n_vertices, dtype=np.bool)
    neighbors = [[] for i in range(n_vertices)]
    pairwise_weights = [[] for i in range(n_vertices)]
    for pw, edge in zip(pairwise_potentials, edges):
        neighbors[edge[0]].append(edge[1])
        pairwise_weights[edge[0]].append(pw) # pw for edge[0] when sending msg along edge
        neighbors[edge[1]].append(edge[0])
        pairwise_weights[edge[1]].append(pw.T)

    pw_forward = np.zeros((n_vertices, n_states, n_states))

    # build a breadth first search of the tree
    traversal = [] # message ordering
    lonely = 0
    while lonely < n_vertices:
        for i in range(lonely, n_vertices):
            if not visited[i]:
                queue = [i]
                lonely = i + 1
                visited[i] = True
                break
            lonely = n_vertices

        while queue:
            node = queue.pop(0)
            traversal.append(node)
            for pw, neighbor in zip(pairwise_weights[node], neighbors[node]):
                if not visited[neighbor]:
                    parents[neighbor] = node
                    queue.append(neighbor)
                    visited[neighbor] = True
                    pw_forward[neighbor] = pw

                elif not parents[node] == neighbor:
                    raise ValueError("Graph not a tree")

    messages_bottomup = np.zeros((n_vertices, n_states))
    messages_topdown = np.zeros((n_vertices, n_states))
    msgs_node_parent = {}

    # messages from leaves to root
    for node in traversal[::-1]:
        parent = parents[node]

        # if node isn't root, pass msg to its parent
        if parent != -1:
            # \log M_node->parent = \log-sum-exp {\log unary + \log pair + \log messages_kids}
            msgs_node_parent[(node, parent)] = logsumexp(messages_bottomup[node]
                    + unary_potentials[node]
                    + pw_forward[node], axis=1)
            msgs_node_parent[(node, parent)] -= logsumexp(msgs_node_parent[(node, parent)])

            # \sum_ch \log M_from_ch
            messages_bottomup[parent] += msgs_node_parent[(node, parent)]

    # messages from root back to leaves
    for node in traversal:
        parent = parents[node]

        if parent != -1:
            # \log M_parent->node = \log-sum-exp{ \log unary + \log pair + msg_other_childrens + msg_upwards}
            message = unary_potentials[parent] + pw_forward[node].T + messages_topdown[parent] \
                      + messages_bottomup[parent] - msgs_node_parent[(node, parent)]
            message = logsumexp(message, axis=1)
            message -= logsumexp(message)

            # \sum_parent \log M_parent->node
            messages_topdown[node] += message

    ret = np.exp(messages_topdown + messages_bottomup + unary_potentials)
    return ret / ret.sum(axis=1)[:, np.newaxis]


def logsumexp(a, axis=None, b=None):
    """Compute the log of the sum of exponentials of input elements.
    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis over which the sum is taken. By default `axis` is None,
        and all elements are summed.
        .. versionadded:: 0.11.0
    b : array-like, optional
        Scaling factor for exp(`a`) must be of the same shape as `a` or
        broadcastable to `a`.
        .. versionadded:: 0.12.0
    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned.
    """
    a = np.asarray(a)
    if axis is None:
        a = a.ravel()
    else:
        a = np.rollaxis(a, axis)
    a_max = a.max(axis=0)
    if b is not None:
        b = np.asarray(b)
        if axis is None:
            b = b.ravel()
        else:
            b = np.rollaxis(b, axis)
        out = np.log(np.sum(b * np.exp(a - a_max), axis=0))
    else:
        out = np.log(np.sum(np.exp(a - a_max), axis=0))
    out += a_max
    return out


if __name__ == '__main__':
    # taken from Sebastian Nowozin's grante-1.0/src/TreeInference_test.cpp:Simple
    edges = [[0, 1]]
    unary = [[-0.1, -0.7], [-0.3, -0.6]]
    pair = [[[0, -0.2], [-0.3, 0]]]
    edges = np.array(edges)
    unary = np.array(unary)
    pair = np.array(pair)
    ret = inference_sum_product(unary, pair, edges)
    print(ret)
