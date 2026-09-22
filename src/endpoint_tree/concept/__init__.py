"""The sparsity-pattern figure explaining the endpoint ordering.

:mod:`ordering_merged` draws the paper's conceptual figure: how the
within-tail block order decides where the factor acquires fill, and
what the cyclic-reduction elimination costs in exchange for O(log N)
depth.  It is a drawing program, not a benchmark -- no timing is
reported, and the example matrix is only a structural stand-in.
"""
