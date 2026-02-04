from collections import defaultdict


class NestedDefault:
    """Pickleable callable that returns a nested defaultdict."""
    def __init__(self, depth: int, leaf_factory: type = dict):
        self.depth = depth
        self.leaf_factory = leaf_factory

    def __call__(self) -> defaultdict:
        if self.depth <= 1:
            return defaultdict(self.leaf_factory)
        # Create next level with depth-1
        return defaultdict(NestedDefault(self.depth - 1, self.leaf_factory))


def nested_defaultdict(
        depth: int = 1,
        factory: type = dict
) -> defaultdict:
    """
    Create a nested defaultdict of specified depth.
    Pickleable, no lambdas or nested functions.
    """
    return NestedDefault(depth, factory)()
