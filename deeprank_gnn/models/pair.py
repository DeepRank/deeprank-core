from typing import Any


class Pair:
    """ A hashable, comparable object for any set of two inputs where order doesn't matter.
        Args:
            item1 (object): the pair's first object, must be convertable to string
            item2 (object): the pair's second object, must be convertable to string
    """

    def __init__(self, item1: Any, item2: Any):
        self.item1 = item1
        self.item2 = item2

    def __hash__(self):
        # The hash should be solely based on the two paired items, not on their order.

        s1 = str(self.item1)
        s2 = str(self.item2)

        if s1 < s2:
            return hash(s1 + s2)
        else:
            return hash(s2 + s1)

    def __eq__(self, other):
        # Compare the pairs as sets, so the order doesn't matter.

        return self.item1 == other.item1 and self.item2 == other.item2 or \
               self.item1 == other.item2 and self.item2 == other.item1

    def __iter__(self):
        # Iterate over the two items in the pair.
        return iter([self.item1, self.item2])

    def __repr__(self):
        return (str(self.item1), str(self.item2))
