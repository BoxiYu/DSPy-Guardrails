"""Base interfaces for adversarial attack mutation strategies."""


class MutationStrategy:
    """Base class for mutation strategies."""

    def mutate(self, payload: str) -> str:
        raise NotImplementedError
