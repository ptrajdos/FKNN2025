import abc
class TNorm(abc.ABC):
    """
    Abstract base class for t-norms.
    """

    @abc.abstractmethod
    def t_norm(self, a, b):
        """
        Compute the t-norm of two values.

        Parameters
        ----------
        a : float
            The first value.
        b : float
            The second value.

        Returns
        -------
        float
            The t-norm of the two values.
        """
