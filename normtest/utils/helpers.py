### self made ###
from paramcheckup import types, numbers


# with test, with database, with docstring
class AlphaManagement:
    """Instanciates a class for ``alpha`` managment. It is primarily for internal use."""

    def __init__(self, alpha=0.05, **kwargs):
        super().__init__(**kwargs)
        """Constructs the significance level value

        Parameters
        ----------
        alpha : ``float``
            The significance level (default is ``0.05``);

        Notes
        -----
        This method only allows input of type ``float`` and between ``0.0`` and ``1.0``.

        """
        self.func_name = "AlphaManagement"

        if alpha != 0.05:
            types.is_float(value=alpha, param_name="alpha", func_name=self.func_name)
            numbers.is_between_a_and_b(
                value=alpha,
                a=0.005,
                b=0.995,
                param_name="alpha",
                func_name=self.func_name,
                inclusive=True,
            )
        self.alpha = alpha

    def get_alpha(self):
        """Returns the current ``alpha`` value"""
        return self.alpha

    def set_alpha(self, alpha):
        """Changes the ``alpha`` value

        Parameters
        ----------
        alpha : ``float``
            The new significance level

        Notes
        -----
        This method only allows input of type ``float`` and between ``0.0`` and ``1.0``.

        """
        types.is_float(value=alpha, param_name="alpha", func_name=self.func_name)
        numbers.is_between_a_and_b(
            value=alpha,
            a=0.005,
            b=0.995,
            param_name="alpha",
            func_name=self.func_name,
            inclusive=True,
        )
        self.alpha = alpha

    def __repr__(self):
        return self.alpha

    def __str__(self):
        return f"The current significance level is '{self.alpha}'"
