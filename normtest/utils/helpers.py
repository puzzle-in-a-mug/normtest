### self made ###
from paramcheckup import types, numbers

from normtest.utils import documentation as docs


class SafeManagement:
    """Instanciates a class for `safe` managment. It is primarily for internal use."""

    # @docs.docstring_parameter(
    #     safe=docs.SAFE["type"],
    #     safe_desc=docs.SAFE["description"],
    # )
    def __init__(self, safe=True, **kwargs):
        super().__init__(**kwargs)
        """Constructs the parameter `safe`


        Parameters
        ----------
        {safe}
            {safe_desc}


        """
        self.func_name = "SafeManagement"

        if safe is not True:
            types.is_bool(value=safe, param_name="safe", func_name=self.func_name)
        self.safe = safe

    # @docs.docstring_parameter(
    #     safe=docs.SAFE["type"],
    #     safe_desc=docs.SAFE["description"],
    # )
    def get_safe(self):
        """Returns the current status of parameter `safe`


        Returns
        -------
        {safe}
            {safe_desc}


        """
        return self.safe

    # @docs.docstring_parameter(
    #     safe=docs.SAFE["type"],
    #     safe_desc=docs.SAFE["description"],
    # )
    def set_safe(self, safe):
        """Changes the current status of parameter `safe`


        Parameters
        ----------
        {safe}
            {safe_desc}


        """
        if safe is not True:
            types.is_bool(value=safe, param_name="safe", func_name=self.func_name)
        self.safe = safe

    def __repr__(self):
        return self.safe

    def __str__(self):
        return f"The current state of parameter `safe` is '{self.safe}'"


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
