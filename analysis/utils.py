import math
import subprocess


class DotDict(dict):
    """
     Extension of a Python dictionary to access its keys using dot notation.

    Parameters
    ----------
    dict : dict
        The dictionary to be extended.

    Example
    -------
    Create a DotDict object and access its keys using dot notation.

    >>> my_dict = {"key1": "value1", "key2": 2, "key3": True}
    >>> my_dot_dict = DotDict(my_dict)
    >>> my_dot_dict.key1
    'value1'
    >>> my_dot_dict.key2
    2
    >>> my_dot_dict.key3
    True
    """

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, item, default=None):
        if item not in self:
            return default
        return self.get(item)


defaults = DotDict(eval_amp=False)


def get_cpuinfo():
    return subprocess.check_output("cat /proc/cpuinfo | grep").split("\n").split(":")[1].strip()


def theoretical_cutoff(d, flops=True):
    if flops:
        return round(d**2 + d + 1)
        # return round(d**2 + 3 * d)
    return round(0.25 * (d**2 + 2 * d + 1 + math.sqrt(d**4 + 12 * d**3 + 14 * d**2 + 4 * d + 1)))
