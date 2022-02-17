"""
Utility functions for the `silver_sampler_old` package.
"""


def is_documented_by(original):
    def wrapper(target):
        target.__doc__ = original.__doc__
        return target
    return wrapper