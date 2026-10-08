"""One error type for anything a config or a recorded split can get wrong."""

from __future__ import annotations


class ConfigError(Exception):
    """A config file is malformed, out of range, or names something unknown.

    The message always carries the dotted key and the file that set it. A
    config error that says only "invalid value" sends the reader to the wrong
    file when configs inherit.
    """


class SplitError(Exception):
    """A recorded split disagrees with the corpus, or cannot be built."""
