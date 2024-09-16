# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

import time
from datetime import datetime


class Timer:
    """Timing the running time of the program.

    Examples
    --------
    If you want to start a default timer and then calculate the program interval,
    you can do like:

    >>> timer = Timer()
    >>> do_something()
    >>> h, m, s = timer.stop()
    >>> h, m, s
    1, 2, 13.435

    And you can update the start time of the default timer, just do like:

    >>> timer.start()

    You can also register a specified timer and update the start time of it.

    >>> timer.start('new')
    >>> h, m, s = timer.stop('new')
    """

    def __init__(self, name: str = "dpeeg_timer") -> None:
        """Timing the running time of the program.
        Initialization will automatically start a default timer (dpeeg_timer).
        """
        self._beg = {}
        self.start(name)
        self.start("dpeeg_timer")

    def start(self, name: str = "dpeeg_timer") -> str:
        """Update specified timer or register a timer if it does not exist."""
        self._beg[name] = time.time()
        return self.ctime()

    def stop(
        self, name: str = "dpeeg_timer", restart: bool = False
    ) -> tuple[int, int, float]:
        """Get the hours, minutes, and seconds of a specific timer."""
        sec = self.stop_precise(name, restart)
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)

        return int(h), int(m), s

    def stop_precise(self, name: str = "dpeeg_timer", restart: bool = False) -> float:
        """Get the precise time for a specific timer."""
        if name not in self._beg.keys():
            raise RuntimeError(f"Timer `{name}` has not been registered yet.")
        ct = time.time()

        if restart:
            self.start(name)

        return ct - self._beg[name]

    @staticmethod
    def ctime() -> str:
        """Return the current local time as a human-readable string."""
        return time.ctime()

    @staticmethod
    def cdate() -> str:
        """Get the current date and time in the 'year-month-day_hour:minute'
        format.
        """
        return datetime.now().strftime("%y-%m-%d_%H:%M")
