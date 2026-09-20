"""CUDA-event phase timing and repetition statistics (Warp events).

GPU work is asynchronous; wall-clock around a launch measures submission,
not execution.  :class:`PhaseTimer` records CUDA events on the device's
stream at phase boundaries and converts them to milliseconds only after an
explicit synchronization, so reported numbers are true GPU execution
times.
"""

import numpy as np
import warp as wp


class NullTimer:
    """Timer that does nothing (used when phase timing is not requested)."""

    def begin(self):
        pass

    def mark(self, name):
        pass

    def collect(self):
        return {}


class PhaseTimer:
    """Phase timer based on CUDA events recorded on one device stream.

    Usage::

        timer.begin()          # records the base event
        ...launch phase 1...
        timer.mark("phase1")   # phase1 = time since previous mark
        ...launch phase 2...
        timer.mark("phase2")
        results = timer.collect()   # synchronizes, returns {name: ms}

    Repeated marks with the same name accumulate.  Events are pooled and
    reused, so no allocations occur after the first repetition.
    """

    def __init__(self, device="cuda:0"):
        self._device = wp.get_device(device)
        self._pool = []
        self._used = 0
        self._marks = []  # (name, event)

    def _record(self):
        if self._used == len(self._pool):
            self._pool.append(wp.Event(self._device, enable_timing=True))
        ev = self._pool[self._used]
        self._used += 1
        wp.record_event(ev)
        return ev

    def begin(self):
        self._used = 0
        self._marks = [("__begin__", self._record())]

    def mark(self, name):
        self._marks.append((name, self._record()))

    def collect(self):
        """Synchronize and return ``{phase: ms}``."""
        if len(self._marks) < 2:
            return {}
        wp.synchronize_event(self._marks[-1][1])
        out = {}
        for (_, prev), (name, ev) in zip(self._marks, self._marks[1:]):
            ms = wp.get_event_elapsed_time(prev, ev, synchronize=False)
            out[name] = out.get(name, 0.0) + float(ms)
        return out


class IntervalTimer:
    """Single-interval CUDA-event timer for repetition loops.

    ``start()`` and ``stop()`` bracket enqueued GPU work; ``stop()``
    synchronizes and returns milliseconds.
    """

    def __init__(self, device="cuda:0"):
        dev = wp.get_device(device)
        self._start = wp.Event(dev, enable_timing=True)
        self._stop = wp.Event(dev, enable_timing=True)

    def start(self):
        wp.record_event(self._start)

    def stop(self):
        wp.record_event(self._stop)
        wp.synchronize_event(self._stop)
        return float(wp.get_event_elapsed_time(self._start, self._stop,
                                               synchronize=False))


def summarize(samples):
    """Robust statistics of repeated measurements (milliseconds).

    Returns median, IQR, p10, p90, mean, min, max, and the sample count;
    raw samples are preserved by the caller.
    """
    a = np.asarray(samples, dtype=np.float64)
    if a.size == 0:
        return {"count": 0}
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    p10, p90 = np.percentile(a, [10, 90])
    return {
        "median": float(med),
        "iqr": float(q3 - q1),
        "p10": float(p10),
        "p90": float(p90),
        "mean": float(a.mean()),
        "min": float(a.min()),
        "max": float(a.max()),
        "count": int(a.size),
    }
