

class TimeWindow:
    """
    Base class specifying a time window. Has start time and end time attributes.
    Times are expressed using integers, representing the number of minutes since
    the beginning of the day.
    """
    def __init__(self, start: int, end: int):
        if start < 0 or end < 0:
            raise ValueError("Start and end must be >= 0")
        if start > end:
            raise ValueError("Start must be <= end")
        self.start = start
        self.end = end
    def __repr__(self):
        return f"TimeWindow(start={self.start}, end={self.end})"
