from functools import wraps
import os
import time
from collections import defaultdict

times = defaultdict(list)
def timed(log_level="development", log_start=False, log_end=True):
    """Decorator that logs the execution of the decorated function

    Usage:

    @timed()
    def foo(bar):
        ....

    @timed(log_type="development")
    def foo(bar):
        ....

    """

    def log_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            if os.environ['profiling'].lower() == "true":
                print(f"Finished running: {func.__name__}" f" in {(end - start)*1000}ms")
                #times[func.__name__].append((end - start))
            return result

        return wrapper

    return log_wrapper