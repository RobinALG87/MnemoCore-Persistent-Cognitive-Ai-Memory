class CircuitBreakerError(Exception): pass
class CircuitBreakerListener: pass
class CircuitBreaker:
    def __init__(self, *args, **kwargs):
        self.state = type('State', (), {'name': 'CLOSED'})
        self.listeners = []
    def call_async(self, func, *args, **kwargs):
        return func(*args, **kwargs)
def circuit_breaker(*args, **kwargs):
    return lambda f: f
