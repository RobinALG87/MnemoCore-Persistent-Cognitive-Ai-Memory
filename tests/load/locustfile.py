"""
Locust load testing file for MnemoCore API.

Usage:
    locust -f tests/load/locustfile.py --host http://localhost:8100

Then open http://localhost:8089 to configure and run the load test.
"""

import random
import string

from locust import HttpUser, between, task


def random_content(length: int = 100) -> str:
    """Generate random content string."""
    return "".join(random.choices(string.ascii_letters + string.digits + " ", k=length))


class StoreMemoryUser(HttpUser):
    """User that stores memories to MnemoCore."""

    wait_time = between(0.1, 0.5)

    def on_start(self):
        """Initialize user with API key."""
        self.api_key = "test-api-key"
        self.headers = {"X-API-Key": self.api_key}

    @task(10)
    def store_memory(self):
        """Store a random memory."""
        payload = {
            "content": random_content(200),
            "metadata": {
                "source": "load_test",
                "timestamp": random.randint(1000000000, 2000000000),
            },
        }
        self.client.post(
            "/memories", json=payload, headers=self.headers, name="/memories [STORE]"
        )


class QueryMemoryUser(HttpUser):
    """User that queries memories from MnemoCore."""

    wait_time = between(0.05, 0.2)

    def on_start(self):
        """Initialize user with API key and some queries."""
        self.api_key = "test-api-key"
        self.headers = {"X-API-Key": self.api_key}
        self.queries = [
            "test query",
            "memory search",
            "find similar",
            "cognitive recall",
            "semantic search",
        ]

    @task(20)
    def query_memory(self):
        """Query for similar memories."""
        query = random.choice(self.queries)
        self.client.get(
            f"/query?q={query}&limit=10", headers=self.headers, name="/query [SEARCH]"
        )


class MixedUser(HttpUser):
    """User that performs both store and query operations."""

    wait_time = between(0.1, 0.3)

    def on_start(self):
        """Initialize user."""
        self.api_key = "test-api-key"
        self.headers = {"X-API-Key": self.api_key}

    @task(3)
    def store_memory(self):
        """Store a random memory."""
        payload = {"content": random_content(150), "metadata": {"source": "mixed_user"}}
        self.client.post(
            "/memories", json=payload, headers=self.headers, name="/memories [STORE]"
        )

    @task(7)
    def query_memory(self):
        """Query for memories."""
        self.client.get(
            "/query?q=test&limit=5", headers=self.headers, name="/query [SEARCH]"
        )

    @task(1)
    def health_check(self):
        """Check API health."""
        self.client.get("/health", name="/health [CHECK]")
