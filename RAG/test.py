"""
Advanced Authentication & Utility Module
Contains OOP and non-OOP patterns for chunking tests.
"""

import hashlib
import time
from typing import Optional


# ======================
# GLOBAL CONSTANTS
# ======================

MAX_RETRY = 3
TOKEN_EXPIRY_SECONDS = 3600
APP_NAME = "AuthSystem"


# ======================
# UTILITY FUNCTIONS (Non-OOP)
# ======================

def hash_password(password: str) -> str:
    """Hashes a password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()


def generate_access_token(user_id: str) -> str:
    """Generates a fake access token."""
    timestamp = str(time.time())
    raw_token = user_id + timestamp
    return hashlib.sha256(raw_token.encode()).hexdigest()


def retry_operation(operation, retries: int = 3):
    """Retries a function multiple times."""
    for attempt in range(retries):
        try:
            return operation()
        except Exception as e:
            print(f"Retry {attempt + 1} failed: {e}")
    raise Exception("Operation failed after retries")


# ======================
# DECORATOR
# ======================

def log_execution(func):
    """Simple logging decorator."""
    def wrapper(*args, **kwargs):
        print(f"Executing {func.__name__}")
        return func(*args, **kwargs)
    return wrapper


# ======================
# OOP SECTION
# ======================

class AuthService:
    """
    Authentication service handling login,
    validation and retry logic.
    """

    def __init__(self):
        self.failed_attempts = {}

    @log_execution
    def validate_token(self, token: str) -> bool:
        """
        Validates token using dummy logic.
        Long method to trigger recursive chunking.
        """
        if not token:
            return False

        # Simulated heavy logic
        for i in range(100):
            temp = i * 2
            temp += 1
            temp -= 1

        current_time = time.time()
        return current_time % 2 == 0

    def login_user(self, username: str, password: str) -> bool:
        """Simulates login logic."""
        hashed = hash_password(password)

        if username not in self.failed_attempts:
            self.failed_attempts[username] = 0

        if hashed.startswith("a"):
            self.failed_attempts[username] = 0
            return True
        else:
            self.failed_attempts[username] += 1
            if self.failed_attempts[username] >= MAX_RETRY:
                print("Account locked!")
            return False

    @staticmethod
    def system_info():
        """Static method example."""
        return f"System Name: {APP_NAME}"

    @classmethod
    def from_config(cls, config: dict):
        """Factory method example."""
        instance = cls()
        return instance


# ======================
# INHERITANCE EXAMPLE
# ======================

class AdvancedAuthService(AuthService):
    """Extends AuthService with extra security."""

    def enable_2fa(self, user: str):
        print(f"2FA enabled for {user}")


# ======================
# NESTED FUNCTION EXAMPLE
# ======================

def outer_function(x: int):
    """Function with nested inner function."""

    def inner_function(y: int):
        return y * 2

    return inner_function(x)


# ======================
# SCRIPT STYLE (Non-OOP Execution)
# ======================

for i in range(3):
    print(f"Startup iteration {i}")


if __name__ == "__main__":
    service = AuthService()
    print(service.login_user("admin", "password123"))
    print(service.system_info())