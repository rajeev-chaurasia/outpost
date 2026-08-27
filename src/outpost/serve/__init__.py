"""The served API: ask a tenant-scoped question, list tenants, and read
the audit trail.
"""

from outpost.serve.app import create_app

__all__ = ["create_app"]
