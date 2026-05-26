"""
backend/middleware/auth.py
===========================
JWT helper decorators and role-based access utilities.
"""

from functools import wraps
from flask import jsonify
from flask_jwt_extended import get_jwt_identity, verify_jwt_in_request
from backend.models.user import User


def roles_required(*roles):
    """
    Decorator: ensures the calling JWT user has one of the given roles.

    Usage::

        @app.route("/admin")
        @jwt_required()
        @roles_required("admin", "operator")
        def admin_view():
            ...
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            verify_jwt_in_request()
            user_id = int(get_jwt_identity())
            user = User.query.get(user_id)
            if user is None or user.role not in roles:
                return jsonify({
                    "error": "Insufficient permissions",
                    "required_roles": list(roles),
                }), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def get_current_user() -> User:
    """Return the User object for the current JWT identity, or None."""
    try:
        user_id = int(get_jwt_identity())
        return User.query.get(user_id)
    except Exception:
        return None
