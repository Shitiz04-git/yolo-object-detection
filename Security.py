"""
Security Module - Authentication, Encryption, Input Validation
"""

import hashlib
import hmac
import jwt
import secrets
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify
import re

class SecurityManager:
    """Security management class"""
    
    def __init__(self, secret_key):
        self.secret_key = secret_key
        self._blacklisted_tokens = set()
    
    def generate_api_key(self, name: str, permissions: list) -> str:
        """Generate secure API key"""
        salt = secrets.token_hex(16)
        key = hashlib.sha256(f"{name}{salt}{self.secret_key}".encode()).hexdigest()
        return key
    
    def validate_input(self, data: dict) -> bool:
        """Validate input to prevent injection attacks"""
        # Check for malicious patterns
        dangerous_patterns = [
            r'<script', r'javascript:', r'exec\(', r'eval\(',
            r'SELECT.*FROM', r'DROP TABLE', r'DELETE.*FROM'
        ]
        
        for key, value in data.items():
            if isinstance(value, str):
                for pattern in dangerous_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        return False
        return True
    
    def generate_jwt_token(self, user_id: str) -> str:
        """Generate JWT for authentication"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_jwt_token(self, token: str) -> dict:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            if token in self._blacklisted_tokens:
                return None
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def blacklist_token(self, token: str):
        """Blacklist token on logout"""
        self._blacklisted_tokens.add(token)
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize file names to prevent path traversal"""
        # Remove directory traversal patterns
        filename = filename.replace('..', '')
        filename = re.sub(r'[^\w\.-]', '', filename)
        return filename

# Rate limiter for API endpoints
class RateLimiter:
    """Simple rate limiter for API"""
    
    def __init__(self, max_requests=100, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = datetime.now()
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if (now - req_time).seconds < self.time_window
        ]
        
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        self.requests[client_id].append(now)
        return True