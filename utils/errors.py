from enum import Enum
from typing import Dict, Any, Optional

class ErrorDetail:
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message

class ErrorCode:
    """System error code and message definitions"""
    
    # Common errors 
    COMMON_INTERNAL_ERROR = ErrorDetail("COMMON_INTERNAL_ERROR", "An unexpected internal server error occurred.")
    COMMON_NOT_FOUND = ErrorDetail("COMMON_NOT_FOUND", "The requested resource does not exist")
    COMMON_INVALID_PARAMS = ErrorDetail("COMMON_INVALID_PARAMS", "Invalid parameters")
    COMMON_VALIDATION_ERROR = ErrorDetail("COMMON_VALIDATION_ERROR", "Data validation failed")
    COMMON_SERVICE_UNAVAILABLE = ErrorDetail("COMMON_SERVICE_UNAVAILABLE", "Service unavailable")
    COMMON_UNAUTHORIZED = ErrorDetail("COMMON_UNAUTHORIZED", "Unauthorized")
    COMMON_INSUFFICIENT_PERMISSIONS = ErrorDetail("COMMON_INSUFFICIENT_PERMISSIONS", "Insufficient permissions")
    
    # Authentication related errors 
    AUTH_INVALID_CREDENTIALS = ErrorDetail("AUTH_INVALID_CREDENTIALS", "Invalid credentials")
    AUTH_INVALID_TOKEN = ErrorDetail("AUTH_INVALID_TOKEN", "Invalid access token")
    AUTH_TOKEN_EXPIRED = ErrorDetail("AUTH_TOKEN_EXPIRED", "Access token expired")
    AUTH_INSUFFICIENT_PERMISSIONS = ErrorDetail("AUTH_INSUFFICIENT_PERMISSIONS", "Insufficient permissions")
    AUTH_USER_EXISTS = ErrorDetail("AUTH_USER_EXISTS", "User already exists")
    AUTH_REFRESH_TOKEN_NOT_FOUND = ErrorDetail("AUTH_REFRESH_TOKEN_NOT_FOUND", "Refresh token not found")
    AUTH_USER_NOT_FOUND = ErrorDetail("AUTH_USER_NOT_FOUND", "User not found")

    # Model related errors 
    MODEL_NOT_FOUND = ErrorDetail("MODEL_NOT_FOUND", "Model not found")
    MODEL_NOT_PUBLISHED = ErrorDetail("MODEL_NOT_PUBLISHED", "Model not published")
    MODEL_LOAD_ERROR = ErrorDetail("MODEL_LOAD_ERROR", "Model load failed")
    MODEL_EXISTS = ErrorDetail("MODEL_EXISTS", "Model already exists")
    MODEL_VALIDATION_ERROR = ErrorDetail("MODEL_VALIDATION_ERROR", "Model validation failed")
    
    # Inference related errors 
    INFERENCE_QUEUE_FULL = ErrorDetail("INFERENCE_QUEUE_FULL", "Inference queue is full, please try again later")
    INFERENCE_TIMEOUT = ErrorDetail("INFERENCE_TIMEOUT", "Inference request timeout")
    INFERENCE_FAILED = ErrorDetail("INFERENCE_FAILED", "Inference execution failed")
    INFERENCE_INPUT_ERROR = ErrorDetail("INFERENCE_INPUT_ERROR", "Inference input data error")
    
    # Token related errors 
    TOKEN_NOT_FOUND = ErrorDetail("TOKEN_NOT_FOUND", "Token not found")
    TOKEN_NAME_ALREADY_TAKEN = ErrorDetail("TOKEN_NAME_ALREADY_TAKEN", "Token name already taken")
    TOKEN_RATE_LIMIT_EXCEEDED = ErrorDetail("TOKEN_RATE_LIMIT_EXCEEDED", "Request rate limit exceeded")
    TOKEN_QUOTA_EXCEEDED = ErrorDetail("TOKEN_QUOTA_EXCEEDED", "Request quota exceeded")
    TOKEN_IP_FORBIDDEN = ErrorDetail("TOKEN_IP_FORBIDDEN", "Current IP address is not in the allowed list")
    
    # System configuration errors 
    CONFIG_INVALID = ErrorDetail("CONFIG_INVALID", "Invalid system configuration")