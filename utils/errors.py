from enum import Enum
from typing import Dict, Any, Optional
from fastapi import status

class ErrorDetail:
    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message

class ErrorCode:
    """System error code and message definitions"""
    
    # Common errors 
    COMMON_INTERNAL_ERROR = ErrorDetail(status.HTTP_500_INTERNAL_SERVER_ERROR, "COMMON_INTERNAL_ERROR", "An unexpected internal server error occurred.")
    COMMON_NOT_FOUND = ErrorDetail(status.HTTP_404_NOT_FOUND, "COMMON_NOT_FOUND", "The requested resource does not exist")
    COMMON_INVALID_PARAMS = ErrorDetail(status.HTTP_400_BAD_REQUEST, "COMMON_INVALID_PARAMS", "Invalid parameters")
    COMMON_VALIDATION_ERROR = ErrorDetail(status.HTTP_400_BAD_REQUEST, "COMMON_VALIDATION_ERROR", "Data validation failed")
    COMMON_SERVICE_UNAVAILABLE = ErrorDetail(status.HTTP_503_SERVICE_UNAVAILABLE, "COMMON_SERVICE_UNAVAILABLE", "Service unavailable")
    COMMON_UNAUTHORIZED = ErrorDetail(status.HTTP_401_UNAUTHORIZED, "COMMON_UNAUTHORIZED", "Unauthorized")
    COMMON_INSUFFICIENT_PERMISSIONS = ErrorDetail(status.HTTP_403_FORBIDDEN, "COMMON_INSUFFICIENT_PERMISSIONS", "Insufficient permissions")
    COMMON_FORBIDDEN = ErrorDetail(status.HTTP_403_FORBIDDEN, "COMMON_FORBIDDEN", "Forbidden")
    COMMON_BAD_REQUEST = ErrorDetail(status.HTTP_400_BAD_REQUEST, "COMMON_BAD_REQUEST", "Bad request")
    # Authentication related errors 
    AUTH_INVALID_CREDENTIALS = ErrorDetail(status.HTTP_401_UNAUTHORIZED, "AUTH_INVALID_CREDENTIALS", "Invalid credentials")
    AUTH_INVALID_TOKEN = ErrorDetail(status.HTTP_401_UNAUTHORIZED, "AUTH_INVALID_TOKEN", "Invalid access token")
    AUTH_TOKEN_EXPIRED = ErrorDetail(status.HTTP_401_UNAUTHORIZED, "AUTH_TOKEN_EXPIRED", "Access token expired")
    AUTH_INSUFFICIENT_PERMISSIONS = ErrorDetail(status.HTTP_403_FORBIDDEN, "AUTH_INSUFFICIENT_PERMISSIONS", "Insufficient permissions")
    AUTH_USER_EXISTS = ErrorDetail(status.HTTP_400_BAD_REQUEST, "AUTH_USER_EXISTS", "User already exists")
    AUTH_REFRESH_TOKEN_NOT_FOUND = ErrorDetail(status.HTTP_401_UNAUTHORIZED, "AUTH_REFRESH_TOKEN_NOT_FOUND", "Refresh token not found")
    AUTH_USER_NOT_FOUND = ErrorDetail(status.HTTP_404_NOT_FOUND, "AUTH_USER_NOT_FOUND", "User not found")

    # Model related errors 
    MODEL_NOT_FOUND = ErrorDetail(status.HTTP_404_NOT_FOUND, "MODEL_NOT_FOUND", "Model not found")
    MODEL_REGISTRATION_FAILED = ErrorDetail(status.HTTP_400_BAD_REQUEST, "MODEL_REGISTRATION_FAILED", "Model registration failed")
    MODEL_NOT_PUBLISHED = ErrorDetail(status.HTTP_404_NOT_FOUND, "MODEL_NOT_PUBLISHED", "Model not published")
    MODEL_LOAD_ERROR = ErrorDetail(status.HTTP_500_INTERNAL_SERVER_ERROR, "MODEL_LOAD_ERROR", "Model load failed")
    MODEL_EXISTS = ErrorDetail(status.HTTP_400_BAD_REQUEST, "MODEL_EXISTS", "Model already exists")
    MODEL_VALIDATION_ERROR = ErrorDetail(status.HTTP_400_BAD_REQUEST, "MODEL_VALIDATION_ERROR", "Model validation failed")
    MODEL_YAML_NOT_FOUND = ErrorDetail(status.HTTP_404_NOT_FOUND, "MODEL_YAML_NOT_FOUND", "Model YAML file not found")
    MODEL_PUBLISH_FAILED = ErrorDetail(status.HTTP_500_INTERNAL_SERVER_ERROR, "MODEL_PUBLISH_FAILED", "Model publish failed")
    MODEL_UPDATE_FAILED = ErrorDetail(status.HTTP_500_INTERNAL_SERVER_ERROR, "MODEL_UPDATE_FAILED", "Model update failed")
    
    # Inference related errors 
    INFERENCE_QUEUE_FULL = ErrorDetail(status.HTTP_503_SERVICE_UNAVAILABLE, "INFERENCE_QUEUE_FULL", "Inference queue is full, please try again later")
    INFERENCE_TIMEOUT = ErrorDetail(status.HTTP_504_GATEWAY_TIMEOUT, "INFERENCE_TIMEOUT", "Inference request timeout")
    INFERENCE_FAILED = ErrorDetail(status.HTTP_500_INTERNAL_SERVER_ERROR, "INFERENCE_FAILED", "Inference execution failed")
    INFERENCE_INPUT_ERROR = ErrorDetail(status.HTTP_400_BAD_REQUEST, "INFERENCE_INPUT_ERROR", "Inference input data error")
    
    # Token related errors 
    TOKEN_NOT_FOUND = ErrorDetail(status.HTTP_404_NOT_FOUND, "TOKEN_NOT_FOUND", "Token not found")
    TOKEN_NAME_ALREADY_TAKEN = ErrorDetail(status.HTTP_400_BAD_REQUEST, "TOKEN_NAME_ALREADY_TAKEN", "Token name already taken")
    TOKEN_RATE_LIMIT_EXCEEDED = ErrorDetail(status.HTTP_429_TOO_MANY_REQUESTS, "TOKEN_RATE_LIMIT_EXCEEDED", "Request rate limit exceeded")
    TOKEN_QUOTA_EXCEEDED = ErrorDetail(status.HTTP_403_FORBIDDEN, "TOKEN_QUOTA_EXCEEDED", "Request quota exceeded")
    TOKEN_IP_FORBIDDEN = ErrorDetail(status.HTTP_403_FORBIDDEN, "TOKEN_IP_FORBIDDEN", "Current IP address is not in the allowed list")
    
