# cinfer/auth/permissions.py
from enum import Enum
from typing import List

class Scope(str, Enum):
    """
    Defines API access scopes for fine-grained permission control.
    These would be assigned to tokens.
    """
    # Model Management Scopes
    MODEL_READ = "model:read"             # Allows listing and reading model details
    MODEL_REGISTER = "model:register"       # Allows uploading/registering new models
    MODEL_UPDATE = "model:update"         # Allows updating existing models
    MODEL_MANAGE = "model:manage"         # Allows publishing, unpublishing, deleting models (potentially includes update)
    
    # Inference Scopes
    INFERENCE_PREDICT = "inference:predict" # Allows making inference calls

    # Token Management Scopes (for admins or token owners if self-service)
    TOKEN_READ_SELF = "token:read_self"   # Allows reading own tokens
    TOKEN_MANAGE_SELF = "token:manage_self" # Allows managing own tokens
    TOKEN_MANAGE_ALL = "token:manage_all" # Allows managing all tokens (admin)
    
    # System Admin Scopes
    SYSTEM_READ_INFO = "system:read_info" # Allows reading system information and status
    ADMIN_FULL_ACCESS = "admin:full_access" # Superuser scope, grants all permissions


# Example helper function (could be part of AuthService or a utility class)
def check_scopes(required_scopes: List[Scope], token_scopes: List[str]) -> bool:
    """
    Checks if all required scopes are present in the token's scopes.
    """
    if not required_scopes: # No specific scopes required, access granted
        return True
    if Scope.ADMIN_FULL_ACCESS in token_scopes: # Admin bypasses specific scope checks
        return True
    return all(scope.value in token_scopes for scope in required_scopes)