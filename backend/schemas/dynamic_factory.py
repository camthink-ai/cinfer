from pydantic import BaseModel, Field, create_model, ValidationError
from typing import List, Dict, Any, Optional, Type
from schemas.engine import InputOutputDefinition

def create_dynamic_model_from_definition(
    base_name: str, 
    definitions: List[InputOutputDefinition]
) -> Type[BaseModel]:
    """
    Create a Pydantic model class dynamically based on a list of field definitions.

    Args:
        base_name (str): The name of the Pydantic model to be created.
        definitions (List[InputOutputDefinition]): A list of field definitions.

    Returns:
        Type[BaseModel]: The newly created Pydantic model class.
    """
    fields: Dict[str, Any] = {}
    
    type_mapping = {
        "string": str,
        "float": float,
        "integer": int,
        "boolean": bool,
    }

    for definition in definitions:
        field_type = type_mapping.get(definition.type, Any)

        # Example: Handle simple nested cases, possibly more complex recursive
        if definition.type == "array" and definition.items:
            # Here you can recursively call to create more complex nested models
            # For simplicity, we assume the array contains simple types or Any
            inner_type = type_mapping.get(definition.items.type, Any)
            field_type = List[inner_type]

        field_info = Field(
            default=... if definition.required else definition.default,
            description=definition.description,
            ge=definition.minimum,
            le=definition.maximum,
        )
        fields[definition.name] = (field_type, field_info)
    
    # Use pydantic.create_model to dynamically create the model class
    DynamicModel = create_model(base_name, **fields)
    return DynamicModel