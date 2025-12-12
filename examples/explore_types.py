"""
Explore A2A Types and Message Structure
This script investigates the types module to understand message structure,
configuration options, and any limits or constraints.
"""

import inspect
from typing import Any, get_args, get_origin
from pydantic import BaseModel


def explore_pydantic_model(model_class: type[BaseModel]) -> dict[str, Any]:
    """Explore a Pydantic model's fields and constraints."""
    if not hasattr(model_class, 'model_fields'):
        return {}

    result = {
        "class_name": model_class.__name__,
        "doc": inspect.getdoc(model_class),
        "fields": []
    }

    try:
        for field_name, field_info in model_class.model_fields.items():
            field_data = {
                "name": field_name,
                "type": str(field_info.annotation),
                "required": field_info.is_required(),
                "default": str(field_info.default) if field_info.default is not None else None,
                "description": field_info.description,
            }

            # Check for constraints
            if hasattr(field_info, 'constraints'):
                field_data["constraints"] = str(field_info.constraints)

            # Check for metadata
            if hasattr(field_info, 'metadata'):
                field_data["metadata"] = [str(m) for m in field_info.metadata]

            result["fields"].append(field_data)
    except Exception as e:
        result["error"] = str(e)

    return result


def print_model_info(model_info: dict[str, Any]) -> None:
    """Pretty print model information."""
    print(f"\n{'='*80}")
    print(f"MODEL: {model_info['class_name']}")
    print('='*80)

    if model_info.get('doc'):
        print(f"Description: {model_info['doc'][:300]}")

    if model_info.get('error'):
        print(f"ERROR: {model_info['error']}")
        return

    print(f"\nFields ({len(model_info.get('fields', []))}):")
    print("-" * 80)

    for field in model_info.get('fields', []):
        required_marker = "[REQUIRED]" if field['required'] else "[OPTIONAL]"
        print(f"\n  {required_marker} {field['name']}: {field['type']}")

        if field.get('description'):
            print(f"    Description: {field['description']}")

        if field.get('default'):
            print(f"    Default: {field['default']}")

        if field.get('constraints'):
            print(f"    Constraints: {field['constraints']}")

        if field.get('metadata'):
            print(f"    Metadata: {field['metadata']}")


if __name__ == "__main__":
    import a2a.types as types

    print("="*80)
    print("A2A TYPES EXPLORATION")
    print("="*80)

    # Key types to explore
    key_types = [
        "Message",
        "MessagePart",
        "Artifact",
        "Task",
        "TaskStatusUpdateEvent",
        "TaskArtifactUpdateEvent",
        "MessageSendConfiguration",
        "AgentCard",
        "SendMessageRequest",
        "SendStreamingMessageRequest",
        "SendMessageResponse",
        "SendStreamingMessageResponse",
        "TextPart",
        "DataPart",
        "ResourcePart",
    ]

    for type_name in key_types:
        if hasattr(types, type_name):
            type_class = getattr(types, type_name)
            if inspect.isclass(type_class) and issubclass(type_class, BaseModel):
                model_info = explore_pydantic_model(type_class)
                print_model_info(model_info)
        else:
            print(f"\n{'='*80}")
            print(f"MODEL: {type_name}")
            print('='*80)
            print("NOT FOUND in a2a.types")

    print("\n\n" + "="*80)
    print("Exploring all available types in a2a.types")
    print("="*80)

    all_types = []
    for name in dir(types):
        if not name.startswith("_"):
            obj = getattr(types, name)
            if inspect.isclass(obj):
                try:
                    if issubclass(obj, BaseModel):
                        all_types.append(name)
                except:
                    pass

    print(f"\nFound {len(all_types)} Pydantic models:")
    for i, type_name in enumerate(sorted(all_types), 1):
        print(f"  {i}. {type_name}")

    print("\n" + "="*80)
    print("EXPLORATION COMPLETE")
    print("="*80)
