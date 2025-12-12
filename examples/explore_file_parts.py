"""
Explore File Parts and Attachment Capabilities
This script investigates FilePart, FileWithBytes, FileWithUri types
to understand how to attach files in A2A messages.
"""

import inspect
from pydantic import BaseModel
import a2a.types as types


def explore_model_detailed(model_class: type[BaseModel]) -> None:
    """Explore a model in detail."""
    print(f"\n{'='*80}")
    print(f"MODEL: {model_class.__name__}")
    print('='*80)
    print(f"Doc: {inspect.getdoc(model_class)}")

    if hasattr(model_class, 'model_fields'):
        print(f"\nFields:")
        print("-" * 80)
        for field_name, field_info in model_class.model_fields.items():
            required = "[REQUIRED]" if field_info.is_required() else "[OPTIONAL]"
            print(f"\n  {required} {field_name}")
            print(f"    Type: {field_info.annotation}")
            if field_info.description:
                print(f"    Description: {field_info.description}")
            if field_info.default is not None:
                print(f"    Default: {field_info.default}")

            # Check for validation constraints
            if hasattr(field_info, 'constraints') and field_info.constraints:
                print(f"    Constraints: {field_info.constraints}")

    # Show example schema
    print(f"\nJSON Schema:")
    print("-" * 80)
    try:
        schema = model_class.model_json_schema()
        import json
        print(json.dumps(schema, indent=2))
    except Exception as e:
        print(f"Error generating schema: {e}")


if __name__ == "__main__":
    print("="*80)
    print("FILE ATTACHMENT EXPLORATION")
    print("="*80)

    # Explore file-related types
    file_types = [
        "FilePart",
        "FileBase",
        "FileWithBytes",
        "FileWithUri",
        "Part",  # Base type for all parts
    ]

    for type_name in file_types:
        if hasattr(types, type_name):
            type_class = getattr(types, type_name)
            if inspect.isclass(type_class):
                try:
                    if issubclass(type_class, BaseModel):
                        explore_model_detailed(type_class)
                except:
                    print(f"\n{type_name} is not a BaseModel")

    print("\n\n" + "="*80)
    print("PART TYPES SUMMARY")
    print("="*80)

    # Find all Part types
    part_types = []
    for name in dir(types):
        if 'Part' in name and not name.startswith("_"):
            obj = getattr(types, name)
            if inspect.isclass(obj):
                try:
                    if issubclass(obj, BaseModel):
                        part_types.append(name)
                except:
                    pass

    print(f"\nAll Part types found ({len(part_types)}):")
    for part_type in sorted(part_types):
        print(f"  - {part_type}")

    print("\n" + "="*80)
    print("EXPLORATION COMPLETE")
    print("="*80)
