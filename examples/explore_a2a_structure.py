"""
A2A SDK Structure Explorer
This script explores the a2a-sdk library structure, components, and capabilities.
"""

import inspect
import sys
from typing import Any

def explore_module_structure(module_name: str) -> dict[str, Any]:
    """Explore and document the structure of a module."""
    try:
        module = __import__(module_name)

        structure = {
            "module_name": module_name,
            "doc": module.__doc__,
            "version": getattr(module, "__version__", "N/A"),
            "file": getattr(module, "__file__", "N/A"),
            "submodules": [],
            "classes": [],
            "functions": [],
            "constants": []
        }

        # Get all members
        members = inspect.getmembers(module)

        for name, obj in members:
            if name.startswith("_"):
                continue

            if inspect.ismodule(obj):
                structure["submodules"].append({
                    "name": name,
                    "doc": obj.__doc__,
                    "file": getattr(obj, "__file__", "N/A")
                })
            elif inspect.isclass(obj):
                methods = []
                properties = []
                for method_name, method_obj in inspect.getmembers(obj):
                    if not method_name.startswith("_") or method_name in ["__init__", "__call__"]:
                        if inspect.ismethod(method_obj) or inspect.isfunction(method_obj):
                            try:
                                sig = str(inspect.signature(method_obj))
                                methods.append({
                                    "name": method_name,
                                    "signature": sig,
                                    "doc": inspect.getdoc(method_obj)
                                })
                            except:
                                methods.append({
                                    "name": method_name,
                                    "signature": "N/A",
                                    "doc": inspect.getdoc(method_obj)
                                })
                        elif isinstance(inspect.getattr_static(obj, method_name), property):
                            properties.append({
                                "name": method_name,
                                "doc": inspect.getdoc(method_obj)
                            })

                structure["classes"].append({
                    "name": name,
                    "doc": inspect.getdoc(obj),
                    "bases": [base.__name__ for base in obj.__bases__],
                    "methods": methods,
                    "properties": properties
                })
            elif inspect.isfunction(obj):
                try:
                    sig = str(inspect.signature(obj))
                    structure["functions"].append({
                        "name": name,
                        "signature": sig,
                        "doc": inspect.getdoc(obj)
                    })
                except:
                    structure["functions"].append({
                        "name": name,
                        "signature": "N/A",
                        "doc": inspect.getdoc(obj)
                    })
            elif isinstance(obj, (str, int, float, bool, list, dict, tuple)):
                structure["constants"].append({
                    "name": name,
                    "type": type(obj).__name__,
                    "value": str(obj) if len(str(obj)) < 100 else str(obj)[:100] + "..."
                })

        return structure
    except Exception as e:
        return {"error": str(e), "module_name": module_name}

def explore_nested_module(base_module: str, nested_path: str) -> dict[str, Any]:
    """Explore nested modules within a package."""
    try:
        full_path = f"{base_module}.{nested_path}" if nested_path else base_module
        return explore_module_structure(full_path)
    except Exception as e:
        return {"error": str(e), "module_path": f"{base_module}.{nested_path}"}

def print_structure(structure: dict[str, Any], indent: int = 0) -> None:
    """Pretty print the structure."""
    prefix = "  " * indent

    if "error" in structure:
        print(f"{prefix}ERROR: {structure['error']}")
        return

    print(f"{prefix}Module: {structure['module_name']}")
    if structure.get('version', 'N/A') != 'N/A':
        print(f"{prefix}Version: {structure['version']}")
    if structure.get('doc'):
        print(f"{prefix}Doc: {structure['doc'][:200]}")

    if structure.get('submodules'):
        print(f"{prefix}Submodules ({len(structure['submodules'])}):")
        for submod in structure['submodules']:
            print(f"{prefix}  - {submod['name']}")
            if submod.get('doc'):
                print(f"{prefix}    Doc: {submod['doc'][:100]}")

    if structure.get('classes'):
        print(f"{prefix}Classes ({len(structure['classes'])}):")
        for cls in structure['classes']:
            print(f"{prefix}  - {cls['name']} (bases: {', '.join(cls['bases'])})")
            if cls.get('doc'):
                doc_preview = cls['doc'][:150].replace('\n', ' ')
                print(f"{prefix}    Doc: {doc_preview}")
            if cls.get('methods'):
                print(f"{prefix}    Methods ({len(cls['methods'])}):")
                for method in cls['methods'][:10]:  # Show first 10 methods
                    print(f"{prefix}      - {method['name']}{method['signature']}")
            if cls.get('properties'):
                print(f"{prefix}    Properties ({len(cls['properties'])}):")
                for prop in cls['properties']:
                    print(f"{prefix}      - {prop['name']}")

    if structure.get('functions'):
        print(f"{prefix}Functions ({len(structure['functions'])}):")
        for func in structure['functions']:
            print(f"{prefix}  - {func['name']}{func['signature']}")
            if func.get('doc'):
                doc_preview = func['doc'][:100].replace('\n', ' ')
                print(f"{prefix}    Doc: {doc_preview}")

    if structure.get('constants'):
        print(f"{prefix}Constants ({len(structure['constants'])}):")
        for const in structure['constants'][:10]:  # Show first 10
            print(f"{prefix}  - {const['name']}: {const['type']}")

if __name__ == "__main__":
    print("=" * 80)
    print("A2A SDK STRUCTURE EXPLORATION")
    print("=" * 80)

    # Explore main a2a module
    print("\n--- Main a2a module ---")
    a2a_structure = explore_module_structure("a2a")
    print_structure(a2a_structure)

    # Explore common submodules
    common_submodules = [
        "client",
        "server",
        "protocol",
        "types",
        "models",
        "streaming",
        "exceptions",
        "utils"
    ]

    print("\n\n" + "=" * 80)
    print("EXPLORING SUBMODULES")
    print("=" * 80)

    for submod in common_submodules:
        print(f"\n--- a2a.{submod} ---")
        submod_structure = explore_nested_module("a2a", submod)
        print_structure(submod_structure, indent=0)

    print("\n\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)
