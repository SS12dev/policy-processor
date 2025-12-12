"""
Deep exploration of A2A SDK components
This script thoroughly explores client, server, types, and other key components.
"""

import inspect
import json
from typing import Any

def explore_classes_in_module(module_path: str) -> dict[str, Any]:
    """Deeply explore classes in a module."""
    try:
        parts = module_path.split('.')
        module = __import__(module_path, fromlist=[parts[-1]])

        result = {
            "module_path": module_path,
            "classes": [],
            "functions": [],
            "constants": []
        }

        for name, obj in inspect.getmembers(module):
            if name.startswith("_"):
                continue

            if inspect.isclass(obj):
                class_info = {
                    "name": name,
                    "doc": inspect.getdoc(obj),
                    "bases": [base.__name__ for base in obj.__bases__ if base.__name__ != 'object'],
                    "methods": [],
                    "attributes": [],
                    "init_signature": None
                }

                # Get __init__ signature
                try:
                    if hasattr(obj, '__init__'):
                        sig = inspect.signature(obj.__init__)
                        class_info["init_signature"] = str(sig)
                except:
                    pass

                # Get methods and attributes
                for member_name, member_obj in inspect.getmembers(obj):
                    if member_name.startswith("_") and member_name not in ["__init__", "__call__", "__enter__", "__exit__", "__aenter__", "__aexit__"]:
                        continue

                    if inspect.ismethod(member_obj) or inspect.isfunction(member_obj):
                        try:
                            sig = str(inspect.signature(member_obj))
                            doc = inspect.getdoc(member_obj)
                            class_info["methods"].append({
                                "name": member_name,
                                "signature": sig,
                                "doc": doc[:300] if doc else None,
                                "is_async": inspect.iscoroutinefunction(member_obj)
                            })
                        except:
                            pass
                    elif not callable(member_obj) and not member_name.startswith("_"):
                        class_info["attributes"].append({
                            "name": member_name,
                            "type": type(member_obj).__name__
                        })

                result["classes"].append(class_info)

            elif inspect.isfunction(obj):
                try:
                    sig = str(inspect.signature(obj))
                    result["functions"].append({
                        "name": name,
                        "signature": sig,
                        "doc": inspect.getdoc(obj),
                        "is_async": inspect.iscoroutinefunction(obj)
                    })
                except:
                    pass

        return result
    except Exception as e:
        return {"module_path": module_path, "error": str(e)}

def print_exploration_results(results: dict[str, Any]) -> None:
    """Pretty print exploration results."""
    if "error" in results:
        print(f"ERROR exploring {results['module_path']}: {results['error']}")
        return

    print(f"\n{'='*80}")
    print(f"MODULE: {results['module_path']}")
    print('='*80)

    if results.get("classes"):
        print(f"\nCLASSES ({len(results['classes'])})")
        print("-" * 80)
        for cls in results["classes"]:
            print(f"\n  [{cls['name']}]")
            if cls.get("bases"):
                print(f"     Inherits from: {', '.join(cls['bases'])}")
            if cls.get("doc"):
                doc_lines = cls["doc"].split('\n')[:3]
                for line in doc_lines:
                    if line.strip():
                        print(f"     {line.strip()}")
            if cls.get("init_signature"):
                print(f"     Constructor: __init__{cls['init_signature']}")

            if cls.get("methods"):
                print(f"     Methods ({len(cls['methods'])}):")
                for method in cls["methods"]:
                    async_marker = "[async] " if method.get("is_async") else ""
                    print(f"       - {async_marker}{method['name']}{method['signature']}")
                    if method.get("doc"):
                        doc_preview = method["doc"].replace('\n', ' ')[:150]
                        print(f"         > {doc_preview}")

            if cls.get("attributes"):
                print(f"     Attributes ({len(cls['attributes'])}):")
                for attr in cls["attributes"]:
                    print(f"       - {attr['name']}: {attr['type']}")

    if results.get("functions"):
        print(f"\nFUNCTIONS ({len(results['functions'])})")
        print("-" * 80)
        for func in results["functions"]:
            async_marker = "[async] " if func.get("is_async") else ""
            print(f"  - {async_marker}{func['name']}{func['signature']}")
            if func.get("doc"):
                doc_preview = func["doc"].replace('\n', ' ')[:150]
                print(f"    > {doc_preview}")

if __name__ == "__main__":
    print("="*80)
    print("DEEP A2A SDK EXPLORATION")
    print("="*80)

    # Core modules to explore
    modules_to_explore = [
        "a2a.client",
        "a2a.server",
        "a2a.types",
        "a2a.utils",
        "a2a.extensions",
        "a2a.grpc"
    ]

    for module_path in modules_to_explore:
        results = explore_classes_in_module(module_path)
        print_exploration_results(results)

    print("\n" + "="*80)
    print("EXPLORATION COMPLETE")
    print("="*80)
