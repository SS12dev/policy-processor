from a2a import types

fields = types.AgentSkill.model_fields
for name, field in fields.items():
    print(f'{name}: required={field.is_required()}, default={field.default}')
