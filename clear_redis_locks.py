"""
Utility script to clear Redis locks.
"""
import redis

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Get all lock keys
lock_keys = r.keys("job:*:lock")

if lock_keys:
    print(f"Found {len(lock_keys)} locks")
    for key in lock_keys:
        print(f"  Deleting: {key}")
        r.delete(key)
    print(f"Cleared {len(lock_keys)} locks")
else:
    print("No locks found")

# Also show active jobs counter
active_jobs = r.get("agent:active_jobs")
print(f"\nActive jobs counter: {active_jobs or 0}")

# Reset active jobs counter
r.set("agent:active_jobs", 0)
print("Reset active jobs counter to 0")
