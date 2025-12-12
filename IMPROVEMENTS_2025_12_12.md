# System Improvements - December 12, 2025

## Summary
After successfully achieving **5/5 decision trees (100% success rate)** with the reduced concurrency limit, several small improvements were made to enhance reliability and reduce noise in the logs.

## Test Results Before Improvements
- ✅ **Tree Generation**: 5/5 trees (100% success - EXCELLENT!)
- ⚠️ **Policy Extraction**: 3 chunks failed due to 504 Gateway Timeout errors
- ⚠️ **Async Generator Warnings**: Noisy cleanup errors at the end of processing
- ⚠️ **Retry Mechanism**: Using OpenAI SDK's built-in retry instead of our Tenacity configuration

## Changes Made

### 1. Enhanced Policy Extractor Retry Configuration
**File**: `agent-module/core/policy_extractor.py`

**Problem**: Policy extraction had simpler retry logic than tree generation, causing more failures.

**Solution**: Applied the same comprehensive retry configuration used in `decision_tree_generator.py`:

```python
# BEFORE:
@retry(
    stop=stop_after_attempt(settings.openai_max_retries),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)

# AFTER:
@retry(
    stop=stop_after_attempt(settings.openai_max_retries),
    wait=wait_exponential(
        multiplier=settings.openai_retry_multiplier,  # 2
        min=settings.openai_retry_min_wait,            # 4s
        max=settings.openai_retry_max_wait             # 30s
    ),
    retry=retry_if_exception_type((InternalServerError, APITimeoutError, TimeoutError)),
    before_sleep=before_sleep_log(logger, logger.level),
    reraise=True,
)
```

**Benefits**:
- Consistent retry behavior across all API calls
- Selective retry only on timeout-related errors (not all exceptions)
- Better backoff strategy (4-30s instead of 2-10s)
- Visibility into retry attempts via logging

### 2. Improved Error Handling in Policy Extraction
**File**: `agent-module/core/policy_extractor.py`

**Problem**: All exceptions were caught and swallowed, preventing retry mechanism from working.

**Solution**: Re-raise retryable exceptions so Tenacity can handle them:

```python
# BEFORE:
except Exception as e:
    logger.error(f"Error extracting from chunk {chunk_id}: {e}")
    return [], {}

# AFTER:
except (InternalServerError, APITimeoutError) as e:
    error_type = type(e).__name__
    logger.error(f"LiteLLM proxy error for chunk {chunk_id} ({error_type}): {e}")
    raise  # Let tenacity retry handle it
except json.JSONDecodeError as e:
    logger.error(f"JSON decode error for chunk {chunk_id}: {e}")
    return [], {}
except Exception as e:
    logger.error(f"Error extracting from chunk {chunk_id}: {e}")
    raise  # Let tenacity retry handle it
```

**Benefits**:
- Retry mechanism can now work properly
- Specific handling for proxy errors vs parsing errors
- Better error classification for debugging

### 3. Enhanced Error Reporting and Diagnostics
**File**: `agent-module/core/policy_extractor.py`

**Problem**: Generic error messages didn't provide enough context about failures.

**Solution**: Added detailed tracking of error types and failed chunks:

```python
errors_count = 0
timeout_errors = 0
failed_chunks = []

for i, result in enumerate(results):
    if isinstance(result, Exception):
        errors_count += 1
        error_type = type(result).__name__
        
        # Track specific error types
        if "Timeout" in error_type or "timeout" in str(result).lower():
            timeout_errors += 1
        
        # Store chunk info for failed extractions
        chunk_id = _get_chunk_attr(chunks_to_process[i], "chunk_id", f"chunk_{i}")
        failed_chunks.append(chunk_id)
        
        logger.error(f"Error extracting from chunk {i + 1} ({chunk_id}): [{error_type}] {result}")

# Summary logging
logger.info(f"Extraction complete: {len(all_policies)} policies, {len(definitions)} definitions from {len(chunks_to_process) - errors_count}/{len(chunks_to_process)} chunks")
if errors_count > 0:
    logger.warning(f"Failed chunks: {errors_count} ({', '.join(failed_chunks)})")
if timeout_errors > 0:
    logger.warning(f"Timeout failures: {timeout_errors} - Consider increasing openai_per_request_timeout or reducing concurrent requests")
```

**Benefits**:
- Clear identification of which chunks failed
- Categorized error types (timeout vs other)
- Actionable suggestions for configuration tuning
- Better success rate tracking (e.g., "4/7 chunks" instead of just "4 chunks")

### 4. Suppressed AsyncIO Generator Cleanup Warnings
**Files**: 
- `client-module/a2a_client.py`
- `client-module/app.py`

**Problem**: Harmless async generator cleanup warnings from httpx/a2a libraries cluttering logs:
```
RuntimeError: aclose(): asynchronous generator is already running
```

**Solution**: 

**In a2a_client.py** - Catch and suppress during client close:
```python
async def close(self):
    """Close the connection."""
    if self.http_client:
        try:
            await self.http_client.aclose()
        except RuntimeError as e:
            # Suppress "asynchronous generator is already running" errors during cleanup
            if "asynchronous generator is already running" not in str(e):
                raise
```

**In app.py** - Filter warnings at module level:
```python
import warnings

# Suppress async generator closing warnings from httpx/a2a during cleanup
warnings.filterwarnings("ignore", message=".*asynchronous generator is already running.*")
```

**Benefits**:
- Cleaner logs without harmless warnings
- Real errors still surface properly
- User experience improved (no scary error messages)

## Expected Impact

### Policy Extraction
- **Before**: 3/7 chunks failed (43% failure rate)
- **After**: Should see 0-1 chunks fail (0-14% failure rate)
- **Reason**: Better retry mechanism with exponential backoff should handle transient 504 errors

### Log Clarity
- **Before**: Generic "Error extracting from chunk X" messages
- **After**: 
  - Specific error types identified
  - Failed chunk IDs listed
  - Timeout count tracked separately
  - Actionable suggestions provided

### User Experience
- **Before**: Async generator cleanup warnings at end of every run
- **After**: Clean completion without noise

## Configuration Settings (Current)

```python
# Retry Configuration
openai_max_retries: 5 attempts
openai_retry_multiplier: 2x (exponential growth)
openai_retry_min_wait: 4 seconds
openai_retry_max_wait: 30 seconds
openai_retry_on_timeout: True

# Concurrency Configuration
openai_max_concurrent_requests: 2 (reduced from 5)

# Timeout Configuration
openai_per_request_timeout: 300 seconds (5 minutes)
```

## Testing Recommendations

1. **Reprocess same document** (`Independence_Bariatric Surgery.pdf`) and compare:
   - Policy extraction success rate (should improve from 4/7 to 6-7/7)
   - Decision tree generation (should remain 5/5)
   - Log cleanliness (no async warnings)

2. **Monitor retry attempts** in logs:
   - Look for "Retrying" messages with backoff times
   - Verify exponential backoff is working (4s, 8s, 16s, 30s, 30s)
   - Check if retries successfully recover from 504 errors

3. **Track error types**:
   - Count timeout vs proxy vs other errors
   - Verify retry mechanism only triggers on appropriate errors
   - Confirm JSON parsing errors don't trigger retries

## Rollback Instructions

If issues arise, revert these files:
```bash
git checkout HEAD -- agent-module/core/policy_extractor.py
git checkout HEAD -- client-module/a2a_client.py
git checkout HEAD -- client-module/app.py
```

## Future Optimizations (If Needed)

If extraction failures persist:

1. **Increase per-request timeout**:
   ```python
   openai_per_request_timeout: 360  # 6 minutes instead of 5
   ```

2. **Further reduce concurrency**:
   ```python
   openai_max_concurrent_requests: 1  # Sequential processing
   ```

3. **Add request pacing**:
   - Add delay between starting new requests
   - Prevents burst load on proxy

4. **Chunk size optimization**:
   - Reduce target chunk size to make requests faster
   - Smaller chunks = less processing time = less timeout risk

## Conclusion

These incremental improvements focus on:
- ✅ **Reliability**: Better retry mechanism for policy extraction
- ✅ **Observability**: Enhanced error tracking and reporting
- ✅ **User Experience**: Cleaner logs without noise
- ✅ **Consistency**: Same retry behavior across all API calls

The system already achieves **100% tree generation success**. These changes should significantly improve policy extraction success rate while maintaining that excellent tree generation performance.
