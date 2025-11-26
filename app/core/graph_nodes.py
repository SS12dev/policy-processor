"""
LangGraph Node Implementations for Policy Document Processing.

Each node represents a stage in the processing pipeline and updates the state.
"""
import asyncio

from app.core.pdf_processor import PDFProcessor
from app.core.document_analyzer import DocumentAnalyzer
from app.core.chunking_strategy import ChunkingStrategy
from app.core.policy_extractor import PolicyExtractor
from app.core.decision_tree_generator import DecisionTreeGenerator
from app.core.validator import Validator
from app.core.graph_state import ProcessingState
from app.models.schemas import ProcessingStage
from app.utils.logger import get_logger
from app.utils.redis_client import get_redis_client

logger = get_logger(__name__)


# Helper function to publish status updates
async def _publish_status(state: ProcessingState):
    """Publish status update to Redis for streaming."""
    try:
        redis_client = get_redis_client()
        status_data = {
            "job_id": state["job_id"],
            "stage": state["current_stage"].value,
            "progress_percentage": state["progress_percentage"],
            "message": state["status_message"],
            "errors": state["errors"],
        }
        redis_client.set_status(state["job_id"], status_data)
        redis_client.publish(f"job:{state['job_id']}:status", status_data)
    except Exception as e:
        logger.error(f"Failed to publish status: {e}")


# Node 1: PDF Parsing
async def parse_pdf_node(state: ProcessingState) -> ProcessingState:
    """
    Parse PDF document and extract pages.

    Updates state fields:
    - pdf_bytes
    - pages
    - pdf_metadata
    - structure (basic)
    """
    job_id = state["job_id"]
    logger.info(f"[{job_id}] ========== STAGE 1: PDF PARSING ==========")

    state["current_stage"] = ProcessingStage.PARSING_PDF
    state["progress_percentage"] = 5.0
    state["status_message"] = "Parsing PDF document..."
    state["logs"].append(f"Starting PDF parsing for job {job_id}")

    if state.get("enable_streaming", True):
        await _publish_status(state)

    try:
        # Process PDF (processor will handle base64 decoding)
        import base64
        pdf_processor = PDFProcessor()
        pages, pdf_metadata = pdf_processor.process_document(state["document_base64"])

        # Store the decoded bytes for later use
        state["pdf_bytes"] = base64.b64decode(state["document_base64"])

        state["pages"] = pages
        state["pdf_metadata"] = pdf_metadata

        logger.info(
            f"[{job_id}] PDF parsing complete: {pdf_metadata['total_pages']} pages, "
            f"{len([p for p in pages if p.tables])} pages with tables"
        )
        state["logs"].append(
            f"PDF parsed: {pdf_metadata['total_pages']} pages, "
            f"{len([p for p in pages if p.tables])} with tables"
        )

    except Exception as e:
        logger.error(f"[{job_id}] PDF parsing failed: {e}", exc_info=True)
        state["errors"].append(f"PDF parsing error: {str(e)}")
        state["is_failed"] = True
        state["current_stage"] = ProcessingStage.FAILED
        state["status_message"] = f"PDF parsing failed: {str(e)}"

    return state


# Node 2: Document Analysis
async def analyze_document_node(state: ProcessingState) -> ProcessingState:
    """
    Analyze document structure and determine complexity.

    Updates state fields:
    - structure
    - metadata
    - should_use_gpt4_extraction
    """
    job_id = state["job_id"]
    logger.info(f"[{job_id}] ========== STAGE 2: DOCUMENT ANALYSIS ==========")

    state["current_stage"] = ProcessingStage.ANALYZING_DOCUMENT
    state["progress_percentage"] = 15.0
    state["status_message"] = "Analyzing document structure and type..."
    state["logs"].append("Starting document analysis")

    if state.get("enable_streaming", True):
        await _publish_status(state)

    try:
        pdf_processor = PDFProcessor()
        document_analyzer = DocumentAnalyzer()

        # Extract document structure
        logger.info(f"[{job_id}] Extracting document structure...")
        structure = pdf_processor.extract_structure(state["pages"])
        state["structure"] = structure

        logger.info(
            f"[{job_id}] Structure extracted: "
            f"{structure.get('has_numbered_sections', False) and 'numbered sections' or 'no numbered sections'}"
        )

        # Analyze document characteristics
        logger.info(f"[{job_id}] Analyzing document characteristics...")
        metadata = await document_analyzer.analyze_document(state["pages"], structure)

        # Update metadata with PDF info
        metadata.total_pages = state["pdf_metadata"]["total_pages"]
        metadata.has_images = state["pdf_metadata"].get("has_images", False)
        metadata.has_tables = state["pdf_metadata"].get("has_tables", False)

        state["metadata"] = metadata

        logger.info(
            f"[{job_id}] Document analysis complete: type={metadata.document_type.value}, "
            f"complexity={metadata.complexity_score:.2f}"
        )

        # Determine model selection
        logger.info(f"[{job_id}] ========== MODEL SELECTION ==========")
        use_gpt4_extraction = document_analyzer.should_use_gpt4(metadata, state["use_gpt4"])
        state["should_use_gpt4_extraction"] = use_gpt4_extraction

        logger.info(
            f"[{job_id}] Model selection - Extraction: {'GPT-4' if use_gpt4_extraction else 'GPT-4o-mini'}, "
            f"Trees: GPT-4"
        )
        state["logs"].append(
            f"Document analyzed: type={metadata.document_type.value}, "
            f"complexity={metadata.complexity_score:.2f}, "
            f"model={'GPT-4' if use_gpt4_extraction else 'GPT-4o-mini'}"
        )

    except Exception as e:
        logger.error(f"[{job_id}] Document analysis failed: {e}", exc_info=True)
        state["errors"].append(f"Document analysis error: {str(e)}")
        state["is_failed"] = True
        state["current_stage"] = ProcessingStage.FAILED
        state["status_message"] = f"Document analysis failed: {str(e)}"

    return state


# Node 3: Chunking
async def chunk_document_node(state: ProcessingState) -> ProcessingState:
    """
    Create intelligent chunks with LLM-assisted policy boundary detection.

    Updates state fields:
    - chunks
    - chunk_summary
    """
    job_id = state["job_id"]
    logger.info(f"[{job_id}] ========== STAGE 3: INTELLIGENT CHUNKING ==========")

    state["current_stage"] = ProcessingStage.CHUNKING
    state["progress_percentage"] = 25.0
    state["status_message"] = "Creating intelligent chunks with LLM-assisted policy boundary detection..."
    state["logs"].append("Starting intelligent chunking")

    if state.get("enable_streaming", True):
        await _publish_status(state)

    try:
        logger.info(f"[{job_id}] Initializing LLM-assisted chunking strategy...")
        chunking_strategy = ChunkingStrategy()

        logger.info(f"[{job_id}] Starting document chunking with policy boundary detection...")
        chunks = await chunking_strategy.chunk_document(state["pages"], state["structure"])

        chunk_summary = chunking_strategy.get_chunk_summary(chunks)

        state["chunks"] = chunks
        state["chunk_summary"] = chunk_summary

        logger.info(f"[{job_id}] Chunking complete: {chunk_summary['total_chunks']} chunks created")
        logger.info(
            f"[{job_id}] Chunk statistics - Total tokens: {chunk_summary['total_tokens']:,}, "
            f"Avg: {chunk_summary['avg_tokens_per_chunk']:.0f}, "
            f"Min: {chunk_summary['min_tokens']}, Max: {chunk_summary['max_tokens']}"
        )
        state["logs"].append(
            f"Chunking complete: {chunk_summary['total_chunks']} chunks, "
            f"{chunk_summary['total_tokens']:,} tokens"
        )

    except Exception as e:
        logger.error(f"[{job_id}] Chunking failed: {e}", exc_info=True)
        state["errors"].append(f"Chunking error: {str(e)}")
        state["is_failed"] = True
        state["current_stage"] = ProcessingStage.FAILED
        state["status_message"] = f"Chunking failed: {str(e)}"

    return state


# Node 4: Policy Extraction
async def extract_policies_node(state: ProcessingState) -> ProcessingState:
    """
    Extract policies and conditions from chunks using LLM.

    Updates state fields:
    - policy_hierarchy
    """
    job_id = state["job_id"]
    logger.info(f"[{job_id}] ========== STAGE 4: POLICY EXTRACTION ==========")

    state["current_stage"] = ProcessingStage.EXTRACTING_POLICIES
    state["progress_percentage"] = 40.0
    state["status_message"] = "Extracting policies and conditions..."
    state["logs"].append("Starting policy extraction")

    if state.get("enable_streaming", True):
        await _publish_status(state)

    try:
        use_gpt4 = state.get("should_use_gpt4_extraction", False)
        logger.info(f"[{job_id}] Initializing policy extractor (model: {'gpt-4o' if use_gpt4 else 'gpt-4o-mini'})...")

        policy_extractor = PolicyExtractor(use_gpt4=use_gpt4)

        logger.info(f"[{job_id}] Starting policy extraction from {len(state['chunks'])} chunks...")
        policy_hierarchy = await policy_extractor.extract_policies(state["chunks"], state["pages"])

        state["policy_hierarchy"] = policy_hierarchy

        logger.info(f"[{job_id}] Policy extraction complete: {policy_hierarchy.total_policies} policies extracted")
        logger.info(
            f"[{job_id}] Hierarchy details - Root policies: {len(policy_hierarchy.root_policies)}, "
            f"Max depth: {policy_hierarchy.max_depth}, "
            f"Definitions: {len(policy_hierarchy.definitions)}"
        )
        state["logs"].append(
            f"Policy extraction complete: {policy_hierarchy.total_policies} policies, "
            f"depth={policy_hierarchy.max_depth}"
        )

    except Exception as e:
        logger.error(f"[{job_id}] Policy extraction failed: {e}", exc_info=True)
        state["errors"].append(f"Policy extraction error: {str(e)}")
        state["is_failed"] = True
        state["current_stage"] = ProcessingStage.FAILED
        state["status_message"] = f"Policy extraction failed: {str(e)}"

    return state


# Node 5: Decision Tree Generation
async def generate_trees_node(state: ProcessingState) -> ProcessingState:
    """
    Generate decision trees for policies using GPT-4.

    Updates state fields:
    - decision_trees
    """
    job_id = state["job_id"]
    logger.info(f"[{job_id}] ========== STAGE 5: DECISION TREE GENERATION ==========")

    state["current_stage"] = ProcessingStage.GENERATING_TREES
    state["progress_percentage"] = 60.0
    state["status_message"] = "Generating decision trees..."
    state["logs"].append("Starting decision tree generation")

    if state.get("enable_streaming", True):
        await _publish_status(state)

    try:
        logger.info(f"[{job_id}] Initializing decision tree generator (model: gpt-4o)...")
        tree_generator = DecisionTreeGenerator(use_gpt4=True)  # Always use GPT-4 for trees

        logger.info(
            f"[{job_id}] Starting hierarchical tree generation for "
            f"{state['policy_hierarchy'].total_policies} policies..."
        )

        decision_trees = await tree_generator.generate_hierarchical_trees(state["policy_hierarchy"])
        state["decision_trees"] = decision_trees

        logger.info(f"[{job_id}] Tree generation complete: {len(decision_trees)} decision trees created")

        # Log tree statistics
        total_nodes = sum(t.total_nodes for t in decision_trees)
        total_paths = sum(t.total_paths for t in decision_trees)
        avg_confidence = sum(t.confidence_score for t in decision_trees) / len(decision_trees) if decision_trees else 0

        logger.info(
            f"[{job_id}] Tree statistics - Total nodes: {total_nodes}, "
            f"Total paths: {total_paths}, "
            f"Avg confidence: {avg_confidence:.2f}"
        )
        state["logs"].append(
            f"Tree generation complete: {len(decision_trees)} trees, "
            f"{total_nodes} nodes, avg confidence={avg_confidence:.2f}"
        )

    except Exception as e:
        logger.error(f"[{job_id}] Tree generation failed: {e}", exc_info=True)
        state["errors"].append(f"Tree generation error: {str(e)}")
        state["is_failed"] = True
        state["current_stage"] = ProcessingStage.FAILED
        state["status_message"] = f"Tree generation failed: {str(e)}"

    return state


# Node 6: Validation
async def validate_node(state: ProcessingState) -> ProcessingState:
    """
    Validate extracted policies and decision trees.

    Updates state fields:
    - validation_result
    - validation_passed
    - needs_retry
    - failed_policy_ids
    """
    job_id = state["job_id"]
    logger.info(f"[{job_id}] ========== STAGE 6: VALIDATION ==========")

    state["current_stage"] = ProcessingStage.VALIDATING
    state["progress_percentage"] = 85.0
    state["status_message"] = "Validating results..."
    state["logs"].append("Starting validation")

    if state.get("enable_streaming", True):
        await _publish_status(state)

    try:
        logger.info(f"[{job_id}] Initializing validator...")
        validator = Validator(use_gpt4=True)  # Always use GPT-4 for validation

        # Get all text for validation
        logger.info(f"[{job_id}] Preparing validation data...")
        all_text = "\n\n".join([p.text for p in state["pages"]])

        logger.info(f"[{job_id}] Starting comprehensive validation...")
        validation_result = await validator.validate_all(
            state["policy_hierarchy"],
            state["decision_trees"],
            all_text
        )

        state["validation_result"] = validation_result
        state["validation_passed"] = validation_result.is_valid

        logger.info(
            f"[{job_id}] Validation complete - Valid: {validation_result.is_valid}, "
            f"Confidence: {validation_result.overall_confidence:.2f}, "
            f"Issues: {len(validation_result.issues)}"
        )
        state["logs"].append(
            f"Validation complete: {'PASSED' if validation_result.is_valid else 'FAILED'}, "
            f"confidence={validation_result.overall_confidence:.2f}"
        )

        # Check if retry is needed
        if not validation_result.is_valid and validation_result.sections_requiring_gpt4:
            state["needs_retry"] = True

            # Identify failed policy IDs
            failed_policy_ids = set()
            logger.info(f"[{job_id}] Identifying failed policies from validation results...")

            for section_name in validation_result.sections_requiring_gpt4:
                if section_name.startswith("Decision Tree:"):
                    policy_title = section_name.replace("Decision Tree: ", "").strip()
                    for policy in _get_all_policies(state["policy_hierarchy"].root_policies):
                        if policy.title == policy_title:
                            failed_policy_ids.add(policy.policy_id)
                            logger.debug(f"[{job_id}] Identified failed policy: {policy.policy_id} - {policy.title}")
                            break

            state["failed_policy_ids"] = list(failed_policy_ids)
            logger.warning(
                f"[{job_id}] Validation failed. Will retry {len(failed_policy_ids)} policies"
            )
        else:
            state["needs_retry"] = False

    except Exception as e:
        logger.error(f"[{job_id}] Validation failed: {e}", exc_info=True)
        state["errors"].append(f"Validation error: {str(e)}")
        state["is_failed"] = True
        state["current_stage"] = ProcessingStage.FAILED
        state["status_message"] = f"Validation failed: {str(e)}"

    return state


# Node 7: Retry Failed Trees
async def retry_failed_trees_node(state: ProcessingState) -> ProcessingState:
    """
    Retry generation of failed decision trees.

    Updates state fields:
    - decision_trees (replaces failed ones)
    - retry_count
    """
    job_id = state["job_id"]
    logger.warning(f"[{job_id}] ========== RETRY: LOW-CONFIDENCE TREES DETECTED ==========")

    state["current_stage"] = ProcessingStage.GENERATING_TREES
    state["progress_percentage"] = 70.0
    state["status_message"] = f"Retrying {len(state['failed_policy_ids'])} failed trees..."
    state["retry_count"] += 1
    state["logs"].append(f"Retrying {len(state['failed_policy_ids'])} failed trees (attempt {state['retry_count']})")

    if state.get("enable_streaming", True):
        await _publish_status(state)

    try:
        failed_policy_ids = set(state["failed_policy_ids"])

        # Get failed policies
        all_policies = _get_all_policies(state["policy_hierarchy"].root_policies)
        failed_policies = [p for p in all_policies if p.policy_id in failed_policy_ids]

        logger.info(f"[{job_id}] Retrying {len(failed_policies)} failed policies with enhanced prompts...")

        # Retry with GPT-4
        tree_generator = DecisionTreeGenerator(use_gpt4=True)

        logger.info(f"[{job_id}] Preparing {len(failed_policies)} policies for retry...")
        retry_tasks = [tree_generator.generate_tree_for_policy(policy) for policy in failed_policies]

        logger.info(f"[{job_id}] Executing {len(retry_tasks)} retry tasks in parallel...")
        retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)

        # Replace failed trees with retried ones
        logger.info(f"[{job_id}] Processing retry results...")
        tree_map = {t.policy_id: t for t in state["decision_trees"]}
        successful_retries = 0
        failed_retries = 0

        for policy, retry_tree in zip(failed_policies, retry_results):
            if isinstance(retry_tree, Exception):
                logger.error(f"[{job_id}] Retry failed for policy {policy.policy_id}: {retry_tree}")
                failed_retries += 1
            elif retry_tree:
                tree_map[policy.policy_id] = retry_tree
                logger.info(f"[{job_id}] Successfully retried tree for policy {policy.policy_id}")
                successful_retries += 1
            else:
                logger.warning(f"[{job_id}] Retry returned None for policy {policy.policy_id}")
                failed_retries += 1

        state["decision_trees"] = list(tree_map.values())

        logger.info(f"[{job_id}] Retry complete - Successful: {successful_retries}, Failed: {failed_retries}")
        state["logs"].append(f"Retry complete: {successful_retries} successful, {failed_retries} failed")

        # Reset needs_retry flag - will be re-evaluated in next validation
        state["needs_retry"] = False

    except Exception as e:
        logger.error(f"[{job_id}] Tree retry failed: {e}", exc_info=True)
        state["errors"].append(f"Tree retry error: {str(e)}")
        state["needs_retry"] = False  # Don't retry again

    return state


# Node 8: Complete Processing
async def complete_node(state: ProcessingState) -> ProcessingState:
    """
    Finalize processing and store results.

    Updates state fields:
    - processing_stats
    - processing_time_seconds
    - current_stage
    """
    job_id = state["job_id"]
    logger.info(f"[{job_id}] ========== PROCESSING COMPLETE ==========")

    state["current_stage"] = ProcessingStage.COMPLETED
    state["progress_percentage"] = 100.0
    state["status_message"] = "Processing complete!"
    state["logs"].append("Processing completed successfully")

    if state.get("enable_streaming", True):
        await _publish_status(state)

    try:
        # Calculate processing stats
        chunk_summary = state["chunk_summary"]
        policy_hierarchy = state["policy_hierarchy"]
        decision_trees = state["decision_trees"]
        validation_result = state["validation_result"]

        processing_stats = {
            "total_chunks": chunk_summary["total_chunks"],
            "total_tokens": chunk_summary["total_tokens"],
            "total_policies": policy_hierarchy.total_policies,
            "total_decision_trees": len(decision_trees),
            "used_gpt4_extraction": state.get("should_use_gpt4_extraction", False),
            "used_gpt4_trees": True,
            "retry_count": state["retry_count"],
        }

        state["processing_stats"] = processing_stats

        # Store result in Redis
        logger.info(f"[{job_id}] Storing results in Redis...")
        redis_client = get_redis_client()

        result_data = {
            "job_id": job_id,
            "status": ProcessingStage.COMPLETED.value,
            "metadata": state["metadata"].model_dump() if state["metadata"] else None,
            "policy_hierarchy": policy_hierarchy.model_dump() if policy_hierarchy else None,
            "decision_trees": [t.model_dump() for t in decision_trees] if decision_trees else [],
            "validation_result": validation_result.model_dump() if validation_result else None,
            "processing_stats": processing_stats,
        }

        redis_client.set_result(job_id, result_data)

        logger.info(f"[{job_id}] ========================================")
        logger.info(f"[{job_id}] FINAL SUMMARY:")
        logger.info(f"[{job_id}]   Policies Extracted: {policy_hierarchy.total_policies}")
        logger.info(f"[{job_id}]   Decision Trees: {len(decision_trees)}")
        logger.info(
            f"[{job_id}]   Validation: {'PASSED' if validation_result.is_valid else 'FAILED'} "
            f"({validation_result.overall_confidence:.2f} confidence)"
        )
        logger.info(
            f"[{job_id}]   Model Usage: Extraction="
            f"{'GPT-4' if state.get('should_use_gpt4_extraction') else 'GPT-4o-mini'}, Trees=GPT-4"
        )
        logger.info(f"[{job_id}] ========================================")

    except Exception as e:
        logger.error(f"[{job_id}] Failed to complete processing: {e}", exc_info=True)
        state["errors"].append(f"Completion error: {str(e)}")

    return state


# Helper function
def _get_all_policies(policies: list) -> list:
    """
    Get all policies recursively (flattened).

    Args:
        policies: List of SubPolicy objects

    Returns:
        Flattened list of all policies
    """
    result = []
    for policy in policies:
        result.append(policy)
        if policy.children:
            result.extend(_get_all_policies(policy.children))
    return result


# Conditional edge functions for routing

def should_retry(state: ProcessingState) -> str:
    """
    Determine if we should retry failed trees or proceed to completion.

    Returns:
        "retry" if retry needed and retry count < max
        "complete" otherwise
    """
    if state.get("is_failed", False):
        return "complete"  # Don't retry if there's a failure

    if state.get("needs_retry", False) and state.get("retry_count", 0) < 1:
        return "retry"

    return "complete"


def check_for_errors(state: ProcessingState) -> str:
    """
    Check if processing failed at any stage.

    Returns:
        "error" if failed
        "continue" otherwise
    """
    if state.get("is_failed", False):
        return "error"
    return "continue"
