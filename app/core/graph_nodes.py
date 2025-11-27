"""
LangGraph Node Implementations for Policy Document Processing.

Each node represents a stage in the processing pipeline and updates the state.
State updates are automatically streamed by LangGraph's astream() mechanism.
"""
import asyncio
import base64

from app.core.pdf_processor import PDFProcessor
from app.core.document_analyzer import DocumentAnalyzer
from app.core.chunking_strategy import ChunkingStrategy
from app.core.policy_extractor import PolicyExtractor
from app.core.decision_tree_generator import DecisionTreeGenerator
from app.core.validator import Validator
from app.core.graph_state import ProcessingState
from app.models.schemas import ProcessingStage
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def parse_pdf_node(state: ProcessingState) -> ProcessingState:
    """
    Parse PDF document and extract pages.

    Updates state: pdf_bytes, pages, pdf_metadata, structure
    """
    job_id = state["job_id"]
    logger.info(f"[{job_id}] Starting Stage 1: PDF Parsing")

    state["current_stage"] = ProcessingStage.PARSING_PDF
    state["progress_percentage"] = 5.0
    state["status_message"] = "Parsing PDF document"
    state["logs"].append(f"Starting PDF parsing for job {job_id}")

    try:
        pdf_processor = PDFProcessor()
        pages, pdf_metadata = pdf_processor.process_document(state["document_base64"])

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


async def analyze_document_node(state: ProcessingState) -> ProcessingState:
    """
    Analyze document structure and determine complexity.

    Updates state: structure, metadata, should_use_gpt4_extraction
    """
    job_id = state["job_id"]
    logger.info(f"[{job_id}] Starting Stage 2: Document Analysis")

    state["current_stage"] = ProcessingStage.ANALYZING_DOCUMENT
    state["progress_percentage"] = 15.0
    state["status_message"] = "Analyzing document structure and type"
    state["logs"].append("Starting document analysis")

    try:
        pdf_processor = PDFProcessor()
        document_analyzer = DocumentAnalyzer()

        logger.info(f"[{job_id}] Extracting document structure")
        structure = pdf_processor.extract_structure(state["pages"])
        state["structure"] = structure

        logger.info(f"[{job_id}] Analyzing document characteristics")
        metadata = await document_analyzer.analyze_document(state["pages"], structure)

        metadata.total_pages = state["pdf_metadata"]["total_pages"]
        metadata.has_images = state["pdf_metadata"].get("has_images", False)
        metadata.has_tables = state["pdf_metadata"].get("has_tables", False)

        state["metadata"] = metadata

        logger.info(
            f"[{job_id}] Document analysis complete: type={metadata.document_type.value}, "
            f"complexity={metadata.complexity_score:.2f}"
        )

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


async def chunk_document_node(state: ProcessingState) -> ProcessingState:
    """
    Create intelligent chunks with LLM-assisted policy boundary detection.

    Updates state: chunks, chunk_summary
    """
    job_id = state["job_id"]
    logger.info(f"[{job_id}] Starting Stage 3: Intelligent Chunking")

    state["current_stage"] = ProcessingStage.CHUNKING
    state["progress_percentage"] = 25.0
    state["status_message"] = "Creating intelligent chunks with policy boundary detection"
    state["logs"].append("Starting intelligent chunking")

    try:
        logger.info(f"[{job_id}] Initializing LLM-assisted chunking strategy")
        chunking_strategy = ChunkingStrategy()

        logger.info(f"[{job_id}] Starting document chunking with policy boundary detection")
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


async def extract_policies_node(state: ProcessingState) -> ProcessingState:
    """
    Extract policies and conditions from chunks using LLM.

    Updates state: policy_hierarchy
    """
    job_id = state["job_id"]
    logger.info(f"[{job_id}] Starting Stage 4: Policy Extraction")

    state["current_stage"] = ProcessingStage.EXTRACTING_POLICIES
    state["progress_percentage"] = 40.0
    state["status_message"] = "Extracting policies and conditions"
    state["logs"].append("Starting policy extraction")

    try:
        use_gpt4 = state.get("should_use_gpt4_extraction", False)
        logger.info(f"[{job_id}] Initializing policy extractor (model: {'gpt-4o' if use_gpt4 else 'gpt-4o-mini'})")

        policy_extractor = PolicyExtractor(use_gpt4=use_gpt4)

        logger.info(f"[{job_id}] Starting policy extraction from {len(state['chunks'])} chunks")
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


async def generate_trees_node(state: ProcessingState) -> ProcessingState:
    """
    Generate decision trees for policies using GPT-4.

    Updates state: decision_trees
    """
    job_id = state["job_id"]
    logger.info(f"[{job_id}] Starting Stage 5: Decision Tree Generation")

    state["current_stage"] = ProcessingStage.GENERATING_TREES
    state["progress_percentage"] = 60.0
    state["status_message"] = "Generating decision trees"
    state["logs"].append("Starting decision tree generation")

    try:
        logger.info(f"[{job_id}] Initializing decision tree generator (model: gpt-4o)")
        tree_generator = DecisionTreeGenerator(use_gpt4=True)

        logger.info(
            f"[{job_id}] Starting hierarchical tree generation for "
            f"{state['policy_hierarchy'].total_policies} policies"
        )

        decision_trees = await tree_generator.generate_hierarchical_trees(state["policy_hierarchy"])
        state["decision_trees"] = decision_trees

        logger.info(f"[{job_id}] Tree generation complete: {len(decision_trees)} decision trees created")

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


async def validate_node(state: ProcessingState) -> ProcessingState:
    """
    Validate extracted policies and decision trees.

    Updates state: validation_result, validation_passed, needs_retry, failed_policy_ids
    """
    job_id = state["job_id"]
    logger.info(f"[{job_id}] Starting Stage 6: Validation")

    state["current_stage"] = ProcessingStage.VALIDATING
    state["progress_percentage"] = 85.0
    state["status_message"] = "Validating results"
    state["logs"].append("Starting validation")

    try:
        logger.info(f"[{job_id}] Initializing validator")
        validator = Validator(use_gpt4=True)

        logger.info(f"[{job_id}] Preparing validation data")
        all_text = "\n\n".join([p.text for p in state["pages"]])

        logger.info(f"[{job_id}] Starting comprehensive validation")
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

            failed_policy_ids = set()
            logger.info(f"[{job_id}] Identifying failed policies from validation results")

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


async def retry_failed_trees_node(state: ProcessingState) -> ProcessingState:
    """
    Retry generation of failed decision trees.

    Updates state: decision_trees, retry_count
    """
    job_id = state["job_id"]
    logger.warning(f"[{job_id}] Retrying low-confidence trees")

    state["current_stage"] = ProcessingStage.GENERATING_TREES
    state["progress_percentage"] = 70.0
    state["status_message"] = f"Retrying {len(state['failed_policy_ids'])} failed trees"
    state["retry_count"] += 1
    state["logs"].append(f"Retrying {len(state['failed_policy_ids'])} failed trees (attempt {state['retry_count']})")

    try:
        failed_policy_ids = set(state["failed_policy_ids"])

        all_policies = _get_all_policies(state["policy_hierarchy"].root_policies)
        failed_policies = [p for p in all_policies if p.policy_id in failed_policy_ids]

        logger.info(f"[{job_id}] Retrying {len(failed_policies)} failed policies with enhanced prompts")

        tree_generator = DecisionTreeGenerator(use_gpt4=True)

        logger.info(f"[{job_id}] Preparing {len(failed_policies)} policies for retry")
        retry_tasks = [tree_generator.generate_tree_for_policy(policy) for policy in failed_policies]

        logger.info(f"[{job_id}] Executing {len(retry_tasks)} retry tasks in parallel")
        retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)

        logger.info(f"[{job_id}] Processing retry results")
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

        state["needs_retry"] = False

    except Exception as e:
        logger.error(f"[{job_id}] Tree retry failed: {e}", exc_info=True)
        state["errors"].append(f"Tree retry error: {str(e)}")
        state["needs_retry"] = False

    return state


async def complete_node(state: ProcessingState) -> ProcessingState:
    """
    Finalize processing and prepare results.

    Updates state: processing_stats, current_stage
    """
    job_id = state["job_id"]
    logger.info(f"[{job_id}] Processing complete")

    state["current_stage"] = ProcessingStage.COMPLETED
    state["progress_percentage"] = 100.0
    state["status_message"] = "Processing complete"
    state["logs"].append("Processing completed successfully")

    try:
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

        logger.info(f"[{job_id}] Final summary:")
        logger.info(f"[{job_id}]   Policies: {policy_hierarchy.total_policies}")
        logger.info(f"[{job_id}]   Decision Trees: {len(decision_trees)}")
        logger.info(
            f"[{job_id}]   Validation: {'PASSED' if validation_result.is_valid else 'FAILED'} "
            f"({validation_result.overall_confidence:.2f} confidence)"
        )
        logger.info(
            f"[{job_id}]   Model Usage: Extraction="
            f"{'GPT-4' if state.get('should_use_gpt4_extraction') else 'GPT-4o-mini'}, Trees=GPT-4"
        )

    except Exception as e:
        logger.error(f"[{job_id}] Failed to complete processing: {e}", exc_info=True)
        state["errors"].append(f"Completion error: {str(e)}")

    return state


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


def should_retry(state: ProcessingState) -> str:
    """
    Determine if we should retry failed trees or proceed to completion.

    Returns:
        "retry" if retry needed and retry count < max
        "complete" otherwise
    """
    if state.get("is_failed", False):
        return "complete"

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
