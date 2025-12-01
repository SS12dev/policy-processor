"""
LangGraph Node Implementations for Policy Document Processing.

Each node represents a stage in the processing pipeline and updates the state.
State updates are automatically streamed by LangGraph's astream() mechanism.
"""
import asyncio
from typing import Union, Dict, Any, List

from app.core.pdf_processor_enhanced import EnhancedPDFProcessor
from app.core.document_analyzer_enhanced import EnhancedDocumentAnalyzer
from app.core.chunking_strategy_enhanced import EnhancedChunkingStrategy
from app.core.policy_extractor import PolicyExtractor
from app.core.decision_tree_generator import DecisionTreeGenerator
from app.core.validator import Validator
from app.core.document_verifier import DocumentVerifier
from app.core.policy_refiner import PolicyRefiner
from app.core.graph_state import ProcessingState
from app.models.schemas import ProcessingStage, EnhancedPDFMetadata, PolicyHierarchy, SubPolicy
from app.utils.logger import get_logger
from app.utils.llm import get_llm

logger = get_logger(__name__)


def flatten_policies(policy: SubPolicy) -> List[SubPolicy]:
    """
    Recursively flatten a policy hierarchy into a list of all policies.
    
    Args:
        policy: Root policy to flatten
        
    Returns:
        List of all policies including the root and all descendants
    """
    result = [policy]
    for child in policy.children:
        result.extend(flatten_policies(child))
    return result


def _ensure_policy_hierarchy(policy_hierarchy: Union[PolicyHierarchy, Dict]) -> PolicyHierarchy:
    """
    Ensure policy_hierarchy is a PolicyHierarchy object.
    
    Args:
        policy_hierarchy: PolicyHierarchy object or dict
        
    Returns:
        PolicyHierarchy object
    """
    if isinstance(policy_hierarchy, dict):
        return PolicyHierarchy(**policy_hierarchy)
    return policy_hierarchy


async def parse_pdf_node(state: ProcessingState) -> ProcessingState:
    """
    Parse PDF document and extract pages with enhanced metadata.
    
    Uses EnhancedPDFProcessor for:
    - Multi-strategy text extraction
    - Intelligent OCR with parallel processing
    - Image and table extraction
    - Heading detection and structure analysis
    - TOC parsing
    - Comprehensive provenance tracking

    Updates state: pages, pdf_metadata
    Note: pdf_bytes is NOT stored in state (security best practice)
    """
    job_id = state["job_id"]
    logger.info(f"[{job_id}] Starting Stage 1: PDF Parsing")

    state["current_stage"] = ProcessingStage.PARSING_PDF
    state["progress_percentage"] = 5.0
    state["status_message"] = "Parsing PDF document with enhanced extraction"
    state["logs"].append(f"Starting enhanced PDF parsing for job {job_id}")

    try:
        # Initialize enhanced PDF processor
        pdf_processor = EnhancedPDFProcessor(
            ocr_dpi=300,
            ocr_language='eng',
            ocr_psm_mode=3,
            max_ocr_workers=4,
            thumbnail_size=(200, 200)
        )
        
        # Process document - pdf_bytes stays in local scope only
        pages, pdf_metadata = pdf_processor.process_document(state["document_base64"])

        # Store results in state (no pdf_bytes!)
        state["pages"] = pages
        state["pdf_metadata"] = pdf_metadata.dict()

        logger.info(
            f"[{job_id}] PDF parsing complete: {pdf_metadata.total_pages} pages, "
            f"{pdf_metadata.total_tables} tables, {pdf_metadata.total_images} images, "
            f"{pdf_metadata.scanned_pages_count} OCR pages, "
            f"{len(pdf_metadata.headings)} headings detected"
        )
        state["logs"].append(
            f"PDF parsed: {pdf_metadata.total_pages} pages, "
            f"{pdf_metadata.total_tables} tables, {pdf_metadata.total_images} images, "
            f"{pdf_metadata.scanned_pages_count} OCR pages"
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
    Analyze document structure with enhanced page-level intelligence.
    
    Uses EnhancedDocumentAnalyzer for:
    - Per-page content classification (policy, admin, definitions, etc.)
    - Policy boundary detection (where policies start/end)
    - Content zone mapping (main policy pages, reference pages, etc.)
    - Semantic continuity tracking (text flow across pages)
    - Policy flow graph (track policies across document)
    - Reference detection and tracking
    
    Provides rich metadata for intelligent chunking and filtering.

    Updates state: metadata, enhanced_document_metadata, should_use_gpt4_extraction
    """
    job_id = state["job_id"]
    logger.info(f"[{job_id}] Starting Stage 2: Enhanced Document Analysis")

    state["current_stage"] = ProcessingStage.ANALYZING_DOCUMENT
    state["progress_percentage"] = 15.0
    state["status_message"] = "Analyzing document with page-level intelligence"
    state["logs"].append("Starting enhanced document analysis with content classification")

    try:
        # Initialize enhanced document analyzer
        document_analyzer = EnhancedDocumentAnalyzer()

        # Get PDF metadata
        pdf_metadata_dict = state["pdf_metadata"]
        pdf_metadata = EnhancedPDFMetadata(**pdf_metadata_dict)

        logger.info(
            f"[{job_id}] Running enhanced analysis: {pdf_metadata.total_pages} pages, "
            f"{len(pdf_metadata.headings)} headings, {len(pdf_metadata.section_boundaries)} sections"
        )

        # Initialize LLM for advanced classification
        llm = get_llm(use_gpt4=False)
        
        # Perform enhanced analysis
        enhanced_metadata = await document_analyzer.analyze_document(
            state["pages"],
            pdf_metadata,
            llm=llm
        )

        # Store enhanced metadata
        state["enhanced_document_metadata"] = enhanced_metadata.dict()

        # Also maintain backward compatibility with old metadata format
        state["metadata"] = {
            "document_type": enhanced_metadata.document_type,
            "total_pages": enhanced_metadata.total_pages,
            "complexity_score": enhanced_metadata.complexity_score,
            "has_images": enhanced_metadata.has_images,
            "has_tables": enhanced_metadata.has_tables,
            "structure_type": enhanced_metadata.structure_type,
            "language": enhanced_metadata.language,
            "processing_time_seconds": enhanced_metadata.processing_time_seconds
        }

        # Build structure info for backward compatibility
        state["structure"] = {
            "has_numbered_sections": len(pdf_metadata.headings) > 0,
            "has_hierarchy": len(pdf_metadata.section_boundaries) > 0,
            "section_pattern": None,
            "has_toc": pdf_metadata.has_toc,
            "has_index": False,
            "has_appendices": False,
            "headings_count": len(pdf_metadata.headings),
            "sections_count": len(pdf_metadata.section_boundaries),
            "toc_entries_count": len(pdf_metadata.toc_entries),
        }

        logger.info(
            f"[{job_id}] Enhanced analysis complete:\n"
            f"  - Document type: {enhanced_metadata.document_type.value}\n"
            f"  - Complexity: {enhanced_metadata.complexity_score:.2f}\n"
            f"  - Structure: {enhanced_metadata.structure_type}\n"
            f"  - Policy pages: {enhanced_metadata.policy_pages_count}/{enhanced_metadata.total_pages}\n"
            f"  - Admin pages: {enhanced_metadata.admin_pages_count}\n"
            f"  - Content zones: {len(enhanced_metadata.content_zones)}\n"
            f"  - Policies detected: {enhanced_metadata.estimated_policy_count}\n"
            f"  - Pages to filter: {len(enhanced_metadata.pages_to_filter)}\n"
            f"  - Extractability: {enhanced_metadata.overall_extractability_score:.2f}"
        )

        # Model selection - use complexity score from enhanced metadata
        # Complexity > 0.7 suggests using GPT-4 for extraction
        complexity_threshold = 0.7
        use_gpt4_extraction = (
            state["use_gpt4"] or 
            enhanced_metadata.complexity_score > complexity_threshold
        )
        state["should_use_gpt4_extraction"] = use_gpt4_extraction

        logger.info(
            f"[{job_id}] Model selection - Extraction: {'GPT-4' if use_gpt4_extraction else 'GPT-4o-mini'}, "
            f"Trees: GPT-4"
        )
        state["logs"].append(
            f"Enhanced analysis: type={enhanced_metadata.document_type.value}, "
            f"policies={enhanced_metadata.estimated_policy_count}, "
            f"zones={len(enhanced_metadata.content_zones)}, "
            f"complexity={enhanced_metadata.complexity_score:.2f}, "
            f"model={'GPT-4' if use_gpt4_extraction else 'GPT-4o-mini'}"
        )

    except Exception as e:
        logger.error(f"[{job_id}] Enhanced document analysis failed: {e}", exc_info=True)
        state["errors"].append(f"Document analysis error: {str(e)}")
        state["is_failed"] = True
        state["current_stage"] = ProcessingStage.FAILED
        state["status_message"] = f"Document analysis failed: {str(e)}"

    return state


async def chunk_document_node(state: ProcessingState) -> ProcessingState:
    """
    Create intelligent chunks with enhanced policy boundary detection.
    
    Uses rich metadata from PDF processor and document analyzer to:
    - Filter out non-policy pages (TOC, bibliography, references, admin)
    - Create policy-boundary-aware chunks (no policy mixing)
    - Preserve semantic continuity across multi-page policies
    - Validate context completeness
    - Detect potential duplicate policies
    
    Updates state: chunks, chunk_summary, chunking_metadata
    """
    job_id = state["job_id"]
    logger.info(f"[{job_id}] Starting Stage 3: Enhanced Intelligent Chunking")

    state["current_stage"] = ProcessingStage.CHUNKING
    state["progress_percentage"] = 25.0
    state["status_message"] = "Creating intelligent policy-aware chunks"
    state["logs"].append("Starting enhanced intelligent chunking")

    try:
        # Check if we have enhanced metadata
        enhanced_metadata = state.get("enhanced_document_metadata")
        pdf_metadata = state.get("pdf_metadata")
        
        if enhanced_metadata and pdf_metadata:
            logger.info(
                f"[{job_id}] Using EnhancedChunkingStrategy with rich metadata "
                f"from PDF processor and document analyzer"
            )
            
            # Initialize enhanced chunking strategy
            llm = get_llm(use_gpt4=False)
            chunking_strategy = EnhancedChunkingStrategy(llm=llm)
            
            # Convert EnhancedPDFPage objects to dicts for chunking
            pages_as_dicts = [
                page.dict() if hasattr(page, 'dict') else page
                for page in state["pages"]
            ]
            
            # Perform enhanced chunking
            logger.info(f"[{job_id}] Performing policy-aware chunking with page filtering...")
            chunking_result = await chunking_strategy.chunk_document(
                pages=pages_as_dicts,
                pdf_metadata=pdf_metadata,
                enhanced_doc_metadata=enhanced_metadata,
            )
            
            # Convert PolicyChunk objects to dicts for state storage
            chunks_as_dicts = []
            for policy_chunk in chunking_result.chunks:
                chunk_dict = {
                    "chunk_id": policy_chunk.chunk_id,
                    "text": policy_chunk.text,
                    "start_page": policy_chunk.start_page,
                    "end_page": policy_chunk.end_page,
                    "token_count": policy_chunk.token_count,
                    "section_context": policy_chunk.metadata.get("heading", ""),
                    "metadata": {
                        "policy_ids": policy_chunk.policy_ids,
                        "content_zones": policy_chunk.content_zones,
                        "has_definitions": policy_chunk.has_definitions,
                        "has_complete_context": policy_chunk.has_complete_context,
                        "continuity_preserved": policy_chunk.continuity_preserved,
                        **policy_chunk.metadata,
                    },
                }
                chunks_as_dicts.append(chunk_dict)
            
            state["chunks"] = chunks_as_dicts
            
            # Store comprehensive chunking metadata
            chunk_summary = chunking_result.statistics
            state["chunk_summary"] = chunk_summary
            
            # Store additional metadata for downstream use
            state["chunking_metadata"] = {
                "filtered_pages": chunking_result.filtered_pages,
                "duplicate_candidates": [
                    {
                        "policy_id_1": dup.policy_id_1,
                        "policy_id_2": dup.policy_id_2,
                        "similarity_score": dup.similarity_score,
                        "chunk_id_1": dup.chunk_id_1,
                        "chunk_id_2": dup.chunk_id_2,
                        "merge_recommendation": dup.merge_recommendation,
                    }
                    for dup in chunking_result.duplicate_candidates
                ],
                "context_validation": chunking_result.context_validation,
                "chunking_method": "enhanced_policy_aware",
            }
            
            logger.info(
                f"[{job_id}] Enhanced chunking complete:\n"
                f"  - Total chunks: {chunk_summary['total_chunks']}\n"
                f"  - Filtered pages: {chunk_summary['filtered_pages_count']}\n"
                f"  - Policy pages: {chunk_summary['policy_pages_count']}\n"
                f"  - Unique policies: {chunk_summary['unique_policies_count']}\n"
                f"  - Duplicate candidates: {chunk_summary['duplicate_candidates_count']}\n"
                f"  - Avg tokens/chunk: {chunk_summary['avg_tokens_per_chunk']:.0f}\n"
                f"  - Complete context chunks: {chunk_summary['chunks_with_complete_context']}/{chunk_summary['total_chunks']}"
            )
            
            state["logs"].append(
                f"Enhanced chunking: {chunk_summary['total_chunks']} chunks, "
                f"{chunk_summary['filtered_pages_count']} pages filtered, "
                f"{chunk_summary['unique_policies_count']} unique policies"
            )
            
        else:
            # Enhanced metadata is required - this shouldn't happen with the current pipeline
            error_msg = (
                "Enhanced metadata not available. Ensure PDF processor and "
                "document analyzer stages complete successfully."
            )
            logger.error(f"[{job_id}] {error_msg}")
            state["logs"].append(f"ERROR: {error_msg}")
            raise ValueError(error_msg)

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

        # Initialize LLM for policy extraction
        llm = get_llm(use_gpt4=use_gpt4)
        policy_extractor = PolicyExtractor(use_gpt4=use_gpt4, llm=llm)

        logger.info(f"[{job_id}] Starting policy extraction from {len(state['chunks'])} chunks")
        
        # Pass enhanced metadata if available
        enhanced_metadata = state.get("enhanced_document_metadata")
        policy_hierarchy = await policy_extractor.extract_policies(
            state["chunks"],
            state["pages"],
            enhanced_metadata=enhanced_metadata
        )

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
        # Initialize LLM for tree generation (always use GPT-4)
        llm = get_llm(use_gpt4=True)
        tree_generator = DecisionTreeGenerator(use_gpt4=True, llm=llm)

        # Ensure policy_hierarchy is an object (not dict)
        policy_hierarchy = _ensure_policy_hierarchy(state["policy_hierarchy"])

        logger.info(
            f"[{job_id}] Starting hierarchical tree generation for "
            f"{policy_hierarchy.total_policies} policies"
        )

        decision_trees = await tree_generator.generate_hierarchical_trees(policy_hierarchy)
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

        # Ensure policy_hierarchy is an object (not dict)
        policy_hierarchy = _ensure_policy_hierarchy(state["policy_hierarchy"])

        logger.info(f"[{job_id}] Preparing validation data")
        all_text = "\n\n".join([p.text for p in state["pages"]])

        logger.info(f"[{job_id}] Starting comprehensive validation")
        validation_result = await validator.validate_all(
            policy_hierarchy,
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
                    for policy in _get_all_policies(policy_hierarchy.root_policies):
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

        # Initialize LLM for retry (use GPT-4 for better quality)
        retry_llm = get_llm(use_gpt4=True)
        tree_generator = DecisionTreeGenerator(use_gpt4=True, llm=retry_llm)

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


async def verification_node(state: ProcessingState) -> ProcessingState:
    """
    Stage 7: Verification & Quality Check
    
    Compare outputs against source document and detect issues:
    - Duplicate policies
    - Missing/over-extracted policies
    - Incomplete decision trees
    - Hierarchy structure issues
    
    Updates state: verification, needs_refinement
    """
    job_id = state["job_id"]
    logger.info(f"[{job_id}] Starting Stage 7: Verification & Quality Check")
    
    state["current_stage"] = ProcessingStage.VALIDATING  # Reuse VALIDATING stage for UI
    state["progress_percentage"] = 92.0
    state["status_message"] = "Verifying quality and completeness..."
    state["logs"].append("Starting verification & quality check")
    
    try:
        # Get policies from hierarchy
        policy_hierarchy = state.get("policy_hierarchy")
        if not policy_hierarchy:
            logger.warning(f"[{job_id}] No policy hierarchy found, skipping verification")
            state["needs_refinement"] = False
            return state
        
        # Flatten policies from hierarchy
        all_policies = []
        for root in policy_hierarchy.root_policies:
            all_policies.extend(flatten_policies(root))
        
        # Initialize verifier
        verifier = DocumentVerifier(
            target_metrics={
                'min_policies': 5,
                'max_policies': 7,
                'min_roots': 3,
                'max_roots': 4,
                'min_depth': 2,
                'max_depth': 3,
                'min_tree_completeness': 0.80,
                'max_validation_issues': 40
            }
        )
        
        # Run verification
        validation_result = state.get("validation_result")
        validation_dict = {}
        if validation_result:
            # Convert ValidationResult object to dict
            validation_dict = {
                'confidence': validation_result.overall_confidence,
                'is_valid': validation_result.is_valid,
                'issues': len(validation_result.issues)
            }
        
        verification_report = verifier.verify_all(
            document={},  # Not used by verifier
            chunks=state["chunks"],
            policies=all_policies,
            trees=state["decision_trees"],
            validation_results=validation_dict
        )
        
        # Store verification results
        state["verification"] = {
            'total_issues': verification_report.total_issues,
            'needs_refinement': verification_report.needs_refinement,
            'confidence': verification_report.confidence,
            'duplicate_count': len(verification_report.duplicate_policies),
            'hierarchy_issues': len(verification_report.hierarchy_issues),
            'tree_issues': len(verification_report.tree_completeness_issues),
            'coverage_issues': len(verification_report.coverage_issues),
            'summary': verification_report.summary
        }
        
        # Store full report for refinement node
        state["verification_report"] = verification_report
        
        # Determine next action
        if verification_report.needs_refinement:
            state["needs_refinement"] = True
            logger.warning(
                f"[{job_id}] Verification found {verification_report.total_issues} issues - refinement needed"
            )
            state["logs"].append(
                f"Verification detected {verification_report.total_issues} issues requiring refinement"
            )
        else:
            state["needs_refinement"] = False
            logger.info(
                f"[{job_id}] Verification passed (confidence: {verification_report.confidence:.1%})"
            )
            state["logs"].append(f"Verification passed with {verification_report.confidence:.1%} confidence")
        
    except Exception as e:
        logger.error(f"[{job_id}] Verification failed: {e}", exc_info=True)
        state["errors"].append(f"Verification error: {str(e)}")
        state["needs_refinement"] = False  # Continue without refinement on error
    
    return state


async def refinement_node(state: ProcessingState) -> ProcessingState:
    """
    Stage 8: Refinement (Conditional)
    
    Automatically fix quality issues detected by verification:
    - Merge duplicate policies
    - Strengthen hierarchy grouping
    - Regenerate affected decision trees
    
    Updates state: policies, decision_trees, refinement_applied
    """
    job_id = state["job_id"]
    logger.info(f"[{job_id}] Starting Stage 8: Refinement")
    
    state["current_stage"] = ProcessingStage.VALIDATING  # Reuse VALIDATING stage
    state["progress_percentage"] = 94.0
    state["status_message"] = "Refining policies and decision trees..."
    state["logs"].append("Starting automated refinement")
    
    try:
        # Get verification report
        verification_report = state.get("verification_report")
        if not verification_report:
            logger.warning(f"[{job_id}] No verification report found, skipping refinement")
            state["needs_refinement"] = False
            return state
        
        # Get policy hierarchy
        policy_hierarchy = state.get("policy_hierarchy")
        if not policy_hierarchy:
            logger.warning(f"[{job_id}] No policy hierarchy found, skipping refinement")
            state["needs_refinement"] = False
            return state
        
        # Flatten policies
        all_policies = []
        for root in policy_hierarchy.root_policies:
            all_policies.extend(flatten_policies(root))
        
        # Initialize refiner
        refiner = PolicyRefiner(
            extractor=state.get("policy_extractor"),
            tree_generator=state.get("tree_generator")
        )
        
        # Run refinement
        refinement_result = refiner.refine(
            policies=all_policies,
            trees=state["decision_trees"],
            verification_report=verification_report
        )
        
        # Update policy hierarchy with refined policies
        # Note: This is simplified - in production you'd rebuild the hierarchy
        state["policy_hierarchy"].root_policies = refinement_result.policies
        state["decision_trees"] = refinement_result.trees
        state["refinement_applied"] = True
        
        # Store refinement summary
        state["refinement"] = {
            'policies_merged': refinement_result.policies_merged,
            'policies_reparented': refinement_result.policies_reparented,
            'trees_regenerated': refinement_result.trees_regenerated,
            'actions_taken': refinement_result.actions_taken,
            'summary': refinement_result.summary
        }
        
        logger.info(
            f"[{job_id}] Refinement complete - "
            f"Merged: {refinement_result.policies_merged}, "
            f"Re-parented: {refinement_result.policies_reparented}, "
            f"Trees regenerated: {refinement_result.trees_regenerated}"
        )
        
        state["logs"].append(
            f"Refinement applied: {len(refinement_result.actions_taken)} actions taken"
        )
        
        # Check if we should re-verify (max 1 iteration)
        refinement_iterations = state.get("refinement_iterations", 0)
        if refinement_iterations < 1:
            state["refinement_iterations"] = refinement_iterations + 1
            state["needs_reverification"] = True
            logger.info(f"[{job_id}] Will re-verify after refinement")
        else:
            state["needs_reverification"] = False
            state["needs_refinement"] = False
            logger.warning(f"[{job_id}] Max refinement iterations reached, proceeding to completion")
        
    except Exception as e:
        logger.error(f"[{job_id}] Refinement failed: {e}", exc_info=True)
        state["errors"].append(f"Refinement error: {str(e)}")
        state["needs_refinement"] = False
        state["needs_reverification"] = False
    
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
        # Check if processing failed - skip stats if so
        if state.get("is_failed", False):
            logger.warning(f"[{job_id}] Processing failed, skipping stats calculation")
            return state
        
        chunk_summary = state.get("chunk_summary")
        policy_hierarchy = state.get("policy_hierarchy")
        decision_trees = state.get("decision_trees")
        validation_result = state.get("validation_result")
        
        # Verify required fields are present
        if not all([chunk_summary, policy_hierarchy, decision_trees, validation_result]):
            logger.warning(f"[{job_id}] Missing required fields for stats, skipping")
            return state

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


def should_refine(state: ProcessingState) -> str:
    """
    Determine if refinement is needed after verification.
    
    Returns:
        "refine" if refinement needed
        "reverify" if re-verification needed after refinement
        "complete" if no refinement needed or max iterations reached
    """
    # Check if re-verification needed (after refinement)
    if state.get("needs_reverification", False):
        return "reverify"
    
    # Check if refinement needed (from verification)
    if state.get("needs_refinement", False):
        return "refine"
    
    # Otherwise, proceed to completion
    return "complete"
