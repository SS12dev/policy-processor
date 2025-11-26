"""
Main orchestrator that coordinates the entire document processing pipeline.
"""
import asyncio
import uuid
from datetime import datetime
from typing import Optional, AsyncIterator
from app.utils.logger import get_logger
from app.utils.redis_client import get_redis_client
from app.core.pdf_processor import PDFProcessor
from app.core.document_analyzer import DocumentAnalyzer
from app.core.chunking_strategy import ChunkingStrategy
from app.core.policy_extractor import PolicyExtractor
from app.core.decision_tree_generator import DecisionTreeGenerator
from app.core.validator import Validator
from app.models.schemas import (
    ProcessingRequest,
    ProcessingResponse,
    ProcessingStatus,
    ProcessingStage,
    DocumentMetadata,
)

logger = get_logger(__name__)


class ProcessingOrchestrator:
    """Orchestrates the entire document processing pipeline."""

    def __init__(self):
        """Initialize orchestrator."""
        self.redis_client = get_redis_client()
        self.pdf_processor = PDFProcessor()
        self.document_analyzer = DocumentAnalyzer()

    async def process_document(
        self, request: ProcessingRequest, job_id: Optional[str] = None
    ) -> ProcessingResponse:
        """
        Process a policy document through the complete pipeline.

        Args:
            request: ProcessingRequest object
            job_id: Optional job ID (generated if not provided)

        Returns:
            ProcessingResponse object
        """
        # Generate job ID if not provided
        if not job_id:
            job_id = str(uuid.uuid4())

        logger.info(f"Starting document processing for job {job_id}")

        start_time = datetime.utcnow()

        try:
            # Stage 1: Parse PDF
            logger.info(f"[{job_id}] ========== STAGE 1: PDF PARSING ==========")
            await self._update_status(
                job_id,
                ProcessingStage.PARSING_PDF,
                0,
                "Parsing PDF document...",
            )

            logger.info(f"[{job_id}] Starting PDF extraction...")
            pages, pdf_metadata = self.pdf_processor.process_document(request.document)
            logger.info(f"[{job_id}] PDF parsing complete: {pdf_metadata['total_pages']} pages, {len([p for p in pages if p.tables])} pages with tables")

            # Stage 2: Analyze document
            logger.info(f"[{job_id}] ========== STAGE 2: DOCUMENT ANALYSIS ==========")
            await self._update_status(
                job_id,
                ProcessingStage.ANALYZING_DOCUMENT,
                10,
                "Analyzing document structure and type...",
            )

            logger.info(f"[{job_id}] Extracting document structure...")
            structure = self.pdf_processor.extract_structure(pages)
            logger.info(f"[{job_id}] Structure extracted: {structure.get('has_numbered_sections', False) and 'numbered sections' or 'no numbered sections'}")

            logger.info(f"[{job_id}] Analyzing document characteristics...")
            metadata = await self.document_analyzer.analyze_document(pages, structure)
            logger.info(f"[{job_id}] Document analysis complete: type={metadata.document_type.value}, complexity={metadata.complexity_score:.2f}")

            # Update metadata with PDF info
            metadata.total_pages = pdf_metadata["total_pages"]
            metadata.has_images = pdf_metadata.get("has_images", False)
            metadata.has_tables = pdf_metadata.get("has_tables", False)

            # Determine if we should use GPT-4
            logger.info(f"[{job_id}] ========== MODEL SELECTION ==========")
            # For tree generation, always use GPT-4 as it produces significantly better results
            use_gpt4_extraction = self.document_analyzer.should_use_gpt4(
                metadata, request.processing_options.use_gpt4
            )
            use_gpt4_trees = True  # Always use GPT-4 for decision tree generation

            logger.info(f"[{job_id}] Model selection - Extraction: {'GPT-4' if use_gpt4_extraction else 'GPT-4o-mini'}, Trees: GPT-4")
            if use_gpt4_extraction:
                logger.info(f"[{job_id}] Using GPT-4 for extraction due to document complexity ({metadata.complexity_score:.2f})")
            else:
                logger.info(f"[{job_id}] Using GPT-4o-mini for extraction (complexity {metadata.complexity_score:.2f} below threshold)")

            # Stage 3: Chunking with LLM-assisted policy boundary detection
            logger.info(f"[{job_id}] ========== STAGE 3: INTELLIGENT CHUNKING ==========")
            await self._update_status(
                job_id,
                ProcessingStage.CHUNKING,
                20,
                "Creating intelligent chunks with LLM-assisted policy boundary detection...",
            )

            logger.info(f"[{job_id}] Initializing LLM-assisted chunking strategy...")
            chunking_strategy = ChunkingStrategy()

            logger.info(f"[{job_id}] Starting document chunking with policy boundary detection...")
            chunks = await chunking_strategy.chunk_document(pages, structure)

            chunk_summary = chunking_strategy.get_chunk_summary(chunks)
            logger.info(
                f"[{job_id}] Chunking complete: {chunk_summary['total_chunks']} chunks created"
            )
            logger.info(
                f"[{job_id}] Chunk statistics - Total tokens: {chunk_summary['total_tokens']:,}, "
                f"Avg: {chunk_summary['avg_tokens_per_chunk']:.0f}, "
                f"Min: {chunk_summary['min_tokens']}, Max: {chunk_summary['max_tokens']}"
            )

            # Stage 4: Extract policies
            logger.info(f"[{job_id}] ========== STAGE 4: POLICY EXTRACTION ==========")
            await self._update_status(
                job_id,
                ProcessingStage.EXTRACTING_POLICIES,
                30,
                "Extracting policies and conditions...",
            )

            logger.info(f"[{job_id}] Initializing policy extractor (model: {'gpt-4o' if use_gpt4_extraction else 'gpt-4o-mini'})...")
            policy_extractor = PolicyExtractor(use_gpt4=use_gpt4_extraction)

            logger.info(f"[{job_id}] Starting policy extraction from {len(chunks)} chunks...")
            policy_hierarchy = await policy_extractor.extract_policies(chunks, pages)

            logger.info(
                f"[{job_id}] Policy extraction complete: {policy_hierarchy.total_policies} policies extracted"
            )
            logger.info(
                f"[{job_id}] Hierarchy details - Root policies: {len(policy_hierarchy.root_policies)}, "
                f"Max depth: {policy_hierarchy.max_depth}, "
                f"Definitions: {len(policy_hierarchy.definitions)}"
            )

            # Stage 5: Generate decision trees
            logger.info(f"[{job_id}] ========== STAGE 5: DECISION TREE GENERATION ==========")
            await self._update_status(
                job_id,
                ProcessingStage.GENERATING_TREES,
                60,
                "Generating decision trees...",
            )

            logger.info(f"[{job_id}] Initializing decision tree generator (model: gpt-4o)...")
            tree_generator = DecisionTreeGenerator(use_gpt4=use_gpt4_trees)

            # Generate hierarchical decision trees with contextual awareness
            logger.info(f"[{job_id}] Starting hierarchical tree generation for {policy_hierarchy.total_policies} policies...")

            # Generate trees with hierarchy context
            decision_trees = await tree_generator.generate_hierarchical_trees(policy_hierarchy)

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

            # Stage 6: Validation
            logger.info(f"[{job_id}] ========== STAGE 6: VALIDATION ==========")
            await self._update_status(
                job_id,
                ProcessingStage.VALIDATING,
                85,
                "Validating results...",
            )

            logger.info(f"[{job_id}] Initializing validator...")
            validator = Validator(use_gpt4=use_gpt4_trees)

            # Get all text for validation
            logger.info(f"[{job_id}] Preparing validation data...")
            all_text = "\n\n".join([p.text for p in pages])

            logger.info(f"[{job_id}] Starting comprehensive validation...")
            validation_result = await validator.validate_all(
                policy_hierarchy, decision_trees, all_text
            )

            logger.info(
                f"[{job_id}] Validation complete - Valid: {validation_result.is_valid}, "
                f"Confidence: {validation_result.overall_confidence:.2f}, "
                f"Issues: {len(validation_result.issues)}"
            )

            # If validation failed with low confidence, retry failed trees
            if not validation_result.is_valid and validation_result.sections_requiring_gpt4:
                logger.warning(
                    f"[{job_id}] ========== RETRY: LOW-CONFIDENCE TREES DETECTED =========="
                )
                logger.warning(
                    f"[{job_id}] Validation failed with confidence {validation_result.overall_confidence:.2f}. "
                    f"Retrying {len(validation_result.sections_requiring_gpt4)} low-confidence sections"
                )

                await self._update_status(
                    job_id,
                    ProcessingStage.GENERATING_TREES,
                    70,
                    f"Retrying {len(validation_result.sections_requiring_gpt4)} failed trees...",
                )

                # Retry failed trees
                failed_policy_ids = set()
                logger.info(f"[{job_id}] Identifying failed policies from validation results...")
                for section_name in validation_result.sections_requiring_gpt4:
                    # Extract policy ID from section name like "Decision Tree: Policy Name"
                    if section_name.startswith("Decision Tree:"):
                        # Find the policy by title
                        policy_title = section_name.replace("Decision Tree: ", "").strip()
                        for policy in self._get_all_policies(policy_hierarchy.root_policies):
                            if policy.title == policy_title:
                                failed_policy_ids.add(policy.policy_id)
                                logger.debug(f"[{job_id}] Identified failed policy: {policy.policy_id} - {policy.title}")
                                break

                if failed_policy_ids:
                    logger.info(f"[{job_id}] Retrying {len(failed_policy_ids)} failed policies with enhanced prompts...")

                    # Get failed policies
                    failed_policies = [
                        p for p in self._get_all_policies(policy_hierarchy.root_policies)
                        if p.policy_id in failed_policy_ids
                    ]

                    # Retry with same tree generator (already using GPT-4)
                    import asyncio
                    logger.info(f"[{job_id}] Preparing {len(failed_policies)} policies for retry...")
                    retry_tasks = []
                    for policy in failed_policies:
                        logger.debug(f"[{job_id}] Queueing retry for policy: {policy.policy_id}")
                        retry_tasks.append(tree_generator.generate_tree_for_policy(policy))

                    logger.info(f"[{job_id}] Executing {len(retry_tasks)} retry tasks in parallel...")
                    retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)

                    # Replace failed trees with retried ones
                    logger.info(f"[{job_id}] Processing retry results...")
                    tree_map = {t.policy_id: t for t in decision_trees}
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

                    decision_trees = list(tree_map.values())

                    logger.info(f"[{job_id}] Retry complete - Successful: {successful_retries}, Failed: {failed_retries}")
                    logger.info(f"[{job_id}] Re-validating with retried trees...")
                    await self._update_status(
                        job_id,
                        ProcessingStage.VALIDATING,
                        85,
                        "Re-validating after retry...",
                    )

                    # Re-validate
                    validation_result = await validator.validate_all(
                        policy_hierarchy, decision_trees, all_text
                    )

                    logger.info(
                        f"[{job_id}] Re-validation complete - Valid: {validation_result.is_valid}, "
                        f"Confidence: {validation_result.overall_confidence:.2f}, "
                        f"Issues: {len(validation_result.issues)}"
                    )

            # Calculate final processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            metadata.processing_time_seconds = processing_time

            # Stage 7: Complete
            logger.info(f"[{job_id}] ========== PROCESSING COMPLETE ==========")
            await self._update_status(
                job_id,
                ProcessingStage.COMPLETED,
                100,
                "Processing complete!",
            )

            # Build response
            logger.info(f"[{job_id}] Building final response...")
            response = ProcessingResponse(
                job_id=job_id,
                status=ProcessingStage.COMPLETED,
                metadata=metadata,
                policy_hierarchy=policy_hierarchy,
                decision_trees=decision_trees,
                validation_result=validation_result,
                processing_stats={
                    "total_chunks": chunk_summary["total_chunks"],
                    "total_tokens": chunk_summary["total_tokens"],
                    "total_policies": policy_hierarchy.total_policies,
                    "total_decision_trees": len(decision_trees),
                    "used_gpt4_extraction": use_gpt4_extraction,
                    "used_gpt4_trees": use_gpt4_trees,
                    "processing_time_seconds": processing_time,
                },
            )

            # Store result in Redis
            logger.info(f"[{job_id}] Storing results in Redis...")
            self.redis_client.set_result(job_id, response.model_dump())

            logger.info(f"[{job_id}] ========================================")
            logger.info(f"[{job_id}] FINAL SUMMARY:")
            logger.info(f"[{job_id}]   Processing Time: {processing_time:.2f}s")
            logger.info(f"[{job_id}]   Policies Extracted: {policy_hierarchy.total_policies}")
            logger.info(f"[{job_id}]   Decision Trees: {len(decision_trees)}")
            logger.info(f"[{job_id}]   Validation: {'PASSED' if validation_result.is_valid else 'FAILED'} ({validation_result.overall_confidence:.2f} confidence)")
            logger.info(f"[{job_id}]   Model Usage: Extraction={('GPT-4' if use_gpt4_extraction else 'GPT-4o-mini')}, Trees=GPT-4")
            logger.info(f"[{job_id}] ========================================")

            return response

        except Exception as e:
            logger.error(f"Error processing document for job {job_id}: {e}", exc_info=True)

            await self._update_status(
                job_id,
                ProcessingStage.FAILED,
                0,
                f"Processing failed: {str(e)}",
                [str(e)],
            )

            raise

    async def _update_status(
        self,
        job_id: str,
        stage: ProcessingStage,
        progress: float,
        message: str,
        errors: list = None,
    ):
        """
        Update processing status.

        Args:
            job_id: Job identifier
            stage: Current processing stage
            progress: Progress percentage (0-100)
            message: Status message
            errors: List of error messages
        """
        status = ProcessingStatus(
            job_id=job_id,
            stage=stage,
            progress_percentage=progress,
            message=message,
            errors=errors or [],
        )

        # Store in Redis
        self.redis_client.set_status(job_id, status.model_dump())

        # Publish to channel for streaming
        self.redis_client.publish(f"job:{job_id}:status", status.model_dump())

        logger.info(f"Job {job_id}: {stage.value} - {progress}% - {message}")

    async def get_status(self, job_id: str) -> Optional[ProcessingStatus]:
        """
        Get current processing status.

        Args:
            job_id: Job identifier

        Returns:
            ProcessingStatus object or None
        """
        status_data = self.redis_client.get_status(job_id)

        if status_data:
            return ProcessingStatus(**status_data)

        return None

    async def get_result(self, job_id: str) -> Optional[ProcessingResponse]:
        """
        Get processing result.

        Args:
            job_id: Job identifier

        Returns:
            ProcessingResponse object or None
        """
        result_data = self.redis_client.get_result(job_id)

        if result_data:
            return ProcessingResponse(**result_data)

        return None

    def _get_all_policies(self, policies: list) -> list:
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
                result.extend(self._get_all_policies(policy.children))
        return result

    async def stream_status(self, job_id: str) -> AsyncIterator[ProcessingStatus]:
        """
        Stream processing status updates.

        Args:
            job_id: Job identifier

        Yields:
            ProcessingStatus objects
        """
        import json
        import redis

        # Subscribe to Redis channel for job updates
        pubsub = self.redis_client._client.pubsub()
        channel = f"job:{job_id}:status"

        try:
            pubsub.subscribe(channel)

            # First, send current status if available
            current_status = await self.get_status(job_id)
            if current_status:
                yield current_status

            # Then stream updates
            while True:
                try:
                    message = pubsub.get_message(timeout=1.0)

                    if message and message["type"] == "message":
                        status_data = json.loads(message["data"])
                        status = ProcessingStatus(**status_data)

                        yield status

                        # Stop streaming if completed or failed
                        if status.stage in [ProcessingStage.COMPLETED, ProcessingStage.FAILED]:
                            break

                    # If no message, check if job is still active
                    elif message is None:
                        # Check current status to see if job finished while we were waiting
                        current_status = await self.get_status(job_id)
                        if current_status and current_status.stage in [ProcessingStage.COMPLETED, ProcessingStage.FAILED]:
                            yield current_status
                            break

                        # Small delay to avoid tight loop
                        await asyncio.sleep(0.1)

                except redis.TimeoutError:
                    # Redis timeout - check if job is still active
                    logger.debug(f"Redis timeout for job {job_id}, checking status...")
                    current_status = await self.get_status(job_id)
                    if current_status:
                        if current_status.stage in [ProcessingStage.COMPLETED, ProcessingStage.FAILED]:
                            yield current_status
                            break
                    # Continue listening
                    await asyncio.sleep(0.1)
                    continue

        finally:
            pubsub.unsubscribe(channel)
            pubsub.close()
