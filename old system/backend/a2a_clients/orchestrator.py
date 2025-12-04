"""Prior Authorization Orchestrator using A2A agents."""

import logging
from typing import Any, Dict, List

try:
    # Try relative imports first (when used as module)
    from .client_factory import ClientFactory
    from ..pdf_processing.chunking import create_unified_chunks, ChunkingStrategy
    from ..settings import backend_settings
except ImportError:
    # Fall back to absolute imports (when run directly)
    from client_factory import ClientFactory
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from pdf_processing.chunking import create_unified_chunks, ChunkingStrategy
    from settings import backend_settings

logger = logging.getLogger(__name__)


class PriorAuthOrchestrator:
    """Orchestrates the complete prior authorization workflow using A2A agents.
    
    Handles PDF processing, database operations, and agent coordination.
    """

    def __init__(self):
        """Initialize orchestrator with A2A clients."""
        print("ORCHESTRATOR_DEBUG: Initializing PriorAuthOrchestrator...")
        logger.info(f"ORCHESTRATOR_DEBUG: Initializing PriorAuthOrchestrator...")
        self.policy_client = ClientFactory.create_policy_client()
        self.application_client = ClientFactory.create_application_client()
        self.decision_client = ClientFactory.create_decision_client()
        
        client_mode = ClientFactory.get_client_mode()
        logger.info(f"PriorAuthOrchestrator initialized with {client_mode} A2A clients")
        logger.info(f"ORCHESTRATOR_DEBUG: Clients created - policy: {type(self.policy_client)}, app: {type(self.application_client)}, decision: {type(self.decision_client)}")
        print("ORCHESTRATOR_DEBUG: Initialization complete")

    async def close(self):
        """Close all client connections."""
        await self.policy_client.close()
        await self.application_client.close()
        await self.decision_client.close()

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all agents.

        Returns:
            Dict with agent health status.
        """
        return {
            "agent_1_policy": await self.policy_client.health_check(),
            "agent_2_application": await self.application_client.health_check(),
            "agent_3_decision": await self.decision_client.health_check()
        }

    async def process_policy(
        self,
        policy_name: str,
        pages_data: List[Dict],
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Enhanced policy processing with semantic chunking and questionnaire consolidation.
        Uses improved chunking strategy for better source traceability.

        Args:
            policy_name: Name of the policy
            pages_data: Page data with text and detailed line references

        Returns:
            Dict with consolidated questionnaire and source references
        """
        logger.info(f"Processing policy '{policy_name}' with {len(pages_data)} pages")
        
        if progress_callback:
            progress_callback(72, "Analyzing document structure and determining processing strategy...")

        # Step 1: Determine processing strategy based on document size
        total_characters = sum(len(page.get("text", "")) for page in pages_data)
        logger.info(f"Policy size: {total_characters} characters across {len(pages_data)} pages")
        
        # MEMORY OPTIMIZATION: Apply chunking more aggressively for consistent memory usage
        # Only process very small documents (< 20k chars AND < 6 pages) as single unit
        if total_characters < backend_settings.SMALL_DOC_CHAR_THRESHOLD and len(pages_data) < backend_settings.SMALL_DOC_PAGE_THRESHOLD:
            logger.info("Step 1: Processing as single document (small/medium size)")
            if progress_callback:
                progress_callback(75, f"Small document detected ({total_characters:,} chars) - processing as single unit...")
            chunks = []  # No chunks needed
        else:
            logger.info("Step 1: Creating larger chunks for efficient processing")
            if progress_callback:
                progress_callback(75, f"Large document detected ({total_characters:,} chars) - creating processing chunks...")
            # MEMORY OPTIMIZATION: Use configurable chunking with unified system
            chunks = create_unified_chunks(pages_data, doc_type="policy")
            logger.info(f"Created {len(chunks)} chunks for processing")
            if progress_callback:
                progress_callback(78, f"Created {len(chunks)} processing chunks for optimal analysis...")
        
        # Step 2: Process based on strategy
        if not chunks:
            # For small policies, process as single unit
            logger.info("Step 2: Processing as single policy (small document)")
            
            if progress_callback:
                progress_callback(80, "Sending document to AI agent for comprehensive analysis...")
            
            combined_text = "\n\n".join([page["text"] for page in pages_data])
            page_references = [page["page_number"] for page in pages_data]
            
            policy_data = {
                "policy_name": policy_name,
                "policy_text": combined_text,
                "page_references": page_references,
                "total_pages": len(pages_data)
            }
            
            if progress_callback:
                progress_callback(85, "Agent is analysing the Policy Document to generate the questionnaire...")
            
            result = await self.policy_client.analyze_policy(policy_data)
            
            if progress_callback:
                progress_callback(92, "AI analysis complete! Processing results...")
            
            if result.get("status") != "completed":
                error_msg = result.get("message", "Failed to analyze policy")
                logger.error(f"Failed to analyze policy '{policy_name}': {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "details": result
                }
            
            analysis_data = result.get("data", {})
            questions = analysis_data.get("questions", [])
            
        else:
            # For larger policies, process chunks and consolidate
            logger.info("Step 2: Processing chunks individually for large document")
            
            if progress_callback:
                progress_callback(80, f"Processing {len(chunks)} chunks sequentially...")
            
            question_chunks = []
            total_cost_data = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_cost": 0.0,
                "model": "Unknown",
                "chunks_processed": 0
            }
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                
                # Calculate progress: 80-92 range for chunk processing
                chunk_progress = 80 + int((i / len(chunks)) * 12)
                estimated_time = (len(chunks) - i) * 45  # ~45 seconds per chunk
                eta_text = f" (ETA: {estimated_time//60}m {estimated_time%60}s)" if estimated_time > 60 else f" (ETA: {estimated_time}s)"
                
                if progress_callback:
                    progress_callback(chunk_progress, f"Processing chunk {i+1}/{len(chunks)} with AI agent{eta_text}")
                
                chunk_data = {
                    "policy_name": f"{policy_name} (Chunk {i+1})",
                    "policy_text": chunk["text"],
                    "page_references": chunk["pages_covered"],
                    "chunk_metadata": {
                        "chunk_id": chunk["chunk_id"],
                        "start_page": chunk["start_page"],
                        "end_page": chunk["end_page"],
                        "line_references": chunk["line_references"]
                    }
                }
                
                chunk_result = await self.policy_client.analyze_policy(chunk_data)
                
                if chunk_result.get("status") == "completed":
                    chunk_analysis = chunk_result.get("data", {})
                    chunk_questions = chunk_analysis.get("questions", [])
                    
                    # Aggregate cost information from this chunk
                    chunk_cost_info = chunk_analysis.get("cost_info", {})
                    if chunk_cost_info:
                        total_cost_data["input_tokens"] += chunk_cost_info.get("input_tokens", 0)
                        total_cost_data["output_tokens"] += chunk_cost_info.get("output_tokens", 0)
                        total_cost_data["total_cost"] += chunk_cost_info.get("total_cost", 0.0)
                        total_cost_data["chunks_processed"] += 1
                        
                        # Update model information (use the latest one, or first non-unknown)
                        chunk_model = chunk_cost_info.get("model", "Unknown")
                        if chunk_model != "Unknown":
                            total_cost_data["model"] = chunk_model
                    
                    if progress_callback:
                        chunk_cost = chunk_cost_info.get("total_cost", 0.0)
                        cost_text = f" (Cost: ${chunk_cost:.4f})" if chunk_cost > 0 else ""
                        progress_callback(chunk_progress + 1, f"Chunk {i+1}/{len(chunks)} complete - generated {len(chunk_questions)} questions{cost_text}")
                    
                    # Add chunk metadata to questions for traceability
                    for question in chunk_questions:
                        question["chunk_id"] = chunk["chunk_id"]
                        question["line_references"] = chunk["line_references"]
                    
                    question_chunks.append({
                        "chunk_id": chunk["chunk_id"],
                        "questions": chunk_questions,
                        "chunk_metadata": chunk_data["chunk_metadata"],
                        "cost_info": chunk_cost_info
                    })
                else:
                    logger.warning(f"Failed to process chunk {i+1}: {chunk_result.get('message')}")
                    if progress_callback:
                        progress_callback(chunk_progress + 1, f"⚠️ Chunk {i+1}/{len(chunks)} failed - continuing with next chunk")
            
            # Step 3: Consolidate questions from all chunks (optimize payload size)
            logger.info("Step 3: Consolidating questions from all chunks")
            
            if progress_callback:
                progress_callback(93, "Consolidating questions from all chunks...")
            
            # Summarize questions to reduce payload size
            summarized_chunks = []
            for chunk in question_chunks:
                summarized_questions = []
                for q in chunk["questions"][:10]:  # Limit to top 10 questions per chunk
                    summarized_questions.append({
                        "question_id": q.get("question_id", ""),
                        "question_text": q.get("question_text", "")[:200],  # Truncate long questions
                        "question_type": q.get("question_type", "text"),
                        "criterion_type": q.get("criterion_type", "general"),
                        "priority": q.get("priority", "medium"),
                        "source_text_snippet": q.get("source_text_snippet", "")[:100],  # Truncate source
                        "page_number": q.get("page_number"),
                        "chunk_id": q.get("chunk_id")
                    })
                
                summarized_chunks.append({
                    "chunk_id": chunk["chunk_id"],
                    "questions": summarized_questions,
                    "chunk_metadata": chunk["chunk_metadata"]
                })
            
            consolidation_data = {
                "policy_name": policy_name,
                "question_chunks": summarized_chunks,
                "target_question_count": 20,
                "total_original_questions": sum(len(chunk["questions"]) for chunk in question_chunks)
            }
            
            if progress_callback:
                total_questions = sum(len(chunk["questions"]) for chunk in question_chunks)
                progress_callback(95, f"Consolidating {total_questions} questions into final questionnaire...")
            
            consolidation_result = await self.policy_client.consolidate_questionnaire(consolidation_data)
            
            # Track consolidation costs
            if consolidation_result.get("status") == "completed":
                consolidation_data_result = consolidation_result.get("data", {})
                consolidation_cost_info = consolidation_data_result.get("cost_info", {})
                
                if consolidation_cost_info:
                    # Add consolidation costs to the total
                    total_cost_data["input_tokens"] += consolidation_cost_info.get("input_tokens", 0)
                    total_cost_data["output_tokens"] += consolidation_cost_info.get("output_tokens", 0)
                    total_cost_data["total_cost"] += consolidation_cost_info.get("total_cost", 0.0)
                    
                    logger.info(f"Consolidation cost: ${consolidation_cost_info.get('total_cost', 0.0):.4f}")
            
            if consolidation_result.get("status") != "completed":
                logger.warning("Consolidation failed, using smart fallback approach")
                # Smart fallback: manually consolidate by selecting best questions from each chunk
                questions = []
                seen_questions = set()
                
                for chunk in question_chunks:
                    chunk_questions = chunk["questions"]
                    # Sort by question quality and take top questions from each chunk
                    sorted_questions = sorted(
                        chunk_questions,
                        key=lambda q: (
                            len(q.get("question_text", "")),  # Prefer longer, more detailed questions
                            q.get("criterion_type", "") in ["eligibility", "medical_necessity"]  # Prioritize key types
                        ),
                        reverse=True
                    )
                    
                    # Add unique questions (avoid duplicates by checking question text similarity)
                    for q in sorted_questions[:4]:  # Top 4 from each chunk
                        question_text = q.get("question_text", "").lower()
                        # Simple duplicate detection
                        is_duplicate = False
                        for seen in seen_questions:
                            if len(set(question_text.split()) & set(seen.split())) > len(question_text.split()) * 0.7:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate and len(questions) < 25:  # Limit to 25 total
                            questions.append(q)
                            seen_questions.add(question_text)
                
                logger.info(f"Smart fallback selected {len(questions)} unique questions")
            else:
                consolidation_data = consolidation_result.get("data", {})
                print(f"ORCHESTRATOR_DEBUG: consolidation_data keys: {list(consolidation_data.keys())}")
                consolidated_questionnaire = consolidation_data.get("consolidated_questionnaire", {})
                print(f"ORCHESTRATOR_DEBUG: consolidated_questionnaire keys: {list(consolidated_questionnaire.keys()) if consolidated_questionnaire else 'None'}")
                
                # Try multiple possible keys for questions
                questions = []
                for key in ['questions', 'merged_questions', 'selected_questions', 'final_questions']:
                    if key in consolidated_questionnaire:
                        potential_questions = consolidated_questionnaire[key]
                        if isinstance(potential_questions, list) and potential_questions:
                            questions = potential_questions
                            print(f"ORCHESTRATOR_DEBUG: Found {len(questions)} questions in '{key}' field")
                            break
                
                if not questions:
                    print(f"ORCHESTRATOR_DEBUG: No questions found in any expected field")
                    # Fallback: try to extract from the entire consolidated_questionnaire if it's a list
                    if isinstance(consolidated_questionnaire, list):
                        questions = consolidated_questionnaire
                        print(f"ORCHESTRATOR_DEBUG: Using consolidated_questionnaire as questions list: {len(questions)} items")
                
                print(f"ORCHESTRATOR_DEBUG: Final extracted questions count: {len(questions)}")
                logger.info(f"Consolidation successful: {len(questions)} questions")
        
        # Extract page references for return
        page_references = [page["page_number"] for page in pages_data]
        
        if progress_callback:
            progress_callback(98, f"Analysis complete! Generated {len(questions)} intelligent questions")
        
        logger.info(f"Successfully analyzed policy '{policy_name}': {len(questions)} questions generated")
        
        # Extract cost information from the analysis result
        cost_info = {}
        if not chunks:  # Single policy processing
            if result and result.get("data"):
                raw_cost_info = result["data"].get("cost_info", {})
                # Normalize cost format to be consistent
                if raw_cost_info:
                    cost_info = {
                        "input_tokens": raw_cost_info.get("total_input_tokens", raw_cost_info.get("input_tokens", 0)),
                        "output_tokens": raw_cost_info.get("total_output_tokens", raw_cost_info.get("output_tokens", 0)),
                        "total_cost": raw_cost_info.get("total_cost", 0.0),
                        "model": raw_cost_info.get("model", "Unknown")
                    }
        else:  # Chunked processing - use aggregated costs from all chunks
            if total_cost_data["chunks_processed"] > 0 or total_cost_data["total_cost"] > 0:
                cost_info = {
                    "input_tokens": total_cost_data["input_tokens"],
                    "output_tokens": total_cost_data["output_tokens"],
                    "total_cost": total_cost_data["total_cost"],
                    "model": total_cost_data["model"],
                    "chunks_processed": total_cost_data["chunks_processed"]
                }
                logger.info(f"Total aggregated costs (chunks + consolidation): "
                           f"${total_cost_data['total_cost']:.4f} "
                           f"({total_cost_data['input_tokens'] + total_cost_data['output_tokens']} tokens)")
            else:
                logger.warning("No cost information available from any processing steps")
                cost_info = {}
        
        return {
            "success": True,
            "questions": questions,
            "total_questions": len(questions),
            "page_references": page_references,
            "questionnaire_metadata": {
                "policy_name": policy_name,
                "processing_method": "chunked" if len(chunks) > 3 else "single",
                "chunks_processed": len(chunks) if len(chunks) > 3 else 1,
                "source_pages": len(pages_data),
                "total_processing_cost": cost_info.get("total_cost", 0.0),
                "total_tokens_used": cost_info.get("input_tokens", 0) + cost_info.get("output_tokens", 0)
            },
            "cost_info": cost_info
        }

    async def process_application(
        self,
        pages_data: List[Dict],
        questions: List[Dict]
    ) -> Dict[str, Any]:
        """
        Process application document using chunked approach like policy processing.
        Backend handles chunking and merging - agent processes individual chunks.

        Args:
            pages_data: Page data with images and text (preprocessed by backend)  
            questions: Questionnaire questions

        Returns:
            Dict with patient info, summary, and answers
        """
        print("ORCHESTRATOR_DEBUG: process_application method called!")
        print(f"Input validation - pages_data: {type(pages_data)}, length: {len(pages_data) if pages_data else 'None'}")
        print(f"Input validation - questions: {type(questions)}, length: {len(questions) if questions else 'None'}")
        logger.info(f"ORCHESTRATOR_DEBUG: ===== PROCESS_APPLICATION START =====")
        logger.info(f"ORCHESTRATOR_DEBUG: pages_data type: {type(pages_data)}, length: {len(pages_data) if pages_data else 'None'}")
        logger.info(f"ORCHESTRATOR_DEBUG: questions type: {type(questions)}, length: {len(questions) if questions else 'None'}")
        
        if not pages_data:
            print("ERROR: No pages_data provided!")
            logger.error(f"ORCHESTRATOR_DEBUG: No pages_data provided!")
            return {"success": False, "error": "No pages provided", "answers": [], "patient_info": {}, "patient_summary": ""}
            
        if not questions:
            print("ERROR: No questions provided!")
            logger.error(f"ORCHESTRATOR_DEBUG: No questions provided!")
            return {"success": False, "error": "No questions provided", "answers": [], "patient_info": {}, "patient_summary": ""}
        
        print(f"Validation passed - processing {len(pages_data)} pages with {len(questions)} questions")
        
        logger.info(f"Processing application with {len(pages_data)} pages and {len(questions)} questions")
        logger.info(f"ORCHESTRATOR_DEBUG: process_application ENTRY")

        # Step 1: Prepare application data and use backend chunking
        print("Step 1: Preparing application data with backend chunking")
        logger.info("Step 1: Preparing application data with backend chunking")
        logger.info(f"ORCHESTRATOR_DEBUG: Input pages sample: {[p.get('page_number') for p in pages_data[:3]]}")
        
        # Ensure all pages have required fields
        processed_pages = []
        for page in pages_data:
            processed_page = {
                "page_number": page.get("page_number", 1),
                "text": page.get("text", page.get("text_content", "")),  # Use "text" for chunking
                "image_base64": page.get("image_base64", page.get("image", ""))
            }
            processed_pages.append(processed_page)
        
        print(f"Processed {len(processed_pages)} pages for chunking")
        logger.info(f"ORCHESTRATOR_DEBUG: Processed {len(processed_pages)} pages for chunking")
        
        # Debug: Check what's actually in the processed pages
        for i, page in enumerate(processed_pages[:3]):  # Show first 3 pages
            text_preview = page.get("text", "NO_TEXT")
            print(f"ORCHESTRATOR_DEBUG: Page {i+1} text preview: {text_preview[:100] if text_preview else 'EMPTY'}...")
            print(f"ORCHESTRATOR_DEBUG: Page {i+1} keys: {list(page.keys())}")
        
        # Use unified chunking system optimized for application processing
        try:
            app_strategy = ChunkingStrategy(
                pages_per_chunk=backend_settings.APP_PAGES_PER_CHUNK,
                overlap_pages=backend_settings.APP_CHUNK_OVERLAP_PAGES
            )
            
            application_chunks = create_unified_chunks(
                processed_pages, 
                strategy=app_strategy,
                doc_type="application"
            )
            print(f"Created {len(application_chunks)} chunks")
            
        except Exception as e:
            print(f"ERROR in chunking: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"Chunking failed: {e}", "answers": [], "patient_info": {}, "patient_summary": ""}
        
        logger.info(f"Backend created {len(application_chunks)} optimized chunks for application processing")

        # Step 2: Process each chunk through Agent 2
        logger.info(f"Step 2: Processing {len(application_chunks)} chunks through Agent 2")
        
        all_answers = {}  # Map question_id -> best_answer
        best_patient_summary = ""
        best_patient_info = {}
        
        # Initialize cost tracking for application processing
        total_cost_data = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_cost": 0.0,
            "model": "Unknown",
            "chunks_processed": 0
        }
        
        for chunk_idx, chunk in enumerate(application_chunks):
            # Reconstruct page data from chunk
            start_page = chunk.get("start_page", 1)
            end_page = chunk.get("end_page", 1)
            pages_covered = chunk.get("pages_covered", [start_page])
            
            # Extract pages for this chunk
            chunk_pages = []
            for page_num in pages_covered:
                for page in processed_pages:
                    if page.get("page_number") == page_num:
                        chunk_pages.append(page)
                        break
            
            chunk_type = chunk.get("chunk_type", "unknown")
            logger.info(f"ORCHESTRATOR_DEBUG: Processing {chunk_type} chunk {chunk_idx + 1}/{len(application_chunks)} with {len(chunk_pages)} pages ({start_page}-{end_page})")
            
            application_data = {
                "pages_data": chunk_pages,
                "questions": questions
            }
            
            chunk_result = await self.application_client.process_application(application_data)
            
            if chunk_result.get("status") != "completed":
                logger.warning(f"Chunk {chunk_idx + 1} failed: {chunk_result.get('message', 'Unknown error')}")
                continue
            
            # Merge results from this chunk
            chunk_data = chunk_result.get("data", {})
            chunk_answers = chunk_data.get("answers", [])
            chunk_summary = chunk_data.get("patient_summary", "")
            chunk_patient_info = chunk_data.get("patient_info", {})
            
            # Aggregate cost information from this chunk
            chunk_cost_info = chunk_data.get("cost_info", {})
            if chunk_cost_info:
                total_cost_data["input_tokens"] += chunk_cost_info.get("input_tokens", chunk_cost_info.get("total_input_tokens", 0))
                total_cost_data["output_tokens"] += chunk_cost_info.get("output_tokens", chunk_cost_info.get("total_output_tokens", 0))
                total_cost_data["total_cost"] += chunk_cost_info.get("total_cost", 0.0)
                total_cost_data["chunks_processed"] += 1
                
                # Update model information (use the latest one, or first non-unknown)
                chunk_model = chunk_cost_info.get("model", "Unknown")
                if chunk_model != "Unknown":
                    total_cost_data["model"] = chunk_model
                
                logger.info(f"Chunk {chunk_idx + 1} cost: ${chunk_cost_info.get('total_cost', 0.0):.4f}")
            
            # Merge answers (keep best answer for each question)
            chunk_found = 0
            for answer in chunk_answers:
                question_id = str(answer.get("question_id", ""))
                answer_text = answer.get("answer_text", "").strip()
                
                if not question_id:
                    continue
                
                if answer_text != "NOT_FOUND":
                    chunk_found += 1
                
                existing_answer = all_answers.get(question_id)
                
                if not existing_answer:
                    # First answer for this question
                    all_answers[question_id] = answer
                elif answer_text != "NOT_FOUND" and existing_answer.get("answer_text", "").strip() == "NOT_FOUND":
                    # This answer is found but existing was not found - replace
                    all_answers[question_id] = answer
                elif answer_text != "NOT_FOUND" and existing_answer.get("answer_text", "").strip() != "NOT_FOUND":
                    # Both are found - keep the one with more detail
                    if len(answer_text) > len(existing_answer.get("answer_text", "")):
                        all_answers[question_id] = answer
            
            logger.info(f"Chunk {chunk_idx + 1} contributed {chunk_found} new answers")
            
            # Use the most complete patient summary
            if chunk_summary and (not best_patient_summary or len(chunk_summary) > len(best_patient_summary)):
                best_patient_summary = chunk_summary
            
            # Merge patient info (prioritize actual found values over NOT_FOUND)
            for key, value in chunk_patient_info.items():
                if value and value not in ["Not Found", "NOT_FOUND"]:
                    # This chunk has a real value - use it
                    best_patient_info[key] = value
                elif not best_patient_info.get(key) or best_patient_info[key] in ["Not Found", "NOT_FOUND"]:
                    # No existing value or existing value is NOT_FOUND - take this one
                    best_patient_info[key] = value

        # Step 3: Compile final results
        final_answers = list(all_answers.values())
        found_answers = [a for a in final_answers if a.get('answer_text', '').strip() != 'NOT_FOUND']
        not_found_answers = [a for a in final_answers if a.get('answer_text', '').strip() == 'NOT_FOUND']
        
        logger.info(f"Successfully processed application: {len(found_answers)} answers found, {len(not_found_answers)} not found from {len(processed_pages)} pages")
        
        # Prepare cost information for return
        cost_info = {}
        if total_cost_data["total_cost"] > 0:
            cost_info = {
                "input_tokens": total_cost_data["input_tokens"],
                "output_tokens": total_cost_data["output_tokens"],
                "total_cost": total_cost_data["total_cost"],
                "model": total_cost_data["model"],
                "chunks_processed": total_cost_data["chunks_processed"]
            }
            logger.info(f"Total application processing costs: "
                       f"${total_cost_data['total_cost']:.4f} "
                       f"({total_cost_data['input_tokens'] + total_cost_data['output_tokens']} tokens)")
        else:
            logger.warning("No cost information available from application processing")
        
        return {
            "success": True,
            "patient_info": best_patient_info,
            "patient_summary": best_patient_summary,
            "answers": final_answers,
            "total_questions": len(questions),
            "answered_questions": len(found_answers),
            "analysis_summary": {
                "pages_processed": len(processed_pages),
                "chunks_processed": len(application_chunks),
                "questions_answered": len(found_answers),
                "questions_not_found": len(not_found_answers),
                "summary_length": len(best_patient_summary)
            },
            "cost_info": cost_info
        }

    async def make_decision(
        self,
        policy_name: str,
        answers: List[Dict],
        patient_context: Dict[str, Any] = None,
        reference_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate authorization decision based on answers.

        Args:
            policy_name: Policy name
            answers: Extracted answers
            patient_context: Patient details (name, DOB, BMI, etc.)
            reference_context: Reference data (current_date, policy_type, etc.)

        Returns:
            Dict with evaluation and recommendation
        """
        logger.info(f"Making decision for {len(answers)} answers")

        # Prepare reference context with current date and system info
        from datetime import datetime
        if reference_context is None:
            reference_context = {}
        
        reference_context.update({
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "system_version": "1.0",
            "evaluation_mode": "comprehensive"
        })

        # Prepare patient context if not provided
        if patient_context is None:
            patient_context = {}

        all_evaluations = []
        batch_size = 15
        
        # Initialize cost tracking for decision making
        total_cost_data = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_cost": 0.0,
            "model": "Unknown",
            "batches_processed": 0
        }

        for batch_start in range(0, len(answers), batch_size):
            batch_end = min(batch_start + batch_size, len(answers))
            answer_batch = answers[batch_start:batch_end]
            batch_number = batch_start // batch_size + 1

            eval_result = await self.decision_client.evaluate_answers(
                answer_batch, policy_name, batch_number
            )

            if eval_result.get("status") == "completed":
                batch_data = eval_result.get("data", {})
                evaluations = batch_data.get("evaluations", [])
                all_evaluations.extend(evaluations)
                
                # Aggregate cost information from this batch
                batch_cost_info = batch_data.get("cost_info", {})
                if batch_cost_info:
                    total_cost_data["input_tokens"] += batch_cost_info.get("input_tokens", batch_cost_info.get("total_input_tokens", 0))
                    total_cost_data["output_tokens"] += batch_cost_info.get("output_tokens", batch_cost_info.get("total_output_tokens", 0))
                    total_cost_data["total_cost"] += batch_cost_info.get("total_cost", 0.0)
                    total_cost_data["batches_processed"] += 1
                    
                    # Update model information
                    batch_model = batch_cost_info.get("model", "Unknown")
                    if batch_model != "Unknown":
                        total_cost_data["model"] = batch_model
                    
                    logger.info(f"Batch {batch_number} evaluation cost: ${batch_cost_info.get('total_cost', 0.0):.4f}")

        # First, let's properly classify the evaluation results based on the original answers
        requirements_met = 0
        requirements_not_met = 0
        unclear_requirements = 0
        missing_information = 0
        critical_issues = []
        
        logger.info(f"Starting evaluation classification for {len(all_evaluations)} evaluations")
        
        for i, evaluation in enumerate(all_evaluations):
            # Get the original answer to check if it was actually found
            original_answer = answers[i] if i < len(answers) else {}
            original_answer_text = original_answer.get('answer_text', '').strip()
            
            meets_req = evaluation.get("meets_requirement", "Missing")
            question_text = original_answer.get('question_text', f'Question {i+1}')
            
            logger.info(f"Q{i+1}: {question_text[:50]}... | Answer: {original_answer_text[:30]}... | Meets: {meets_req}")
            
            # CRITICAL DEBUG: Log the exact evaluation object
            logger.info(f"  -> Full evaluation object: {evaluation}")
            
            # If original answer was "NOT_FOUND", this is missing documentation
            if original_answer_text == "NOT_FOUND" or original_answer_text == "Not Found":
                missing_information += 1
                logger.info(f"  -> Classified as MISSING INFO")
            elif meets_req == "Yes":
                requirements_met += 1
                logger.info(f"  -> Classified as MET")
            elif meets_req == "No":
                requirements_not_met += 1
                logger.info(f"  -> Classified as NOT MET")
            else:  # Partial, Unclear, etc.
                unclear_requirements += 1
                logger.info(f"  -> Classified as UNCLEAR ({meets_req})")
                logger.warning(f"  -> UNEXPECTED meets_requirement value: '{meets_req}' (type: {type(meets_req)})")
            
            # Check for critical findings
            if evaluation.get("critical_finding", False) or evaluation.get("risk_assessment") == "High":
                critical_issues.append(evaluation.get("clinical_rationale", ""))
        
        logger.info(f"Final counts: Met={requirements_met}, Not Met={requirements_not_met}, Missing={missing_information}, Unclear={unclear_requirements}")
        
        # Log a few sample evaluations for debugging
        if len(all_evaluations) > 0:
            logger.info("Sample evaluations:")
            for i in range(min(3, len(all_evaluations))):
                eval_sample = all_evaluations[i]
                logger.info(f"  Eval {i}: meets_requirement={eval_sample.get('meets_requirement', 'MISSING_FIELD')}, rationale={eval_sample.get('clinical_rationale', 'No rationale')[:50]}...")

        # Adjust risk assessment based on actual found vs missing data
        if critical_issues or requirements_not_met > 3:  # Only high risk if truly critical issues
            overall_risk = "High"
        elif missing_information > 5 or unclear_requirements > 3:
            overall_risk = "Medium" 
        else:
            overall_risk = "Low"

        # Prepare found answers with clinical details for detailed reasoning
        found_answers = []
        for i, answer in enumerate(answers):
            answer_text = answer.get('answer_text', '').strip()
            if answer_text and answer_text not in ['NOT_FOUND', 'Not Found']:
                evaluation = all_evaluations[i] if i < len(all_evaluations) else {}
                found_answers.append({
                    "question_text": answer.get('question_text', ''),
                    "answer_text": answer_text,
                    "clinical_rationale": evaluation.get('clinical_rationale', ''),
                    "page_references": answer.get('source_page_number', ''),
                    "meets_requirement": evaluation.get('meets_requirement', 'Unknown')
                })

        recommendation_result = await self.decision_client.generate_recommendation(
            policy_name=policy_name,
            total_questions=len(answers),
            requirements_met=requirements_met,
            requirements_not_met=requirements_not_met,
            unclear_requirements=unclear_requirements,
            missing_information=missing_information,
            overall_risk_level=overall_risk,
            critical_issues=critical_issues,
            found_answers=found_answers,
            patient_context=patient_context,
            reference_context=reference_context
        )

        if recommendation_result.get("status") != "completed":
            return {
                "success": False,
                "error": "Failed to generate recommendation",
                "details": recommendation_result
            }

        recommendation_data = recommendation_result.get("data", {})
        recommendation = recommendation_data.get("recommendation", {})
        
        # Aggregate cost from final recommendation generation
        final_cost_info = recommendation_data.get("cost_info", {})
        if final_cost_info:
            total_cost_data["input_tokens"] += final_cost_info.get("input_tokens", final_cost_info.get("total_input_tokens", 0))
            total_cost_data["output_tokens"] += final_cost_info.get("output_tokens", final_cost_info.get("total_output_tokens", 0))
            total_cost_data["total_cost"] += final_cost_info.get("total_cost", 0.0)
            
            logger.info(f"Final recommendation generation cost: ${final_cost_info.get('total_cost', 0.0):.4f}")

        # Prepare final cost information
        cost_info = {}
        if total_cost_data["total_cost"] > 0:
            cost_info = {
                "input_tokens": total_cost_data["input_tokens"],
                "output_tokens": total_cost_data["output_tokens"],
                "total_cost": total_cost_data["total_cost"],
                "model": total_cost_data["model"],
                "batches_processed": total_cost_data["batches_processed"]
            }
            logger.info(f"Total decision making costs: "
                       f"${total_cost_data['total_cost']:.4f} "
                       f"({total_cost_data['input_tokens'] + total_cost_data['output_tokens']} tokens)")
        else:
            logger.warning("No cost information available from decision making")

        return {
            "success": True,
            "evaluations": all_evaluations,
            "recommendation": recommendation,
            "summary": {
                "total_questions": len(answers),
                "requirements_met": requirements_met,
                "requirements_not_met": requirements_not_met,
                "unclear_requirements": unclear_requirements,
                "missing_information": missing_information,
                "overall_risk_level": overall_risk
            },
            "cost_info": cost_info
        }