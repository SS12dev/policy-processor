"""Policy Management Component."""

import asyncio
import json
import sys
from pathlib import Path

import streamlit as st

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.pdf_processing.text_extraction import validate_pdf_content
from backend.pdf_processing.chunking import create_unified_chunks, ChunkingStrategy
from backend.pdf_processing.image_conversion import convert_pdf_to_images
from backend.settings import backend_settings
from backend.database.db_crud import (
    create_file_storage, create_policy, create_questionnaire, create_question,
    get_all_policies, get_questionnaires_by_policy, get_questions_by_questionnaire,
    approve_questionnaire
)
from frontend.utils.database import get_db_for_streamlit, close_db
from frontend.utils.session import get_orchestrator


def show_policy_management():
    """Display policy management page."""
    st.header("Policy Management")

    tab1, tab2, tab3 = st.tabs(["Upload Policy", "View Policies", "Review Questionnaire"])

    with tab1:
        st.subheader("Upload New Policy")

        policy_name = st.text_input("Policy Name", placeholder="e.g., Spinal Fusion Prior Auth")
        policy_file = st.file_uploader("Upload Policy PDF", type=['pdf'])

        if st.button("Process Policy", disabled=not (policy_name and policy_file)):
            try:
                # Create initial status containers (will be managed by ProgressTracker later)
                initial_progress = st.progress(0)
                initial_status = st.empty()
                
                # Read PDF
                initial_status.text("Reading PDF file...")
                initial_progress.progress(10)
                pdf_bytes = policy_file.read()

                # Validate PDF
                initial_status.text("Validating PDF content...")
                initial_progress.progress(20)
                validation = validate_pdf_content(pdf_bytes)
                if not validation['valid']:
                    initial_status.text("")
                    initial_progress.empty()
                    st.error(f"Invalid PDF: {validation['error']}")
                    return

                # Extract text with enhanced line references
                initial_status.text("Extracting text from policy...")
                initial_progress.progress(30)
                from backend.pdf_processing.text_extraction import extract_text_with_line_references
                pages_text = extract_text_with_line_references(pdf_bytes)

                # Create larger chunks for efficient processing
                initial_status.text("Creating processing chunks...")
                initial_progress.progress(40)
                # Use unified chunking system with custom strategy for frontend processing
                frontend_strategy = ChunkingStrategy(
                    max_chars_per_chunk=15000,
                    char_overlap=1000
                )
                chunks = create_unified_chunks(pages_text, strategy=frontend_strategy, doc_type="policy")

                # Convert to images for multimodal processing
                initial_status.text("Converting pages to images...")
                initial_progress.progress(50)
                try:
                    images = convert_pdf_to_images(
                        pdf_bytes,
                        poppler_path=backend_settings.POPPLER_PATH
                    )
                except Exception as e:
                    images = []

                # Combine text and images with enhanced metadata
                initial_status.text("Preparing data for analysis...")
                initial_progress.progress(60)
                pages_data = []
                for i, page_data in enumerate(pages_text):
                    enhanced_page_data = {
                        "page_number": page_data['page_number'],
                        "text": page_data['text'],
                        "line_count": page_data['line_count'],
                        "line_data": page_data['line_data']
                    }
                    if i < len(images):
                        enhanced_page_data['image'] = images[i]
                    pages_data.append(enhanced_page_data)

                # Determine processing strategy
                total_chars = sum(len(page.get("text", "")) for page in pages_data)
                # Clear initial progress indicators
                initial_progress.empty()
                initial_status.empty()
                
                # Process with simple, clean timing
                from frontend.utils.timing import SimpleTiming, create_simple_progress_callback
                orch = get_orchestrator()
                
                # Create simple timing tracker
                timing = SimpleTiming()
                timing.start("Analyzing policy document with AI agent...")
                
                # Create progress callback
                callback = create_simple_progress_callback(timing)
                
                result = asyncio.run(orch.process_policy(policy_name, pages_data, progress_callback=callback))

                if result['success']:
                    timing.update(90, "Saving results to database...")
                    
                    # Save to database
                    db = get_db_for_streamlit()
                    try:
                        # Store file
                        file_storage = create_file_storage(
                            db,
                            filename=policy_file.name,
                            file_data=pdf_bytes,
                            file_type='application/pdf'
                        )

                        # Store policy
                        policy = create_policy(
                            db,
                            name=policy_name,
                            file_storage_id=file_storage.id,
                            status='processed'
                        )

                        # Store enhanced medical questionnaire (unapproved) with cost tracking
                        questionnaire_metadata = result.get('questionnaire_metadata', {})
                        cost_info = result.get('cost_info', {})
                        
                        # Debug: Print the result structure to understand what's available
                        print(f"POLICY_RESULT_DEBUG: Available keys in result: {list(result.keys())}")
                        if 'cost_info' in result:
                            print(f"COST_INFO_DEBUG: {result['cost_info']}")
                        if 'questionnaire' in result:
                            questionnaire_data = result['questionnaire']
                            if isinstance(questionnaire_data, dict) and 'cost_info' in questionnaire_data:
                                print(f"QUESTIONNAIRE_COST_DEBUG: {questionnaire_data['cost_info']}")
                                cost_info = questionnaire_data['cost_info']
                        questionnaire = create_questionnaire(
                            db,
                            policy_id=policy.id,
                            approved=False,
                            name=questionnaire_metadata.get('policy_name', policy_name),
                            total_questions=questionnaire_metadata.get('total_questions', len(result['questions'])),
                            complexity_level=questionnaire_metadata.get('complexity_level'),
                            coverage_areas=questionnaire_metadata.get('coverage_areas'),
                            procedure_type=questionnaire_metadata.get('procedure_type'),
                            target_population=questionnaire_metadata.get('target_population'),
                            llm_input_tokens=cost_info.get('input_tokens', cost_info.get('total_input_tokens', 0)),
                            llm_output_tokens=cost_info.get('output_tokens', cost_info.get('total_output_tokens', 0)),
                            llm_total_cost=cost_info.get('total_cost', 0.0),
                            llm_model=cost_info.get('model', 'Unknown')
                        )

                        # Store enhanced medical questions with full metadata
                        for i, q in enumerate(result['questions']):
                            create_question(
                                db,
                                questionnaire_id=questionnaire.id,
                                question_id=q.get('question_id', f"q_{i+1}"),
                                question_text=q['question_text'],
                                question_type=q.get('question_type', 'text'),
                                answer_options=q.get('answer_options', {}),
                                source_text_snippet=q.get('source_text_snippet', ''),
                                page_number=q.get('page_references', q.get('page_number')),
                                line_reference=q.get('line_reference'),
                                criterion_type=q.get('criterion_type'),
                                priority_level=q.get('priority_level'),
                                approval_impact=q.get('approval_impact'),
                                validation_rules=q.get('validation_rules'),
                                order_index=i + 1,
                                chunk_id=q.get('chunk_id'),
                                consolidation_notes=q.get('consolidation_notes')
                            )
                        
                        # Complete progress and show success
                        timing.finish("Policy processing completed successfully!")

                        st.session_state.current_policy_id = policy.id
                        st.session_state.current_questionnaire_id = questionnaire.id

                        st.success(f"Policy processed! Generated {len(result['questions'])} questions")
                        
                        # Add button to directly open review questionnaire tab
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Review & Edit Questions", type="primary", key="goto_review"):
                                st.session_state.selected_policy_for_review = policy.id
                                st.session_state.selected_questionnaire_for_review = questionnaire.id
                                st.success("Questions are ready for review! Switch to the 'Review Questionnaire' tab above.")
                        
                        with col2:
                            st.info("Click to open the review questionnaire with your new questions")

                        # Show preview
                        with st.expander("Preview Questions"):
                            for i, q in enumerate(result['questions'][:5], 1):
                                st.write(f"{i}. {q['question_text']}")
                            if len(result['questions']) > 5:
                                st.write(f"... and {len(result['questions']) - 5} more")
                    finally:
                        close_db(db)
                else:
                    timing.clear()
                    st.error(f"Failed to process policy: {result.get('error')}")

            except Exception as e:
                # Clear any progress indicators
                try:
                    timing.clear()
                except:
                    pass
                st.error(f"Error processing policy: {e}")
                import traceback
                st.code(traceback.format_exc())

    with tab2:
        st.subheader("View Policies & Questions")

        db = get_db_for_streamlit()
        try:
            policies = get_all_policies(db)

            if policies:
                # Add policy selector
                policy_options = {f"{p.name} (ID: {p.id})": p.id for p in policies}
                selected_policy_name = st.selectbox("Select Policy to View", list(policy_options.keys()))

                if selected_policy_name:
                    policy_id = policy_options[selected_policy_name]
                    selected_policy = next(p for p in policies if p.id == policy_id)
                    
                    # Show policy details
                    st.markdown(f"### Policy: {selected_policy.name}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Policy ID:** {selected_policy.id}")
                        if hasattr(selected_policy, 'file') and selected_policy.file:
                            st.write(f"**Uploaded:** {selected_policy.file.uploaded_at}")
                    
                    with col2:
                        questionnaires = get_questionnaires_by_policy(db, policy_id)
                        st.write(f"**Questionnaires:** {len(questionnaires)}")
                    
                    with col3:
                        # Placeholder for alignment
                        st.write("")
                    
                    # Show all questions for this policy
                    st.markdown("---")
                    st.markdown("### Generated Questions")
                    
                    if questionnaires:
                        for q_idx, questionnaire in enumerate(questionnaires, 1):
                            status = "Approved" if questionnaire.is_approved else "Pending Review"
                            st.markdown(f"#### Questionnaire {q_idx} - {status}")
                            
                            questions = get_questions_by_questionnaire(db, questionnaire.id)
                            
                            if questions:
                                # Group questions by criterion type for better organization
                                questions_by_type = {}
                                for question in questions:
                                    criterion = question.criterion_type or 'general'
                                    if criterion not in questions_by_type:
                                        questions_by_type[criterion] = []
                                    questions_by_type[criterion].append(question)
                                
                                # Display questions grouped by type
                                for criterion_type, criterion_questions in questions_by_type.items():
                                    with st.expander(f"{criterion_type.replace('_', ' ').title()} Questions ({len(criterion_questions)})", expanded=True):
                                        for i, question in enumerate(criterion_questions, 1):
                                            st.markdown(f"**{i}. {question.question_text}**")
                                            
                                            # Show question details
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.caption(f"Type: {question.question_type or 'text'}")
                                            with col2:
                                                if question.page_number:
                                                    st.caption(f"Page: {question.page_number}")
                                            with col3:
                                                st.caption(f"Category: {question.criterion_type or 'general'}")
                                            
                                            # Show answer options if available
                                            if question.answer_options:
                                                if isinstance(question.answer_options, list):
                                                    st.caption(f"Options: {', '.join(question.answer_options)}")
                                                elif isinstance(question.answer_options, dict):
                                                    if question.answer_options.get('type') == 'numeric':
                                                        min_val = question.answer_options.get('min', 'N/A')
                                                        max_val = question.answer_options.get('max', 'N/A')
                                                        unit = question.answer_options.get('unit', '')
                                                        st.caption(f"Range: {min_val} - {max_val} {unit}")
                                            
                                            # Show source text snippet if available
                                            if question.source_text_snippet:
                                                with st.expander("Source Text", expanded=False):
                                                    st.text(question.source_text_snippet)
                                            
                                            st.markdown("---")
                                
                                # Action button to go to review tab
                                if st.button(f"Review & Edit Questions", key=f"review_q_{questionnaire.id}", type="primary"):
                                    st.session_state.selected_policy_for_review = policy_id
                                    st.session_state.selected_questionnaire_for_review = questionnaire.id
                                    st.success("Navigate to the 'Review Questionnaire' tab to edit these questions")
                            else:
                                st.info("No questions found for this questionnaire")
                    else:
                        st.info("No questionnaires found for this policy")
            else:
                st.info("No policies uploaded yet")
        finally:
            close_db(db)

    with tab3:
        st.subheader("Review & Edit Questionnaire")

        # Import the new PDF viewer component
        from frontend.components.pdf_viewer import create_side_by_side_review

        db = get_db_for_streamlit()
        try:
            policies = get_all_policies(db)

            if policies:
                policy_options = {f"{p.name} (ID: {p.id})": p.id for p in policies}
                
                # Check if a policy was pre-selected from another tab
                default_index = 0
                if 'selected_policy_for_review' in st.session_state:
                    for i, (name, pid) in enumerate(policy_options.items()):
                        if pid == st.session_state.selected_policy_for_review:
                            default_index = i
                            break
                
                selected_policy = st.selectbox("Select Policy", list(policy_options.keys()), index=default_index)

                if selected_policy:
                    policy_id = policy_options[selected_policy]
                    questionnaires = get_questionnaires_by_policy(db, policy_id)
                    
                    # Get the policy file for PDF display
                    selected_policy_obj = next(p for p in policies if p.id == policy_id)
                    pdf_bytes = None
                    if hasattr(selected_policy_obj, 'file') and selected_policy_obj.file:
                        pdf_bytes = selected_policy_obj.file.content

                    for q in questionnaires:
                        status = "Approved" if q.is_approved else "Pending Review"
                        st.markdown(f"### Medical Questionnaire {q.id} - {status}")
                        
                        questions = get_questions_by_questionnaire(db, q.id)

                        if pdf_bytes:
                            # Convert questions to the format expected by the PDF viewer
                            questions_data = []
                            for question in questions:
                                question_dict = {
                                    'id': question.id,
                                    'question_text': question.question_text,
                                    'question_type': question.question_type,
                                    'criterion_type': question.criterion_type,
                                    'page_number': question.page_number,
                                    'source_text_snippet': question.source_text_snippet,
                                    'answer_options': question.answer_options
                                }
                                questions_data.append(question_dict)
                            
                            # Display side-by-side review (always in edit mode)
                            create_side_by_side_review(questions_data, pdf_bytes, edit_mode=True, questionnaire_id=q.id)
                            
                            # Action buttons for side-by-side mode
                            st.markdown("---")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if not q.is_approved:
                                    if st.button(f"Approve Questionnaire", key=f"approve_sidebar_{q.id}", type="primary"):
                                        approve_questionnaire(db, q.id)
                                        st.success("Questionnaire approved!")
                                        st.rerun()
                            
                            with col2:
                                # Export questionnaire
                                if st.button(f"Export JSON", key=f"export_sidebar_{q.id}"):
                                    questionnaire_data = {
                                        "questionnaire_id": q.id,
                                        "policy_id": policy_id,
                                        "approved": q.is_approved,
                                        "questions": questions_data
                                    }
                                    st.download_button(
                                        label="Download Questionnaire JSON",
                                        data=json.dumps(questionnaire_data, indent=2),
                                        file_name=f"questionnaire_{q.id}.json",
                                        mime="application/json",
                                        key=f"download_sidebar_{q.id}"
                                    )
                            
                            with col3:
                                st.info(f"{len(questions_data)} questions ready for review")
                                
                        else:
                            st.error("PDF file not available for side-by-side view. Please contact administrator.")

            else:
                st.info("No policies available for review")
        finally:
            close_db(db)