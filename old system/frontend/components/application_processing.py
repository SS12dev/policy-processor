"""
Application Processing Component
"""
import streamlit as st
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.pdf_processing.text_extraction import extract_text_with_pages
from backend.pdf_processing.image_conversion import convert_pdf_to_images
from backend.settings import backend_settings
from backend.database.db_crud import (
    create_file_storage, create_application, create_review_session, create_answer,
    get_all_policies, get_questionnaires_by_policy, get_questions_by_questionnaire
)
from frontend.utils.database import get_db_for_streamlit, close_db
from frontend.utils.session import get_orchestrator


def show_application_processing():
    """Display application processing page"""
    
    # Check if we should show the review interface
    if st.session_state.get('show_review_interface', False):
        show_answer_review_interface()
        return
    
    st.header("Application Processing")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Process New Application", "View Processed Applications"])
    
    with tab1:
        show_process_application_tab()
    
    with tab2:
        show_processed_applications_tab()


def show_process_application_tab():
    """Show the tab for processing new applications"""
    db = get_db_for_streamlit()
    try:
        # Select policy and questionnaire
        policies = get_all_policies(db)

        if not policies:
            st.warning("Please upload a policy first")
            return

        policy_options = {f"{p.name} (ID: {p.id})": p.id for p in policies}
        selected_policy = st.selectbox("Select Policy", list(policy_options.keys()))

        if selected_policy:
            policy_id = policy_options[selected_policy]
            questionnaires = [q for q in get_questionnaires_by_policy(db, policy_id) if q.is_approved]

            if not questionnaires:
                st.warning("No approved questionnaires for this policy. Please approve a questionnaire first.")
                return

            questionnaire = questionnaires[0]
            questions = get_questions_by_questionnaire(db, questionnaire.id)

            st.success(f"Using approved questionnaire with {len(questions)} questions")

            # Upload application
            st.subheader("Upload Patient Application")

            patient_name = st.text_input("Patient Name", placeholder="e.g., John Doe")
            application_file = st.file_uploader("Upload Application PDF", type=['pdf'])

            if st.button("Process Application", disabled=not (patient_name and application_file)):
                _process_application(db, patient_name, application_file, questionnaire, questions)
    finally:
        close_db(db)


def _process_application(db, patient_name, application_file, questionnaire, questions):
    """Process a patient application with enhanced timing and progress tracking"""
    try:
        # Initial progress setup
        initial_progress = st.progress(0)
        initial_status = st.empty()
        
        # Read PDF
        initial_status.text("Reading application PDF...")
        initial_progress.progress(10)
        pdf_bytes = application_file.read()

        # Extract text
        initial_status.text("Extracting text from application...")
        initial_progress.progress(30)
        pages_text = extract_text_with_pages(pdf_bytes)

        # Convert to images for multimodal processing
        initial_status.text("Converting pages to images...")
        initial_progress.progress(50)
        
        # Check if document is scanned (no extractable text)
        total_text_length = sum(len(page.get('text', '')) for page in pages_text)
        is_scanned = total_text_length == 0
        
        try:
            images = convert_pdf_to_images(
                pdf_bytes,
                poppler_path=backend_settings.POPPLER_PATH,
                high_compression=is_scanned  # Use compression for scanned docs
            )
            if is_scanned:
                initial_status.text("Scanned document detected - using optimized processing...")
        except Exception as e:
            images = []

        # Combine text and images (convert to base64 for agents)
        initial_status.text("Preparing data for analysis...")
        initial_progress.progress(60)
        import base64
        pages_data = []
        for i, page_text in enumerate(pages_text):
            page_data = {
                "page_number": page_text['page_number'],
                "text": page_text['text'],
                "text_content": page_text['text']  # Agent 2 expects this field name
            }
            if i < len(images):
                # Images are already base64-encoded strings from convert_pdf_to_images
                page_data['image_base64'] = images[i]
            else:
                # Provide empty base64 if no image available
                page_data['image_base64'] = ""
            pages_data.append(page_data)

        # Prepare questions for agent
        questions_data = [
            {
                "question_id": q.id,
                "question_text": q.question_text
            }
            for q in questions
        ]

        # Clear initial progress and setup simple timing
        initial_progress.empty()
        initial_status.empty()
        
        # Setup simple timing for application processing
        from frontend.utils.timing import SimpleTiming, create_simple_progress_callback
        
        # Create simple timing tracker
        timing = SimpleTiming()
        timing.start("Analyzing patient application with AI agent...")
        
        # Process with Agent 2 
        orch = get_orchestrator()
        result = asyncio.run(orch.process_application(pages_data, questions_data))

        if result['success']:
            timing.update(90, "Saving results to database...")

            # Save to database
            file_storage = create_file_storage(
                db,
                filename=application_file.name,
                file_data=pdf_bytes,
                file_type='application/pdf'
            )

            # Extract enhanced patient data from results
            patient_info = result.get('patient_info', {})
            clinical_details = result.get('clinical_details', {})
            
            # Use extracted patient name if available and more complete than user input
            extracted_name = patient_info.get('patient_name', '')
            final_patient_name = patient_name
            if extracted_name and extracted_name != 'Not Found' and len(extracted_name.strip()) > len(patient_name.strip()):
                final_patient_name = extracted_name
            
            # Extract cost information from the orchestrator result
            analysis_summary = result.get('analysis_summary', {})
            cost_data = {
                'llm_input_tokens': 0,
                'llm_output_tokens': 0, 
                'llm_total_cost': 0.0,
                'llm_model': 'Unknown'
            }
            
            # The orchestrator may return cost info in different formats, try to extract it
            if 'cost_info' in result:
                cost_info = result['cost_info']
                cost_data.update({
                    'llm_input_tokens': cost_info.get('input_tokens', 0),
                    'llm_output_tokens': cost_info.get('output_tokens', 0),
                    'llm_total_cost': cost_info.get('total_cost', 0.0),
                    'llm_model': cost_info.get('model', 'Unknown')
                })
                
            application = create_application(
                db,
                patient_name=final_patient_name,
                file_storage_id=file_storage.id,
                status='processed',
                patient_dob=patient_info.get('patient_dob'),
                patient_id=patient_info.get('patient_id'),
                insurance_info=patient_info.get('insurance_info'),
                primary_diagnosis=patient_info.get('primary_diagnosis'),
                bmi=patient_info.get('bmi'),
                weight=patient_info.get('weight'), 
                requesting_physician=patient_info.get('requesting_physician'),
                clinical_details=clinical_details,
                patient_summary=result.get('patient_summary'),
                llm_input_tokens=cost_data['llm_input_tokens'],
                llm_output_tokens=cost_data['llm_output_tokens'],
                llm_total_cost=cost_data['llm_total_cost'],
                llm_model=cost_data['llm_model']
            )

            review_session = create_review_session(
                db,
                application_id=application.id,
                questionnaire_id=questionnaire.id,
                status='pending'
            )

            # Save enhanced clinical answers
            for answer in result['answers']:
                create_answer(
                    db,
                    review_session_id=review_session.id,
                    question_id=answer['question_id'],
                    answer_text=answer['answer_text'],
                    source_page_number=answer.get('source_page_number'),
                    confidence_score=answer.get('confidence_score', 0.0),
                    source_text_snippet=answer.get('source_text_snippet'),
                    source_type=answer.get('source_type'),
                    reasoning=answer.get('reasoning')
                )

            # Complete progress and show success
            timing.finish("Application processing completed successfully!")

            st.session_state.current_application_id = application.id

            # Display success with cost information (internal tracking only)
            st.success(f"Application processed! Found answers for {result['answered_questions']}/{result['total_questions']} questions")
            
            # Display enhanced patient information
            st.subheader("Patient Clinical Summary")
            patient_info = result.get('patient_info', {})
            clinical_details = result.get('clinical_details', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Patient", patient_info.get('patient_name', 'Not Found'))
                st.metric("BMI", patient_info.get('bmi', 'Not Found'))
                st.metric("Weight", patient_info.get('weight', 'Not Found'))
            
            with col2:
                st.metric("Date of Birth", patient_info.get('patient_dob', 'Not Found'))
                st.metric("Patient ID", patient_info.get('patient_id', 'Not Found'))
                st.metric("Primary Diagnosis", patient_info.get('primary_diagnosis', 'Not Found'))
            
            with col3:
                st.metric("Insurance", patient_info.get('insurance_info', 'Not Found'))
                st.metric("Requesting Physician", patient_info.get('requesting_physician', 'Not Found'))
            
            # Clinical details in expandable sections
            if clinical_details:
                col1, col2 = st.columns(2)
                with col1:
                    if clinical_details.get('comorbidities'):
                        with st.expander("Comorbidities"):
                            for condition in clinical_details['comorbidities']:
                                st.write(f"• {condition}")
                    
                    if clinical_details.get('current_medications'):
                        with st.expander("Current Medications"):
                            for med in clinical_details['current_medications']:
                                st.write(f"• {med}")
                
                with col2:
                    if clinical_details.get('previous_treatments'):
                        with st.expander("Previous Treatments"):
                            for treatment in clinical_details['previous_treatments']:
                                st.write(f"• {treatment}")
                    
                    if clinical_details.get('lab_results'):
                        with st.expander("Lab Results"):
                            for lab in clinical_details['lab_results']:
                                st.write(f"• {lab}")
                    
                    if clinical_details.get('specialist_evaluations'):
                        with st.expander("Specialist Evaluations"):
                            for eval in clinical_details['specialist_evaluations']:
                                st.write(f"• {eval}")
            
            if result.get('patient_summary'):
                with st.expander("Complete Patient Summary"):
                    st.write(result['patient_summary'])
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Review & Edit Answers", type="primary", use_container_width=True):
                    st.session_state.show_review_interface = True
                    st.session_state.review_data = {
                        'patient_info': result['patient_info'],
                        'clinical_details': result['clinical_details'],
                        'answers': result['answers'],
                        'application_file': application_file,
                        'file_name': application_file.name if application_file else None
                    }
                    st.rerun()
            
            with col2:
                st.info("Proceed to 'Decision Review' after reviewing answers")
            with st.expander("Patient Information", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                patient_info = result['patient_info']
                with col1:
                    st.metric("Patient Name", patient_info.get('patient_name', 'Not Found'))
                with col2:
                    st.metric("Date of Birth", patient_info.get('patient_dob', 'Not Found'))
                with col3:
                    st.metric("Patient ID", patient_info.get('patient_id', 'Not Found'))

            # Enhanced answers display with page sources
            with st.expander("Application Analysis Results", expanded=True):
                answered = [a for a in result['answers'] if a['answer_text'] != 'NOT_FOUND']
                not_found = [a for a in result['answers'] if a['answer_text'] == 'NOT_FOUND']
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Questions Answered", len(answered))
                with col2:
                    st.metric("Questions Not Found", len(not_found))
                with col3:
                    st.metric("Total Pages Processed", result.get('summary', {}).get('pages_processed', 'N/A'))
                with col4:
                    chunks_processed = result.get('summary', {}).get('chunks_processed', 'N/A')
                    st.metric("Chunks Processed", chunks_processed)
                
                st.divider()
                
                # Show answered questions with enhanced display
                if answered:
                    st.subheader("Questions Successfully Answered")
                    for idx, ans in enumerate(answered):
                        with st.container():
                            st.markdown(f"**Question {idx + 1}:** {ans.get('question_text', 'N/A')}")
                            
                            # Answer details with source information
                            st.success(f"**Answer:** {ans['answer_text']}")
                            
                            # Show source information if available
                            source_info = []
                            if ans.get('source_page_number'):
                                source_info.append(f"Page {ans['source_page_number']}")
                            
                            if source_info:
                                st.caption(" • ".join(source_info))
                            
                            # Expandable sections for additional details
                            col1, col2 = st.columns(2)
                            with col1:
                                if ans.get('source_text_snippet'):
                                    with st.expander("View Source Text"):
                                        st.text(ans['source_text_snippet'])
                            
                            with col2:
                                if ans.get('reasoning'):
                                    with st.expander("AI Reasoning"):
                                        st.text(ans['reasoning'])
                            
                            st.divider()
                
                # Show questions not found
                if not_found:
                    with st.expander(f"Questions Not Found ({len(not_found)})"):
                        for idx, ans in enumerate(not_found[:10]):  # Limit to 10 for display
                            st.markdown(f"**{idx + 1}.** {ans.get('question_text', 'N/A')}")
                            if ans.get('reasoning'):
                                st.caption(f"{ans['reasoning']}")
                        
                        if len(not_found) > 10:
                            st.caption(f"... and {len(not_found) - 10} more questions")
        else:
            timing.clear()
            st.error(f"Failed to process application: {result.get('error')}")

    except Exception as e:
        # Clear any progress indicators
        try:
            timing.clear()
        except:
            pass
        st.error(f"Error processing application: {e}")
        import traceback
        st.code(traceback.format_exc())


def show_answer_review_interface():
    """Display the answer review interface with side-by-side PDF view"""
    import base64
    from io import BytesIO
    
    st.header("Review & Edit Application Answers")
    
    if 'review_data' not in st.session_state:
        st.error("No review data available. Please process an application first.")
        if st.button("← Back to Application Processing"):
            st.session_state.show_review_interface = False
            st.rerun()
        return
    
    review_data = st.session_state.review_data
    patient_info = review_data.get('patient_info', {})
    answers = review_data.get('answers', [])
    application_file = review_data.get('application_file')
    
    # Back button
    if st.button("← Back to Application Processing"):
        st.session_state.show_review_interface = False
        st.rerun()
    
    # Patient information header
    with st.container():
        st.subheader("Patient Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Patient Name", patient_info.get('patient_name', 'Not Found'))
        with col2:
            st.metric("Date of Birth", patient_info.get('patient_dob', 'Not Found'))
        with col3:
            st.metric("Patient ID", patient_info.get('patient_id', 'Not Found'))
    
    st.divider()
    
    # Initialize session state for edited answers if not exists
    if 'edited_answers' not in st.session_state:
        st.session_state.edited_answers = {str(ans.get('question_id', i)): ans for i, ans in enumerate(answers)}
    
    # Main layout with PDF viewer and questions
    pdf_images = []
    if application_file:
        # Convert PDF to images for display using existing backend
        try:
            # Read PDF file
            pdf_bytes = application_file.read()
            application_file.seek(0)  # Reset file pointer
            
            # Use existing PDF to image conversion
            pdf_images_b64 = convert_pdf_to_images(pdf_bytes)
            pdf_images = [(i + 1, base64.b64decode(img_b64)) for i, img_b64 in enumerate(pdf_images_b64)]
        except Exception as e:
            st.error(f"Error loading PDF: {e}")
            pdf_images = []
    
    # Create two columns: PDF viewer and questions
    col_pdf, col_questions = st.columns([1, 1])
    
    with col_pdf:
        st.subheader("Application Document")
        if pdf_images:
            # PDF page selector
            page_options = [f"Page {i}" for i, _ in pdf_images]
            selected_page_idx = st.selectbox("Select Page", range(len(page_options)), 
                                           format_func=lambda x: page_options[x])
            
            # Display selected PDF page
            if selected_page_idx < len(pdf_images):
                page_num, img_data = pdf_images[selected_page_idx]
                st.image(img_data, caption=f"Page {page_num}", use_column_width=True)
        else:
            st.info("PDF viewer not available")
    
    with col_questions:
        st.subheader("Questions & Answers")
        
        # Question navigation
        found_answers = [ans for ans in answers if ans.get('answer_text', '').strip() != 'NOT_FOUND']
        not_found_answers = [ans for ans in answers if ans.get('answer_text', '').strip() == 'NOT_FOUND']
        
        # Tabs for found and not found
        tab1, tab2 = st.tabs([f"Answered ({len(found_answers)})", f"Not Found ({len(not_found_answers)})"])
        
        with tab1:
            if found_answers:
                for idx, answer in enumerate(found_answers):
                    question_id = str(answer.get('question_id', idx))
                    with st.expander(f"Q{idx + 1}: {answer.get('question_text', 'N/A')[:60]}...", expanded=idx == 0):
                        # Display current answer
                        st.markdown(f"**Question:** {answer.get('question_text', 'N/A')}")
                        
                        # Editable answer
                        current_answer = st.session_state.edited_answers.get(question_id, answer)
                        new_answer = st.text_area(
                            "Answer:", 
                            value=current_answer.get('answer_text', ''), 
                            key=f"answer_{question_id}"
                        )
                        
                        # Update edited answers
                        if new_answer != current_answer.get('answer_text', ''):
                            st.session_state.edited_answers[question_id] = {
                                **current_answer,
                                'answer_text': new_answer,
                                'modified': True
                            }
                        
                        # Show source information
                        col1, col2 = st.columns(2)
                        with col1:
                            if answer.get('source_page_number'):
                                st.caption(f"Found on page {answer['source_page_number']}")
                                # Button to jump to that page in PDF viewer
                                if st.button(f"View Page {answer['source_page_number']}", key=f"view_page_{question_id}"):
                                    # Update PDF viewer to show that page
                                    if answer['source_page_number'] <= len(pdf_images):
                                        st.session_state[f"selected_page_{question_id}"] = answer['source_page_number'] - 1
                        
                        with col2:
                            if answer.get('source_text_snippet'):
                                with st.expander("Source Text"):
                                    st.text(answer['source_text_snippet'])
                        
                        if answer.get('reasoning'):
                            with st.expander("AI Reasoning"):
                                st.text(answer['reasoning'])
            else:
                st.info("No answered questions to review")
        
        with tab2:
            if not_found_answers:
                for idx, answer in enumerate(not_found_answers):
                    question_id = str(answer.get('question_id', idx))
                    with st.expander(f"Q{idx + 1}: {answer.get('question_text', 'N/A')[:60]}...", expanded=False):
                        st.markdown(f"**Question:** {answer.get('question_text', 'N/A')}")
                        
                        # Allow adding answer for not found questions
                        current_answer = st.session_state.edited_answers.get(question_id, answer)
                        new_answer = st.text_area(
                            "Add Answer:", 
                            value=current_answer.get('answer_text', '') if current_answer.get('answer_text') != 'NOT_FOUND' else '',
                            key=f"answer_notfound_{question_id}",
                            placeholder="Enter answer if you found it in the document..."
                        )
                        
                        # Update edited answers
                        if new_answer.strip() and new_answer != current_answer.get('answer_text', ''):
                            st.session_state.edited_answers[question_id] = {
                                **current_answer,
                                'answer_text': new_answer,
                                'modified': True
                            }
                        
                        if answer.get('reasoning'):
                            st.caption(f"AI Reasoning: {answer['reasoning']}")
            else:
                st.info("All questions were answered")
    
    # Action buttons at the bottom
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Save Changes", type="primary", use_container_width=True):
            # Update the review data with edited answers
            updated_answers = list(st.session_state.edited_answers.values())
            st.session_state.review_data['answers'] = updated_answers
            st.success("Changes saved successfully!")
    
    with col2:
        if st.button("Reset to Original", use_container_width=True):
            # Reset to original answers
            st.session_state.edited_answers = {str(ans.get('question_id', i)): ans for i, ans in enumerate(answers)}
            st.info("↩Reset to original answers")
            st.rerun()
    
    with col3:
        if st.button("Proceed to Decision Review", use_container_width=True):
            # Save changes and proceed
            updated_answers = list(st.session_state.edited_answers.values())
            st.session_state.review_data['answers'] = updated_answers
            st.session_state.show_review_interface = False
            # Could navigate to decision review page here
            st.success("Proceeding to decision review with updated answers!")


def show_processed_applications_tab():
    """Show the tab for viewing processed applications with enhanced details like new processing"""
    from backend.database.db_crud import (
        get_all_applications, get_application_by_id, get_review_sessions_by_application,
        get_answers_by_session, get_questions_by_questionnaire
    )
    
    db = get_db_for_streamlit()
    try:
        applications = get_all_applications(db)
        
        if not applications:
            st.info("No applications have been processed yet.")
            return
        
        # Application selector
        app_options = {}
        for app in applications:
            patient_name = app.patient_name or "Unknown Patient"
            primary_diagnosis = app.primary_diagnosis or "Unknown Diagnosis"
            created_date = app.submission_date.strftime("%Y-%m-%d") if app.submission_date else "Unknown Date"
            app_options[f"{patient_name} - {primary_diagnosis} ({created_date})"] = app.id
        
        selected_app_name = st.selectbox(
            "Select Application to View", 
            list(app_options.keys()),
            key="processed_app_selector"
        )
        
        if not selected_app_name:
            return
        
        app_id = app_options[selected_app_name]
        application = get_application_by_id(db, app_id)
        
        if not application:
            st.error("Application not found.")
            return
        
        # Get review sessions to access answers
        review_sessions = get_review_sessions_by_application(db, app_id)
        if not review_sessions:
            st.warning("No review sessions found for this application.")
            return
        
        # Use the first (most recent) review session
        session = review_sessions[0]
        answers = get_answers_by_session(db, session.id)
        questions = get_questions_by_questionnaire(db, session.questionnaire_id)
        
        # Display application header similar to new processing
        st.success(f"Application processed! Found answers for {len([a for a in answers if a.answer_text and a.answer_text.strip() != 'NOT_FOUND'])}/{len(questions)} questions")
        
        # Patient Clinical Summary (matching new processing format)
        st.markdown("### Patient Clinical Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Patient", application.patient_name or "Unknown")
            st.metric("BMI", application.bmi or "Not Available")
            st.metric("Weight", application.weight or "Not Available")
        
        with col2:
            st.metric("Date of Birth", application.patient_dob or "Not Available")
            st.metric("Patient ID", application.patient_id or "Not Available")
            st.metric("Primary Diagnosis", application.primary_diagnosis or "Not Available")
        
        # Insurance info and physician
        if application.insurance_info or application.requesting_physician:
            col1, col2 = st.columns(2)
            with col1:
                if application.insurance_info:
                    st.write(f"**Insurance:** {application.insurance_info}")
            with col2:
                if application.requesting_physician:
                    st.write(f"**Requesting Physician:** {application.requesting_physician}")
        
        # Complete Patient Summary
        if application.patient_summary:
            with st.expander("Complete Patient Summary", expanded=False):
                st.write(application.patient_summary)
        
        # Analysis Results - matching new processing format
        st.markdown("### Application Analysis Results")
        
        # Summary metrics
        answered_count = len([a for a in answers if a.answer_text and a.answer_text.strip() != 'NOT_FOUND'])
        not_found_count = len(answers) - answered_count
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Questions Answered", answered_count)
        with col2:
            st.metric("Questions Not Found", not_found_count)
        with col3:
            created_date = application.submission_date.strftime("%Y-%m-%d") if application.submission_date else "N/A"
            st.metric("Total Pages Processed", "N/A")  # Not stored in old format
        with col4:
            st.metric("Chunks Processed", "N/A")  # Not stored in old format
        
        # Answered Questions
        answered_questions = [a for a in answers if a.answer_text and a.answer_text.strip() != 'NOT_FOUND']
        if answered_questions:
            st.markdown("### Questions Successfully Answered")
            
            for i, answer in enumerate(answered_questions, 1):
                # Find the corresponding question
                question = next((q for q in questions if q.id == answer.question_id), None)
                question_text = question.question_text if question else f"Question ID {answer.question_id}"
                
                st.markdown(f"**Question {i}: {question_text}**")
                st.write(f"**Answer:** {answer.answer_text}")
                
                # Additional details
                col1, col2 = st.columns(2)
                with col1:
                    if answer.source_page_number:
                        st.caption(f"Page {answer.source_page_number}")
                with col2:
                    # Additional info placeholder
                    st.caption("")
                
                # Source text and reasoning
                if answer.source_text_snippet:
                    with st.expander("View Source Text", expanded=False):
                        st.text(answer.source_text_snippet)
                
                if answer.reasoning:
                    with st.expander("AI Reasoning", expanded=False):
                        st.write(answer.reasoning)
                
                st.markdown("---")
        
        # Questions Not Found
        not_found_questions = [a for a in answers if not a.answer_text or a.answer_text.strip() == 'NOT_FOUND']
        if not_found_questions:
            st.markdown("### Questions Not Found")
            
            for i, answer in enumerate(not_found_questions, 1):
                question = next((q for q in questions if q.id == answer.question_id), None)
                question_text = question.question_text if question else f"Question ID {answer.question_id}"
                
                st.markdown(f"**{i}. {question_text}**")
                if answer.reasoning:
                    st.write(f"{answer.reasoning}")
                else:
                    st.write("The document does not provide information regarding this question.")
        
        # Review Sessions Details
        if review_sessions:
            st.markdown("### Review Sessions")
            for i, session in enumerate(review_sessions, 1):
                with st.expander(f"Review Session {i} - {session.status}", expanded=i==1):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Status:** {session.status}")
                        st.write(f"**Recommendation:** {session.final_recommendation or 'Pending'}")
                    
                    with col2:
                        st.write(f"**Priority:** {session.review_priority or 'Standard'}")
                    
                    with col3:
                        st.write(f"**Risk Level:** {session.overall_risk_level or 'Not Assessed'}")
                        if session.completed_at:
                            st.write(f"**Completed:** {session.completed_at.strftime('%Y-%m-%d')}")
                    
                    if session.primary_clinical_rationale:
                        st.markdown("**Clinical Rationale:**")
                        st.write(session.primary_clinical_rationale)
                    
    finally:
        close_db(db)