"""
Decision Review Component - Enhanced with tabbed interface and full PDF viewer
"""
import streamlit as st
import asyncio
import sys 
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from backend.database.db_crud import (
    get_all_review_sessions, get_answers_by_session
)
from frontend.utils.database import get_db_for_streamlit, close_db
from frontend.utils.session import get_orchestrator
from frontend.components.pdf_viewer import display_advanced_pdf_viewer


def show_decision_review():
    """Display decision review page with tabbed interface"""
    st.header("Decision Review")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Generate Decision", "Review & Edit Decision", "All Applications"])

    with tab1:
        show_generate_decision_tab()
        
    with tab2:
        show_review_edit_tab()
        
    with tab3:
        show_all_applications_tab()


def show_generate_decision_tab():
    """Generate initial decision for new applications"""
    st.subheader("Generate Authorization Decision")
    
    db = get_db_for_streamlit()
    try:
        # Get sessions that haven't been processed yet
        sessions = get_all_review_sessions(db)
        pending_sessions = [s for s in sessions if s.status in ['pending', 'Pending']]

        if not pending_sessions:
            st.info("No pending applications for decision generation")
            return

        session_options = {
            f"Session {s.id} - {s.application.patient_name}": s.id
            for s in pending_sessions
        }

        selected_session = st.selectbox("Select Application for Decision", list(session_options.keys()))

        if selected_session:
            session_id = session_options[selected_session]
            session = next(s for s in pending_sessions if s.id == session_id)

            st.subheader(f"Patient: {session.application.patient_name}")
            st.write(f"Policy: {session.questionnaire.policy.name}")
            st.write(f"Status: {session.status}")

            # Get answers
            answers = get_answers_by_session(db, session_id)
            st.write(f"Total Answers: {len(answers)}")

            if st.button("Generate Authorization Decision", type="primary", width="stretch"):
                # Setup simple timing for decision generation
                from frontend.utils.timing import SimpleTiming
                
                # Create simple timing tracker
                timing = SimpleTiming()
                timing.start("Generating authorization decision using AI agent...")
                
                try:
                    _generate_decision(db, session, answers, timing)
                except Exception as e:
                    timing.clear()
                    st.error(f"Error generating decision: {str(e)}")
    finally:
        close_db(db)


def show_review_edit_tab():
    """Review and edit existing decisions"""
    st.subheader("Review & Edit Decision")
    
    db = get_db_for_streamlit()
    try:
        # Get all sessions with decisions
        sessions = get_all_review_sessions(db)
        completed_sessions = [s for s in sessions if s.status == 'completed' and s.final_recommendation]

        if not completed_sessions:
            st.info("No completed decisions available for review")
            return

        session_options = {
            f"Session {s.id} - {s.application.patient_name} ({s.final_recommendation})": s.id
            for s in completed_sessions
        }

        selected_session = st.selectbox("Select Session to Review", list(session_options.keys()))

        if selected_session:
            session_id = session_options[selected_session]
            session = next(s for s in completed_sessions if s.id == session_id)
            
            # Display the decision review interface
            show_decision_review_interface(db, session)
            
    finally:
        close_db(db)


def show_all_applications_tab():
    """Show all applications in a table view"""
    st.subheader("All Applications")
    
    db = get_db_for_streamlit()
    try:
        sessions = get_all_review_sessions(db)
        
        if not sessions:
            st.info("No applications found")
            return
        
        # Create table data with cost information
        table_data = []
        for session in sessions:
            # Calculate total cost (application + decision)
            app_cost = session.application.llm_total_cost or 0.0
            decision_cost = session.llm_total_cost or 0.0
            
            table_data.append({
                "Session ID": session.id,
                "Patient Name": session.application.patient_name or "Unknown",
                "Policy": session.questionnaire.policy.name,
                "Status": session.status,
                "Decision": session.final_recommendation or "Pending",
                "Created": session.created_at.strftime("%Y-%m-%d %H:%M") if session.created_at else "N/A"
            })
        
        # Display as dataframe
        st.dataframe(table_data, width="stretch")
        
        # Selection for quick actions
        session_ids = [s.id for s in sessions]
        selected_id = st.selectbox("Select Session for Quick Action", session_ids, 
                                 format_func=lambda x: f"Session {x} - {next(s.application.patient_name for s in sessions if s.id == x)}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("View Details", type="secondary"):
                # Set session state to switch to review tab
                st.session_state.selected_session_for_review = selected_id
                st.success(f"Switch to 'Review & Edit Decision' tab to view Session {selected_id}")
                
        with col2:
            if st.button("Regenerate Decision", type="primary"):
                selected_session = next(s for s in sessions if s.id == selected_id)
                answers = get_answers_by_session(db, selected_id)
                
                # Setup simple timing for decision regeneration
                from frontend.utils.timing import SimpleTiming
                
                timing = SimpleTiming()
                timing.start("Regenerating authorization decision...")
                
                try:
                    _generate_decision(db, selected_session, answers, timing)
                    st.rerun()
                except Exception as e:
                    timing.clear()
                    st.error(f"Error regenerating decision: {str(e)}")
        
    finally:
        close_db(db)


def show_decision_review_interface(db, session):
    """Display the full decision review interface with PDF viewer"""
    st.subheader(f"Review Decision for {session.application.patient_name}")
    
    # Get answers
    answers = get_answers_by_session(db, session.id)
    
    # Display decision summary with enhanced information
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write(f"**Final Decision:** {session.final_recommendation}")
    with col2:
        st.write(f"**Status:** {session.status}")
        if session.review_priority:
            st.write(f"**Priority:** {session.review_priority}")
    with col3:
        # Processing information placeholder
        st.write("")  # Empty space for layout balance
    
    # Enhanced reasoning display
    if session.detailed_reasoning:
        with st.expander("Detailed AI Reasoning", expanded=False):
            st.write(session.detailed_reasoning)
    
    if session.primary_clinical_rationale:
        st.write("**Primary Clinical Rationale:**")
        st.info(session.primary_clinical_rationale)
    
    if session.medical_necessity_assessment:
        st.write("**Medical Necessity Assessment:**")
        st.info(session.medical_necessity_assessment)
    
    # Decision factors
    if session.decision_factors:
        with st.expander("Decision Factors Breakdown", expanded=False):
            factors = session.decision_factors
            if isinstance(factors, dict):
                for factor_type, factor_list in factors.items():
                    st.markdown(f"**{factor_type.replace('_', ' ').title()}:**")
                    if isinstance(factor_list, list):
                        for factor in factor_list:
                            st.write(f"• {factor}")
                    else:
                        st.write(factor_list)
                    st.markdown("---")
    
    # Create layout with PDF viewer and question evaluations side by side
    pdf_col, eval_col = st.columns([1, 1])
    
    with pdf_col:
        st.subheader("Patient Application")
        if session.application.file_id:
            # Get the file from storage and display
            try:
                from backend.database.db_crud import get_file_storage
                file_storage = get_file_storage(db, session.application.file_id)
                if file_storage:
                    # Use the advanced PDF viewer with the file content
                    display_advanced_pdf_viewer(file_storage.content)
                else:
                    st.info("PDF file not found in storage")
            except Exception as e:
                st.error(f"Error loading PDF: {e}")
        else:
            st.info("No PDF available for this application")
    
    with eval_col:
        st.subheader("Question Evaluations")
        
        # Edit mode toggle
        edit_mode = st.checkbox("Edit Mode", key=f"edit_mode_{session.id}")
        
        # Display answers with edit capability
        if answers:
            for i, answer in enumerate(answers, 1):
                with st.expander(f"Question {i}: {answer.question.question_text[:100]}...", expanded=True):
                    st.write(f"**Full Question:** {answer.question.question_text}")
                    
                    if edit_mode:
                        # Editable evaluation
                        new_evaluation = st.text_area(
                            "Clinical Rationale:",
                            value=answer.clinical_rationale or answer.reasoning or "",
                            key=f"eval_{answer.id}",
                            height=100
                        )
                        
                        # Editable answer
                        new_answer = st.selectbox(
                            "Answer:",
                            options=["Yes", "No", "Partially", "N/A"],
                            index=["Yes", "No", "Partially", "N/A"].index(answer.answer_text) if answer.answer_text in ["Yes", "No", "Partially", "N/A"] else 0,
                            key=f"answer_{answer.id}"
                        )
                        
                        if st.button(f"Update Answer {i}", key=f"update_{answer.id}"):
                            # Update answer in database using the correct field
                            answer.clinical_rationale = new_evaluation
                            answer.answer_text = new_answer
                            db.commit()
                            st.success(f"Updated answer {i}")
                            st.rerun()
                    else:
                        # Read-only display
                        st.write(f"**Clinical Rationale:** {answer.clinical_rationale or answer.reasoning or 'No evaluation'}")
                        st.write(f"**Answer:** {answer.answer_text}")
                        
                        if answer.source_page_number:
                            st.write(f"**Page References:** Page {answer.source_page_number}")
        
        # Update decision button
        if edit_mode:
            st.divider()
            
            new_decision = st.selectbox(
                "Final Decision:",
                options=["Approved", "Denied", "Pending Review"],
                index=["Approved", "Denied", "Pending Review"].index(session.final_recommendation) if session.final_recommendation in ["Approved", "Denied", "Pending Review"] else 0
            )
            
            new_rationale = st.text_area(
                "Decision Rationale:",
                value=session.primary_clinical_rationale or "",
                height=150
            )
            
            if st.button("Update Final Decision", type="primary"):
                session.final_recommendation = new_decision
                session.primary_clinical_rationale = new_rationale
                db.commit()
                st.success("Decision updated successfully!")
                st.rerun()


def _generate_decision(db, session, answers, timing=None):
    """Generate authorization decision using the decision agent with progress tracking"""
    try:
        if timing:
            timing.update(30, "Preparing answers for analysis...")
        
        # Prepare answers for agent
        answers_data = [
            {
                "question_id": answer.question_id,
                "question_text": answer.question.question_text,
                "evaluation": answer.reasoning or answer.clinical_rationale or "",
                "answer_text": answer.answer_text,  # Fixed: was "answer", should be "answer_text"
                "page_references": str(answer.source_page_number) if answer.source_page_number else "",
                "confidence_score": 95 if answer.answer_text not in ["NOT_FOUND", "Not Found"] else 0,
                "source_page_number": answer.source_page_number,
                "reasoning": answer.reasoning or answer.clinical_rationale or ""
            }
            for answer in answers
        ]
        
        if timing:
            timing.update(50, "Analyzing patient context and policy requirements...")
        
        # Call decision agent
        orchestrator = get_orchestrator()
        
        async def generate_decision():
            # Prepare patient context from session
            patient_context = {
                "patient_name": session.application.patient_name,
                "patient_dob": session.application.patient_dob,
                "patient_id": session.application.patient_id,
                "bmi": session.application.bmi,
                "weight": session.application.weight,
                "primary_diagnosis": session.application.primary_diagnosis,
                "requesting_physician": session.application.requesting_physician,
                "clinical_details": session.application.clinical_details,
                "patient_summary": session.application.patient_summary
            }
            
            # Prepare reference context
            reference_context = {
                "current_date": datetime.now().strftime("%Y-%m-%d"),
                "policy_type": session.questionnaire.procedure_type,
                "policy_name": session.questionnaire.policy.name,
                "session_id": session.id,
                "application_id": session.application.id
            }
            
            decision_result = await orchestrator.make_decision(
                policy_name=session.questionnaire.policy.name,
                answers=answers_data,
                patient_context=patient_context,
                reference_context=reference_context
            )
            return decision_result
        
        if timing:
            timing.update(70, "Generating decision with AI agent...")
            
        # Run async function
        decision_result = asyncio.run(generate_decision())
        
        if timing:
            timing.update(90, "Saving decision to database...")
        
        if decision_result and decision_result.get('success'):
            recommendation = decision_result.get('recommendation', {})
            
            # Update session with decision and cost information
            session.final_recommendation = recommendation.get('recommendation', 'PEND')
            session.primary_clinical_rationale = recommendation.get('primary_clinical_rationale', '')
            session.confidence_score = recommendation.get('confidence_score', 0.0) / 100.0  # Convert to 0-1 scale
            session.review_priority = recommendation.get('review_priority', 'Routine')
            session.status = 'completed'
            
            # Extract enhanced reasoning fields
            session.detailed_reasoning = recommendation.get('detailed_reasoning', '')
            session.medical_necessity_assessment = recommendation.get('medical_necessity_assessment', '')
            
            # Extract decision factors as JSON
            decision_factors = recommendation.get('decision_factors', {})
            if decision_factors:
                session.decision_factors = decision_factors
            
            # Extract cost information (internal tracking only)
            cost_info = decision_result.get('cost_info', {})
            if cost_info:
                session.llm_input_tokens = cost_info.get('input_tokens', 0)
                session.llm_output_tokens = cost_info.get('output_tokens', 0)
                session.llm_total_cost = cost_info.get('total_cost', 0.0)
                session.llm_model = cost_info.get('model', 'Unknown')
            
            db.commit()
            
            if timing:
                timing.finish("Decision generation completed successfully!")
            
            st.success(f"Decision Generated: **{recommendation.get('recommendation', 'PEND')}**")
            
            if recommendation.get('primary_clinical_rationale'):
                st.write("**Primary Clinical Rationale:**")
                st.info(recommendation['primary_clinical_rationale'])
            
            # Display enhanced reasoning if available
            if recommendation.get('detailed_reasoning'):
                with st.expander("Detailed AI Reasoning", expanded=False):
                    st.write(recommendation['detailed_reasoning'])
            
            # Show summary statistics (no cost display for frontend users)
            summary = decision_result.get('summary', {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Requirements Met", summary.get('requirements_met', 0))
            with col2:
                st.metric("Not Met", summary.get('requirements_not_met', 0))
            with col3:
                st.metric("Missing Info", summary.get('missing_information', 0))
                
            st.rerun()
        else:
            if timing:
                timing.clear()
            error_msg = decision_result.get('error', 'Unknown error') if decision_result else 'No response from decision agent'
            st.error(f"Failed to generate decision: {error_msg}")
            
    except Exception as e:
        if timing:
            timing.clear()
        st.error(f"Error generating decision: {str(e)}")
        print(f"Decision generation error: {e}")  # For debugging