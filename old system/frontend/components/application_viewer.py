"""Application Viewer Component for displaying processed applications."""

import sys
from pathlib import Path

import streamlit as st

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.database.db_crud import (
    get_all_applications, get_application_by_id, get_review_sessions_by_application,
    get_answers_by_session
)
from frontend.utils.database import get_db_for_streamlit, close_db


def show_application_viewer():
    """Display application viewer page."""
    st.header("Application Processing Results")
    st.markdown("View processed applications with analysis results.")

    db = get_db_for_streamlit()
    try:
        applications = get_all_applications(db)

        if not applications:
            st.info("No applications have been processed yet. Go to 'Application Processing' to submit an application.")
            return

        # Application selector
        app_options = {}
        for app in applications:
            display_name = f"{app.patient_name or 'Unknown Patient'} - {app.primary_diagnosis[:50] if app.primary_diagnosis else 'No Diagnosis'}... (ID: {app.id})"
            app_options[display_name] = app.id

        selected_app_name = st.selectbox("Select Application to View", list(app_options.keys()))

        if selected_app_name:
            app_id = app_options[selected_app_name]
            application = get_application_by_id(db, app_id)
            
            if application:
                # Main application information
                st.markdown("---")
                st.markdown(f"### Patient: {application.patient_name or 'Unknown'}")
                
                # Application overview
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Date of Birth", application.patient_dob or "Not provided")
                
                with col2:
                    st.metric("Patient ID", application.patient_id or "Not provided")
                
                with col3:
                    if application.bmi:
                        st.metric("BMI", application.bmi)
                    else:
                        st.metric("BMI", "Not provided")
                
                with col4:
                    if application.weight:
                        st.metric("Weight", application.weight)
                    else:
                        st.metric("Weight", "Not provided")

                # Primary diagnosis and physician
                if application.primary_diagnosis:
                    st.markdown("#### Primary Diagnosis")
                    st.info(application.primary_diagnosis)
                
                if application.requesting_physician:
                    st.markdown("#### Requesting Physician")
                    st.info(application.requesting_physician)

                # Patient summary
                if application.patient_summary:
                    st.markdown("#### Patient Summary")
                    with st.expander("View Full Patient Summary", expanded=False):
                        st.write(application.patient_summary)

                # Clinical details
                if application.clinical_details:
                    st.markdown("#### Clinical Details")
                    with st.expander("View Clinical Details", expanded=False):
                        clinical_data = application.clinical_details
                        
                        if isinstance(clinical_data, dict):
                            for key, value in clinical_data.items():
                                if value:
                                    st.markdown(f"**{key.replace('_', ' ').title()}:**")
                                    if isinstance(value, list):
                                        for item in value:
                                            st.write(f"• {item}")
                                    else:
                                        st.write(value)
                                    st.markdown("---")

                # Review sessions and decisions
                st.markdown("---")
                st.markdown("### Review Sessions & Decisions")
                
                review_sessions = get_review_sessions_by_application(db, app_id)
                
                if review_sessions:
                    for idx, session in enumerate(review_sessions, 1):
                        st.markdown(f"#### Review Session {idx}")
                        
                        # Session overview
                        session_col1, session_col2, session_col3 = st.columns(3)
                        
                        with session_col1:
                            st.metric("Recommendation", session.final_recommendation or "Pending")
                        
                        with session_col2:
                            st.metric("Priority", session.review_priority or "Standard")
                        
                        with session_col3:
                            st.metric("Status", session.status or "Unknown")

                        # Detailed reasoning
                        if session.detailed_reasoning:
                            with st.expander(f"Detailed Reasoning - Session {idx}", expanded=False):
                                st.write(session.detailed_reasoning)

                        # Decision factors
                        if session.decision_factors:
                            with st.expander(f"Decision Factors - Session {idx}", expanded=False):
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

                        # Clinical rationales
                        if session.primary_clinical_rationale:
                            st.markdown(f"**Primary Clinical Rationale:**")
                            st.info(session.primary_clinical_rationale)

                        if session.medical_necessity_assessment:
                            st.markdown(f"**Medical Necessity Assessment:**")
                            st.info(session.medical_necessity_assessment)

                        # Answers for this session
                        st.markdown(f"**Extracted Answers - Session {idx}:**")
                        answers = get_answers_by_session(db, session.id)
                        
                        if answers:
                            # Group answers by found/not found
                            found_answers = [a for a in answers if a.answer_text and a.answer_text.strip() != 'NOT_FOUND']
                            not_found_answers = [a for a in answers if not a.answer_text or a.answer_text.strip() == 'NOT_FOUND']
                            
                            answer_col1, answer_col2 = st.columns(2)
                            
                            with answer_col1:
                                st.metric("Answers Found", len(found_answers))
                            
                            with answer_col2:
                                st.metric("Not Found", len(not_found_answers))

                            # Show found answers
                            if found_answers:
                                with st.expander(f"Found Answers ({len(found_answers)})", expanded=False):
                                    for answer in found_answers:
                                        if hasattr(answer, 'question') and answer.question:
                                            st.markdown(f"**Q:** {answer.question.question_text}")
                                            st.markdown(f"**A:** {answer.answer_text}")
                                            
                                            if answer.source_page_number:
                                                st.caption(f"Source: Page {answer.source_page_number}")
                                            st.markdown("---")

                            # Show not found answers
                            if not_found_answers:
                                with st.expander(f"Not Found Answers ({len(not_found_answers)})", expanded=False):
                                    for answer in not_found_answers:
                                        if hasattr(answer, 'question') and answer.question:
                                            st.markdown(f"**Q:** {answer.question.question_text}")
                                            st.caption("Answer not found in application documents")
                                            st.markdown("---")

                        st.markdown("---")
                
                else:
                    st.info("No review sessions found for this application.")
                # Create side-by-side layout with PDF and processing results
                st.markdown("---")
                st.markdown("### Application Analysis & Documents")
                
                # Create two-column layout
                pdf_col, results_col = st.columns([1, 1])
                
                with pdf_col:
                    st.subheader("Original Application PDF")
                    if hasattr(application, 'file') and application.file:
                        try:
                            from frontend.components.pdf_viewer import display_advanced_pdf_viewer
                            display_advanced_pdf_viewer(application.file.content)
                        except Exception as e:
                            st.error(f"Error loading PDF: {e}")
                            st.info("PDF content not available")
                    else:
                        st.info("No PDF available for this application")
                
                with results_col:
                    st.subheader("Processing Results")
                    
                    # Show key extracted information
                    st.markdown("#### Key Information Extracted")
                    
                    # Patient demographics
                    if application.patient_name or application.patient_dob or application.patient_id:
                        with st.expander("Patient Demographics", expanded=True):
                            if application.patient_name:
                                st.write(f"**Name:** {application.patient_name}")
                            if application.patient_dob:
                                st.write(f"**Date of Birth:** {application.patient_dob}")  
                            if application.patient_id:
                                st.write(f"**Patient ID:** {application.patient_id}")
                            if application.insurance_info:
                                st.write(f"**Insurance:** {application.insurance_info}")
                    
                    # Clinical measurements
                    if application.bmi or application.weight:
                        with st.expander("Clinical Measurements", expanded=True):
                            if application.bmi:
                                st.write(f"**BMI:** {application.bmi}")
                            if application.weight:
                                st.write(f"**Weight:** {application.weight}")
                    
                    # Medical information
                    if application.primary_diagnosis or application.requesting_physician:
                        with st.expander("Medical Information", expanded=True):
                            if application.primary_diagnosis:
                                st.write(f"**Primary Diagnosis:** {application.primary_diagnosis}")
                            if application.requesting_physician:
                                st.write(f"**Requesting Physician:** {application.requesting_physician}")
                    
                    # Clinical details
                    if application.clinical_details:
                        with st.expander("Clinical Details", expanded=False):
                            clinical_data = application.clinical_details
                            
                            if isinstance(clinical_data, dict):
                                for key, value in clinical_data.items():
                                    if value:
                                        st.markdown(f"**{key.replace('_', ' ').title()}:**")
                                        if isinstance(value, list):
                                            for item in value:
                                                st.write(f"• {item}")
                                        else:
                                            st.write(value)
                                        st.markdown("---")
                    
                    # Show answers if available through review sessions
                    if review_sessions:
                        st.markdown("#### Extracted Answers")
                        for session_idx, session in enumerate(review_sessions, 1):
                            answers = get_answers_by_session(db, session.id)
                            if answers:
                                found_answers = [a for a in answers if a.answer_text and a.answer_text.strip() != 'NOT_FOUND']
                                
                                if found_answers:
                                    with st.expander(f"Session {session_idx} - Found Answers ({len(found_answers)})", expanded=False):
                                        for answer in found_answers:
                                            if hasattr(answer, 'question') and answer.question:
                                                st.markdown(f"**Q:** {answer.question.question_text}")
                                                st.markdown(f"**A:** {answer.answer_text}")
                                                
                                                if answer.source_page_number:
                                                    st.caption(f"Source: Page {answer.source_page_number}")
                                                st.markdown("---")

    finally:
        close_db(db)