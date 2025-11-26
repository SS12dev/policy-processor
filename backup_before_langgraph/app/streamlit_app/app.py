"""
Streamlit Application for Policy Document Processor.

Two-tab interface:
1. Upload & Process - Upload policy documents and process them
2. Review Decision Trees - View and interact with generated decision trees
"""

import streamlit as st
import json
import base64
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import app modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.streamlit_app.a2a_client import A2AClientSync
from app.database.operations import DatabaseOperations
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Policy Document Processor",
    page_icon=":page_facing_up:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database operations
@st.cache_resource
def get_database():
    """Get database operations instance."""
    db_path = Path("./data/policy_processor.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return DatabaseOperations(database_url=f"sqlite:///{db_path}")

# Initialize A2A client
@st.cache_resource
def get_a2a_client():
    """Get A2A client instance."""
    return A2AClientSync(base_url="http://localhost:8001", timeout=300.0)

db_ops = get_database()
a2a_client = get_a2a_client()

# App header
st.title("Policy Document Processor")
st.markdown("---")

# Sidebar - System Status
with st.sidebar:
    st.header("System Status")

    # Check A2A server health
    if a2a_client.check_health():
        st.success("OK A2A Server: Online")
    else:
        st.error("X A2A Server: Offline")
        st.warning("Please start the A2A server:\n```python main_a2a_simplified.py```")

    # Get agent card info
    agent_card = a2a_client.get_agent_card()
    if agent_card:
        st.info(f"**Agent:** {agent_card.get('name', 'Unknown')}\n\n**Version:** {agent_card.get('version', '0.0.0')}")

    st.markdown("---")

    # Job Statistics
    st.header("Job Statistics")
    total_jobs = db_ops.count_jobs()
    completed_jobs = db_ops.count_jobs(status="completed")
    failed_jobs = db_ops.count_jobs(status="failed")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Jobs", total_jobs)
        st.metric("Completed", completed_jobs)
    with col2:
        st.metric("Processing", total_jobs - completed_jobs - failed_jobs)
        st.metric("Failed", failed_jobs)

# Main content - Tabs
tab1, tab2 = st.tabs(["Upload & Process", "Review Decision Trees"])

# ============================================================================
# TAB 1: Upload & Process
# ============================================================================
with tab1:
    st.header("Upload Policy Document")

    col1, col2 = st.columns([2, 1])

    with col1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF policy document",
            type=["pdf"],
            help="Upload an insurance policy, legal document, or corporate policy PDF"
        )

    with col2:
        st.subheader("Processing Options")
        use_gpt4 = st.checkbox(
            "Use GPT-4",
            value=False,
            help="Use GPT-4 for complex documents (higher cost, better accuracy)"
        )

        enable_streaming = st.checkbox(
            "Enable Streaming",
            value=True,
            help="Stream processing progress in real-time"
        )

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence for validation"
        )

    # Process button
    if uploaded_file is not None:
        st.markdown("---")

        # Show file info
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(f"**File:** {uploaded_file.name} | **Size:** {file_size_mb:.2f} MB")

        if st.button(" Process Document", type="primary", use_container_width=True):
            with st.spinner("Processing document... This may take a few minutes."):
                try:
                    # Process the document
                    result = a2a_client.process_document(
                        pdf_bytes=uploaded_file.getvalue(),
                        filename=uploaded_file.name,
                        use_gpt4=use_gpt4,
                        enable_streaming=enable_streaming,
                        confidence_threshold=confidence_threshold
                    )

                    # Parse the result
                    if "response" in result and isinstance(result["response"], str):
                        response_text = result["response"]
                        st.success("Document processed successfully!")
                        st.markdown(response_text)

                        # Try to extract job_id
                        if "Job ID" in response_text:
                            # Store in session state for tab 2
                            st.session_state.last_job_id = response_text.split("`")[1] if "`" in response_text else None
                            if st.session_state.last_job_id:
                                st.info(f"Job ID: `{st.session_state.last_job_id}` - Switch to 'Review Decision Trees' tab to view results")
                    else:
                        st.success("Document processed successfully!")
                        st.json(result)

                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    logger.error(f"Processing error: {e}", exc_info=True)

    # Recent Jobs Section
    st.markdown("---")
    st.subheader("Recent Processing Jobs")

    # Get recent jobs
    recent_jobs = db_ops.list_jobs(limit=10)

    if recent_jobs:
        for job in recent_jobs:
            with st.expander(
                f"Job {job['job_id'][:8]}... - {job['status'].upper()} - {job.get('created_at', 'Unknown time')}"
            ):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(f"**Status:** {job['status']}")
                    st.write(f"**Type:** {job.get('document_type', 'N/A')}")

                with col2:
                    st.write(f"**Policies:** {job.get('total_policies', 'N/A')}")
                    st.write(f"**Confidence:** {job.get('validation_confidence', 0):.0%}" if job.get('validation_confidence') else "**Confidence:** N/A")

                with col3:
                    st.write(f"**Processing Time:** {job.get('processing_time_seconds', 'N/A')}s" if job.get('processing_time_seconds') else "**Processing Time:** N/A")

                # Load results button
                if job['status'] == 'completed':
                    if st.button(f"View Results", key=f"view_{job['job_id']}"):
                        st.session_state.selected_job_id = job['job_id']
                        st.rerun()
    else:
        st.info("No jobs found. Upload a document to get started!")

# ============================================================================
# TAB 2: Review Decision Trees
# ============================================================================
with tab2:
    st.header("Review Decision Trees")

    # Job selector
    col1, col2 = st.columns([3, 1])

    with col1:
        # Get job ID from session state or input
        selected_job_id = st.session_state.get('selected_job_id', st.session_state.get('last_job_id', ''))

        job_id_input = st.text_input(
            "Enter Job ID",
            value=selected_job_id,
            help="Enter the job ID from the processing step"
        )

    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        load_button = st.button(" Load Results", type="primary", use_container_width=True)

    if load_button and job_id_input:
        with st.spinner("Loading results..."):
            try:
                # Get results from A2A server
                results = a2a_client.get_results(job_id_input)

                # Store in session state
                st.session_state.current_results = results
                st.session_state.current_job_id = job_id_input

                st.success(f"Results loaded for job {job_id_input[:8]}...")

            except Exception as e:
                st.error(f"Error loading results: {str(e)}")
                logger.error(f"Error loading results: {e}", exc_info=True)

    # Display results if available
    if 'current_results' in st.session_state and st.session_state.current_results:
        results = st.session_state.current_results
        job_id = st.session_state.current_job_id

        st.markdown("---")

        # Parse results
        if "response" in results and isinstance(results["response"], str):
            # Try to parse JSON from response
            try:
                results_data = json.loads(results["response"])
            except json.JSONDecodeError:
                st.warning("Could not parse results as JSON. Showing raw response.")
                st.text(results["response"])
                results_data = None
        else:
            results_data = results

        if results_data:
            # Create side-by-side view option
            view_mode = st.radio(
                "View Mode",
                ["Decision Trees Only", "Side-by-Side (PDF + Decision Trees)"],
                horizontal=True
            )

            if view_mode == "Side-by-Side (PDF + Decision Trees)":
                # Get the original PDF from database
                with db_ops.get_session() as session:
                    from app.database.models import PolicyDocument
                    doc = session.query(PolicyDocument).filter_by(job_id=job_id).first()

                    if doc and doc.content_base64:
                        col_pdf, col_trees = st.columns([1, 1])

                        with col_pdf:
                            st.subheader(" Original Policy Document")

                            # Display PDF
                            pdf_bytes = base64.b64decode(doc.content_base64)
                            st.download_button(
                                " Download PDF",
                                data=pdf_bytes,
                                file_name=doc.filename or "policy.pdf",
                                mime="application/pdf"
                            )

                            # Embed PDF viewer
                            pdf_display = f'<iframe src="data:application/pdf;base64,{doc.content_base64}" width="100%" height="800px" type="application/pdf"></iframe>'
                            st.markdown(pdf_display, unsafe_allow_html=True)

                        with col_trees:
                            st.subheader(" Generated Decision Trees")
                            _display_decision_trees(results_data)
                    else:
                        st.warning("PDF not found in database. Showing decision trees only.")
                        _display_decision_trees(results_data)
            else:
                # Decision trees only
                _display_decision_trees(results_data)


def _display_decision_trees(results_data: dict):
    """Display decision trees from results data."""

    # Show metadata
    if "metadata" in results_data:
        metadata = results_data["metadata"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Pages", metadata.get("total_pages", "N/A"))
        with col2:
            st.metric("Document Type", metadata.get("document_type", "N/A"))
        with col3:
            st.metric("Processing Model", "GPT-4" if metadata.get("llm_model", "").endswith("-4") else "GPT-4o-mini")

    st.markdown("---")

    # Show validation results
    if "validation_result" in results_data:
        validation = results_data["validation_result"]

        st.subheader("OK Validation Results")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Overall Confidence",
                f"{validation.get('overall_confidence', 0):.0%}",
                delta="Pass" if validation.get('is_valid') else "Fail"
            )
        with col2:
            st.metric("Completeness", f"{validation.get('completeness_score', 0):.0%}")
        with col3:
            st.metric("Consistency", f"{validation.get('consistency_score', 0):.0%}")
        with col4:
            st.metric("Traceability", f"{validation.get('traceability_score', 0):.0%}")

    st.markdown("---")

    # Show policy hierarchy
    if "policy_hierarchy" in results_data:
        hierarchy = results_data["policy_hierarchy"]

        st.subheader(" Policy Hierarchy")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Policies", hierarchy.get("total_policies", 0))
        with col2:
            st.metric("Root Policies", hierarchy.get("total_root_policies", 0))
        with col3:
            st.metric("Max Depth", hierarchy.get("max_depth", 0))

        # Display root policies
        if "root_policies" in hierarchy:
            st.markdown("#### Root Policies")
            for policy in hierarchy["root_policies"]:
                with st.expander(f" {policy.get('title', 'Untitled Policy')}"):
                    st.write(f"**Description:** {policy.get('description', 'No description')}")
                    st.write(f"**Level:** {policy.get('level', 0)}")

                    if policy.get("conditions"):
                        st.write("**Conditions:**")
                        for cond in policy["conditions"]:
                            st.write(f"- {cond.get('description', 'No description')}")

                    if policy.get("children"):
                        st.write(f"**Sub-policies:** {len(policy['children'])}")

    st.markdown("---")

    # Show decision trees
    if "decision_trees" in results_data:
        trees = results_data["decision_trees"]

        st.subheader(f" Decision Trees ({len(trees)})")

        for i, tree in enumerate(trees, 1):
            with st.expander(f"Decision Tree {i}: {tree.get('policy_title', 'Untitled')}"):
                st.write(f"**Policy ID:** {tree.get('policy_id', 'N/A')}")
                st.write(f"**Total Questions:** {tree.get('total_questions', 0)}")
                st.write(f"**Max Depth:** {tree.get('max_depth', 0)}")

                # Show questions
                if tree.get("questions"):
                    st.markdown("#### Questions:")
                    for q_id, question in enumerate(tree["questions"], 1):
                        st.write(f"{q_id}. {question.get('question_text', 'No question')}")
                        if question.get("answer_type"):
                            st.write(f"   *Answer Type:* {question['answer_type']}")

    # Export button
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        # Export JSON
        json_str = json.dumps(results_data, indent=2, default=str)
        st.download_button(
            " Export as JSON",
            data=json_str,
            file_name=f"results_{st.session_state.current_job_id[:8]}.json",
            mime="application/json"
        )

    with col2:
        # Export Decision Trees only
        if "decision_trees" in results_data:
            trees_json = json.dumps(results_data["decision_trees"], indent=2, default=str)
            st.download_button(
                " Export Decision Trees",
                data=trees_json,
                file_name=f"decision_trees_{st.session_state.current_job_id[:8]}.json",
                mime="application/json"
            )


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Policy Document Processor v2.0 | Powered by A2A Protocol"
    "</div>",
    unsafe_allow_html=True
)
