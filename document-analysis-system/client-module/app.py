"""
Streamlit UI for PDF Document Analysis
Main application file.
"""

import streamlit as st
import asyncio
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

from settings import settings
from a2a_client import get_client
from database import get_database

# Configure Streamlit page
st.set_page_config(
    page_title=settings.app_title,
    page_icon=settings.app_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "processing" not in st.session_state:
    st.session_state.processing = False


def main():
    """Main application."""
    st.title(f"{settings.app_icon} {settings.app_title}")
    st.markdown(f"**Version**: {settings.client_version}")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        streaming_enabled = st.toggle(
            "Enable Streaming",
            value=settings.prefer_streaming,
            help="Get real-time updates during processing"
        )

        st.divider()

        st.subheader("Agent Info")
        st.text(f"URL: {settings.agent_url}")
        st.text(f"Timeout: {settings.agent_timeout_seconds}s")

        # Check agent health
        if st.button("Check Agent Health"):
            with st.spinner("Checking..."):
                check_agent_health()

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📋 History", "📊 Stats"])

    with tab1:
        upload_and_analyze_tab(streaming_enabled)

    with tab2:
        history_tab()

    with tab3:
        stats_tab()


def check_agent_health():
    """Check if agent is healthy."""
    import httpx

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{settings.agent_url}/health")
            if response.status_code == 200:
                st.success("✅ Agent is healthy!")
                st.json(response.json())
            else:
                st.error(f"❌ Agent returned status {response.status_code}")
    except Exception as e:
        st.error(f"❌ Cannot connect to agent: {str(e)}")


def upload_and_analyze_tab(streaming_enabled: bool):
    """Upload and analyze documents tab."""
    st.header("Upload PDF Document")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help=f"Maximum file size: {settings.max_file_size_mb} MB"
    )

    if uploaded_file is not None:
        # Display file info
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filename", uploaded_file.name)
        with col2:
            st.metric("Size", f"{file_size_mb:.2f} MB")
        with col3:
            st.metric("Type", uploaded_file.type)

        # Validate file size
        if file_size_mb > settings.max_file_size_mb:
            st.error(f"❌ File too large! Maximum size is {settings.max_file_size_mb} MB")
            return

        # Analyze button
        if st.button("🚀 Analyze Document", type="primary", disabled=st.session_state.processing):
            st.session_state.processing = True
            asyncio.run(analyze_document(
                uploaded_file.getvalue(),
                uploaded_file.name,
                streaming_enabled
            ))
            st.session_state.processing = False

        # Display results if available
        if st.session_state.analysis_results:
            display_results(st.session_state.analysis_results, key_suffix="_current")


async def analyze_document(file_bytes: bytes, filename: str, streaming: bool):
    """Analyze a document."""
    client = get_client()

    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_placeholder = st.empty()

    try:
        # Connect to agent
        status_text.text("Connecting to agent...")
        await client.connect()
        progress_bar.progress(10)

        # Send document
        status_text.text("Sending document...")
        progress_bar.progress(20)

        # Process events
        messages = []
        final_response = None

        async for event in client.analyze_document(file_bytes, filename):
            event_type = event.get("type")

            if event_type == "status":
                status_msg = event.get("message", "Processing...")
                status_text.text(f"📄 {status_msg}")

                # Update progress based on status
                if "validating" in status_msg.lower():
                    progress_bar.progress(30)
                elif "parsing" in status_msg.lower():
                    progress_bar.progress(40)
                elif "extracting headings" in status_msg.lower():
                    progress_bar.progress(55)
                elif "extracting keywords" in status_msg.lower():
                    progress_bar.progress(70)
                elif "analyzing" in status_msg.lower():
                    progress_bar.progress(85)
                elif "formatting" in status_msg.lower():
                    progress_bar.progress(95)

            elif event_type == "message":
                messages.append(event.get("text", ""))

            elif event_type == "artifact":
                final_response = event.get("data")

            elif event_type == "complete":
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                final_response = event.get("response")
                break

            elif event_type == "error":
                status_text.error(f"❌ Error: {event.get('error')}")
                progress_bar.progress(100)
                return

        # Store and display results
        if final_response:
            st.session_state.analysis_results = final_response
            status_text.success("✅ Analysis completed successfully!")
        else:
            status_text.warning("⚠️ No results received")

    except Exception as e:
        status_text.error(f"❌ Error: {str(e)}")
        st.exception(e)
    finally:
        await client.close()


def display_results(results: dict, key_suffix: str = ""):
    """Display analysis results.

    Args:
        results: Analysis results dictionary
        key_suffix: Unique suffix for widget keys to avoid duplicate IDs
    """
    st.divider()
    st.header("📊 Analysis Results")

    # Document Info
    doc_info = results.get("document_info", {})
    st.subheader("📄 Document Information")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pages", doc_info.get("page_count") or 0)
    with col2:
        st.metric("Words", f"{doc_info.get('word_count') or 0:,}")
    with col3:
        st.metric("Size", f"{doc_info.get('size_mb') or 0:.2f} MB")
    with col4:
        st.metric("Processing", f"{results.get('processing_time') or 0:.1f}s")

    # Headings
    extraction = results.get("extraction_results", {})
    headings_data = extraction.get("headings", {})

    st.subheader("📑 Extracted Headings")
    if headings_data.get("items"):
        headings_df = pd.DataFrame(headings_data["items"])
        st.dataframe(headings_df[["level", "text", "type"]], width="stretch")
    else:
        st.info("No headings found")

    # Keywords
    keywords_data = extraction.get("keywords", {})
    st.subheader("🔑 Keywords")
    if keywords_data.get("items"):
        st.write(", ".join(keywords_data["items"]))
    else:
        st.info("No keywords found")

    # Analysis
    analysis = results.get("analysis", {})
    if analysis:
        st.subheader("🤖 AI Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Document Type:**", analysis.get("document_type", "Unknown"))
            st.write("**Complexity:**", analysis.get("complexity_level", "Unknown"))

        with col2:
            topics = analysis.get("main_topics", [])
            if topics:
                st.write("**Main Topics:**")
                for topic in topics:
                    st.write(f"- {topic}")

        st.write("**Summary:**")
        st.write(analysis.get("summary", "No summary available"))

        insights = analysis.get("key_insights", [])
        if insights:
            st.write("**Key Insights:**")
            for insight in insights:
                st.write(f"- {insight}")

    # Export options
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="📥 Download JSON",
            data=json.dumps(results, indent=2),
            file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key=f"download_json{key_suffix}"
        )

    with col2:
        # Convert to CSV (flatten the results)
        flat_data = {
            "pages": doc_info.get("page_count"),
            "words": doc_info.get("word_count"),
            "headings_count": headings_data.get("count"),
            "keywords_count": keywords_data.get("count"),
            "document_type": analysis.get("document_type"),
            "summary": analysis.get("summary")
        }
        df = pd.DataFrame([flat_data])

        st.download_button(
            label="📥 Download CSV",
            data=df.to_csv(index=False),
            file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key=f"download_csv{key_suffix}"
        )


def history_tab():
    """Display analysis history."""
    st.header("📋 Analysis History")

    db = get_database()

    # Get recent documents
    documents = db.get_recent_documents(limit=50)

    if not documents:
        st.info("No documents analyzed yet")
        return

    # Display as table
    df = pd.DataFrame(documents)
    df["upload_timestamp"] = pd.to_datetime(df["upload_timestamp"])
    df["file_size_mb"] = df["file_size_bytes"] / (1024 * 1024)

    st.dataframe(
        df[["filename", "file_size_mb", "upload_timestamp", "request_count"]].rename(columns={
            "filename": "Filename",
            "file_size_mb": "Size (MB)",
            "upload_timestamp": "Uploaded",
            "request_count": "Analyses"
        }),
        width="stretch"
    )

    # Select document to view details
    selected_idx = st.selectbox(
        "Select document to view details",
        range(len(documents)),
        format_func=lambda i: documents[i]["filename"]
    )

    if selected_idx is not None:
        selected_doc = documents[selected_idx]
        st.subheader(f"Details: {selected_doc['filename']}")

        # Get analyses for this document
        analyses = db.get_document_analyses(selected_doc["id"])

        if analyses:
            for i, analysis in enumerate(analyses):
                with st.expander(f"Analysis {i+1} - {analysis['request_timestamp']}", expanded=(i==0)):
                    st.write(f"**Status:** {analysis['status']}")
                    processing_time = analysis.get('processing_time_seconds') or 0
                    st.write(f"**Processing Time:** {processing_time:.2f}s")

                    if analysis.get("response_data"):
                        display_results(analysis["response_data"], key_suffix=f"_history_{i}")
        else:
            st.info("No analyses found for this document")


def stats_tab():
    """Display statistics."""
    st.header("📊 Statistics")

    db = get_database()

    col1, col2, col3 = st.columns(3)

    with col1:
        # Count total documents
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]
            st.metric("Total Documents", total_docs)

    with col2:
        # Count total analyses
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM responses WHERE status = 'success'")
            total_analyses = cursor.fetchone()[0]
            st.metric("Successful Analyses", total_analyses)

    with col3:
        # Average processing time
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT AVG(processing_time_seconds) FROM responses WHERE status = 'success'")
            avg_time = cursor.fetchone()[0] or 0
            st.metric("Avg Processing Time", f"{avg_time:.1f}s")

    st.divider()

    # Recent activity chart
    st.subheader("Recent Activity")

    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DATE(request_timestamp) as date, COUNT(*) as count
            FROM requests
            GROUP BY DATE(request_timestamp)
            ORDER BY date DESC
            LIMIT 30
        """)
        activity_data = cursor.fetchall()

    if activity_data:
        df = pd.DataFrame(activity_data, columns=["Date", "Count"])
        st.bar_chart(df.set_index("Date"))
    else:
        st.info("No activity data yet")


if __name__ == "__main__":
    main()
