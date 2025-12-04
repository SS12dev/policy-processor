"""
PDF Viewer Component with Page Highlighting
"""
import streamlit as st
import base64
import sys
from pathlib import Path
from typing import Optional, List

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from backend.database.db_crud import (
    create_question, delete_question, update_question, reorder_questions
)
from frontend.utils.database import get_db_for_streamlit, close_db

def display_pdf_with_highlighting(pdf_bytes: bytes, highlight_page: Optional[int] = None) -> None:
    """
    Display PDF with optional page highlighting using PDF.js viewer
    
    Args:
        pdf_bytes: PDF file content as bytes
        highlight_page: Page number to highlight (1-indexed)
    """
    # Convert PDF bytes to base64 for embedding
    pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
    
    # Create PDF.js viewer URL with highlighting
    if highlight_page:
        viewer_url = f"#page={highlight_page}&zoom=auto&highlight=true"
    else:
        viewer_url = "#zoom=auto"
    
    # Custom HTML for PDF viewer with highlighting
    pdf_viewer_html = f"""
    <div style="width: 100%; height: 800px; border: 1px solid #ddd; border-radius: 5px; overflow: hidden;">
        <iframe 
            src="data:application/pdf;base64,{pdf_base64}{viewer_url}"
            width="100%" 
            height="100%" 
            style="border: none;">
            <p>Your browser does not support PDFs. Please download the PDF to view it.</p>
        </iframe>
    </div>
    """
    
    # Add custom CSS for highlighting
    highlight_css = """
    <style>
    .pdf-highlight {
        background-color: rgba(255, 255, 0, 0.3);
        border: 2px solid #ffcc00;
        animation: highlightPulse 2s ease-in-out;
    }
    
    @keyframes highlightPulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 204, 0, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 204, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 204, 0, 0); }
    }
    </style>
    """
    
    # Display the viewer
    st.components.v1.html(highlight_css + pdf_viewer_html, height=820)


def create_pdf_js_viewer(pdf_bytes: bytes, highlight_page: Optional[int] = None) -> str:
    """
    Create a more advanced PDF.js viewer with custom controls
    
    Args:
        pdf_bytes: PDF file content as bytes
        highlight_page: Page number to highlight (1-indexed)
    
    Returns:
        HTML string for the PDF viewer
    """
    pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
    
    viewer_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
        <style>
            body {{ margin: 0; font-family: Arial, sans-serif; }}
            #pdfContainer {{ width: 100%; height: 800px; overflow: auto; border: 1px solid #ddd; }}
            .pdf-page {{ margin: 10px auto; display: block; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .highlighted-page {{ 
                border: 3px solid #ffcc00; 
                box-shadow: 0 0 20px rgba(255, 204, 0, 0.5);
                animation: highlightPulse 2s ease-in-out;
            }}
            .page-number {{ 
                text-align: center; 
                padding: 5px; 
                background: #f0f0f0; 
                margin: 5px 0; 
                font-weight: bold;
            }}
            @keyframes highlightPulse {{
                0% {{ box-shadow: 0 0 0 0 rgba(255, 204, 0, 0.7); }}
                70% {{ box-shadow: 0 0 0 15px rgba(255, 204, 0, 0); }}
                100% {{ box-shadow: 0 0 0 0 rgba(255, 204, 0, 0); }}
            }}
        </style>
    </head>
    <body>
        <div id="pdfContainer"></div>
        
        <script>
            const pdfData = 'data:application/pdf;base64,{pdf_base64}';
            const highlightPage = {highlight_page or 'null'};
            
            pdfjsLib.getDocument(pdfData).promise.then(function(pdf) {{
                const container = document.getElementById('pdfContainer');
                
                for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {{
                    pdf.getPage(pageNum).then(function(page) {{
                        const scale = 1.2;
                        const viewport = page.getViewport({{ scale: scale }});
                        
                        // Create page container
                        const pageDiv = document.createElement('div');
                        pageDiv.style.position = 'relative';
                        pageDiv.style.marginBottom = '20px';
                        
                        // Add page number
                        const pageLabel = document.createElement('div');
                        pageLabel.className = 'page-number';
                        pageLabel.textContent = `Page ${{pageNum}}`;
                        pageDiv.appendChild(pageLabel);
                        
                        // Create canvas
                        const canvas = document.createElement('canvas');
                        const context = canvas.getContext('2d');
                        canvas.height = viewport.height;
                        canvas.width = viewport.width;
                        canvas.className = 'pdf-page';
                        
                        // Highlight specific page
                        if (highlightPage === pageNum) {{
                            canvas.className += ' highlighted-page';
                            // Scroll to highlighted page
                            setTimeout(() => {{
                                canvas.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                            }}, 500);
                        }}
                        
                        pageDiv.appendChild(canvas);
                        container.appendChild(pageDiv);
                        
                        // Render page
                        const renderContext = {{
                            canvasContext: context,
                            viewport: viewport
                        }};
                        page.render(renderContext);
                    }});
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    return viewer_html


def display_advanced_pdf_viewer(pdf_bytes: bytes, highlight_page: Optional[int] = None) -> None:
    """
    Display advanced PDF viewer with page highlighting and smooth scrolling
    
    Args:
        pdf_bytes: PDF file content as bytes
        highlight_page: Page number to highlight (1-indexed)
    """
    viewer_html = create_pdf_js_viewer(pdf_bytes, highlight_page)
    st.components.v1.html(viewer_html, height=820, scrolling=True)


def create_question_with_source_display(question_data: dict, pdf_bytes: bytes, on_click_callback=None) -> None:
    """
    Display a question with source information and PDF highlighting capability
    
    Args:
        question_data: Dictionary containing question information
        pdf_bytes: PDF file content for highlighting
        on_click_callback: Function to call when question is clicked
    """
    page_num = question_data.get('page_number')
    source_snippet = question_data.get('source_text_snippet', '')
    
    # Create clickable question container
    with st.container():
        # Question header with source info
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{question_data['question_text']}**")
            if source_snippet:
                st.caption(f"Source: \"{source_snippet[:100]}...\"")
        
        with col2:
            # Page button functionality removed as requested
            pass
        
        # Question metadata
        metadata_cols = st.columns(3)
        with metadata_cols[0]:
            st.caption(f"Type: {question_data.get('question_type', 'text')}")
        with metadata_cols[1]:
            st.caption(f"Category: {question_data.get('criterion_type', 'general')}")
        with metadata_cols[2]:
            # Handle both single page numbers and page ranges
            if page_num:
                if isinstance(page_num, (list, tuple)):
                    if len(page_num) == 1:
                        st.caption(f"Page: {page_num[0]}")
                    else:
                        st.caption(f"Pages: {'-'.join(map(str, sorted(page_num)))}")
                elif isinstance(page_num, str) and '-' in page_num:
                    st.caption(f"Pages: {page_num}")
                else:
                    st.caption(f"Page: {page_num}")


def create_side_by_side_review(questions: List[dict], pdf_bytes: bytes, edit_mode: bool = False, questionnaire_id: int = None) -> None:
    """
    Create a side-by-side view of questions and PDF
    
    Args:
        questions: List of question dictionaries
        pdf_bytes: PDF file content
        edit_mode: Whether to enable editing of questions
        questionnaire_id: ID of the questionnaire for database operations
    """
    st.markdown("### Side-by-Side Questionnaire Review")
    
    # Questions are now managed directly in the database
    # No need for session state tracking
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Generated Questions")
        
        # Questions are now directly from database (fresh data on each load)
        all_questions = questions.copy()
        
        # Show question summary
        total_count = len(all_questions)
        st.info(f"Total: {total_count} questions")
        
        # Create question options with proper numbering
        question_options = ["Add New Question"]
        for i, q in enumerate(all_questions):
            question_text = q.get('question_text', '')
            display_text = f"{question_text[:50]}..." if len(question_text) > 50 else question_text
            question_options.append(f"Q{i+1}: {display_text}")
        
        # Reset selector if it's out of bounds due to deletions
        current_selector_value = st.session_state.get("question_selector", 0)
        if current_selector_value >= len(question_options):
            st.session_state.question_selector = 0
            current_selector_value = 0
        
        selected_question_index = st.selectbox(
            "Select Question to Review/Edit:",
            range(len(question_options)),
            format_func=lambda x: question_options[x],
            key="question_selector",
            index=current_selector_value
        )
        
        # Display selected question or new question form
        if selected_question_index == 0:  # Add New Question
            st.markdown("### Add New Question")
            create_new_question_form(pdf_bytes, questionnaire_id)
        else:
            # Display selected question for editing
            question_idx = selected_question_index - 1
            if question_idx < len(all_questions):
                selected_question = all_questions[question_idx]
                st.markdown(f"### Edit Question {question_idx + 1}")
                
                if edit_mode:
                    create_editable_question_display(
                        selected_question, 
                        pdf_bytes,
                        question_idx,
                        all_questions,  # Pass all questions for proper deletion handling
                        questionnaire_id  # Pass questionnaire_id for database operations
                    )
                else:
                    create_question_with_source_display(
                        selected_question, 
                        pdf_bytes,
                        None
                    )
            else:
                st.error("Selected question no longer exists. Please select another question.")
    
    with col2:
        st.markdown("#### Policy Document")
        
        # Display PDF
        display_advanced_pdf_viewer(pdf_bytes)


def create_editable_question_display(question: dict, pdf_bytes: bytes, question_index: int, all_questions: List[dict] = None, questionnaire_id: int = None) -> None:
    """
    Create an editable question display
    
    Args:
        question: Question dictionary
        pdf_bytes: PDF file content
        question_index: Index of the question for unique keys
        all_questions: List of all questions for deletion handling
        questionnaire_id: ID of the questionnaire for database operations
    """
    question_key = f"edit_q_{question.get('id', question_index)}"
    
    # Editable question text
    edited_text = st.text_area(
        "Question Text", 
        value=question.get('question_text', ''),
        key=f"{question_key}_text",
        height=100
    )
    
    col1, col2 = st.columns(2)
    with col1:
        # Question type selection
        type_options = ["yes_no", "yes_no_unknown", "numeric", "text", "date", "multiple_choice"]
        current_type = question.get('question_type', 'text')
        edited_type = st.selectbox(
            "Question Type",
            type_options,
            index=type_options.index(current_type) if current_type in type_options else 0,
            key=f"{question_key}_type"
        )
    
    with col2:
        # Criterion type selection
        criterion_options = ["eligibility", "medical_necessity", "documentation", "procedural", "exclusion"]
        current_criterion = question.get('criterion_type', 'eligibility')
        edited_criterion = st.selectbox(
            "Category",
            criterion_options,
            index=criterion_options.index(current_criterion) if current_criterion in criterion_options else 0,
            key=f"{question_key}_criterion"
        )
    
    # Page number display (read-only reference)
    page_reference = question.get('page_number', 'N/A')
    st.text_input(
        "Source Page Reference",
        value=str(page_reference),
        key=f"{question_key}_page_display",
        disabled=True,
        help="This shows where the question originated from in the PDF"
    )
    
    # Answer options editing
    st.write("**Answer Options:**")
    if edited_type in ["yes_no", "yes_no_unknown"]:
        if edited_type == "yes_no":
            options_text = "Yes, No, N/A"
        else:
            options_text = "Yes, No, Unknown, N/A"
        st.text_input("Options (comma-separated)", value=options_text, key=f"{question_key}_options", disabled=True)
    elif edited_type == "multiple_choice":
        current_options = question.get('answer_options', ["Option 1", "Option 2", "Option 3"])
        if isinstance(current_options, list):
            options_text = ", ".join(current_options)
        else:
            options_text = "Option 1, Option 2, Option 3"
        st.text_input("Options (comma-separated)", value=options_text, key=f"{question_key}_options")
    elif edited_type == "numeric":
        col_min, col_max, col_unit = st.columns(3)
        with col_min:
            min_val = st.number_input("Min Value", value=0, key=f"{question_key}_min")
        with col_max:
            max_val = st.number_input("Max Value", value=100, key=f"{question_key}_max")
        with col_unit:
            unit_val = st.text_input("Unit", value="", key=f"{question_key}_unit")
    
    # Source text editing
    edited_source = st.text_area(
        "Source Text Snippet",
        value=question.get('source_text_snippet', ''),
        key=f"{question_key}_source",
        height=80
    )
    
    # Save changes button
    st.markdown("---")
    if st.button(f"Save Changes to Question {question_index + 1}", key=f"{question_key}_save", type="primary"):
        if questionnaire_id and question.get('id'):
            db = get_db_for_streamlit()
            try:
                # Get updated values from form
                updated_text = st.session_state.get(f"{question_key}_text", question.get('question_text'))
                updated_type = st.session_state.get(f"{question_key}_type", question.get('question_type'))
                updated_criterion = st.session_state.get(f"{question_key}_criterion", question.get('criterion_type'))
                updated_source = st.session_state.get(f"{question_key}_source", question.get('source_text_snippet'))
                
                # Handle answer options based on question type
                updated_options = None
                if updated_type in ["yes_no", "yes_no_unknown"]:
                    if updated_type == "yes_no":
                        updated_options = ["Yes", "No", "N/A"]
                    else:
                        updated_options = ["Yes", "No", "Unknown", "N/A"]
                elif updated_type == "multiple_choice":
                    options_text = st.session_state.get(f"{question_key}_options", "")
                    if options_text:
                        updated_options = [opt.strip() for opt in options_text.split(",")]
                elif updated_type == "numeric":
                    min_val = st.session_state.get(f"{question_key}_min", 0)
                    max_val = st.session_state.get(f"{question_key}_max", 100)
                    unit_val = st.session_state.get(f"{question_key}_unit", "")
                    updated_options = {"type": "numeric", "min": min_val, "max": max_val, "unit": unit_val}
                
                # Update question in database
                updated_question = update_question(
                    db=db,
                    question_id=question.get('id'),
                    question_text=updated_text,
                    question_type=updated_type,
                    criterion_type=updated_criterion,
                    answer_options=updated_options,
                    source_text_snippet=updated_source
                )
                
                if updated_question:
                    st.success(f"Changes saved to database for Question {question_index + 1}!")
                else:
                    st.error("Failed to update question in database")
                    
            except Exception as e:
                st.error(f"Error saving changes: {e}")
            finally:
                close_db(db)
        else:
            st.error("Cannot save: Missing questionnaire ID or question ID")
    
    # Delete question option with confirmation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"Delete Question {question_index + 1}", key=f"{question_key}_delete", type="secondary"):
            # Use a confirmation step
            st.session_state[f"confirm_delete_{question.get('id')}"] = True
            st.rerun()
    
    with col2:
        # Show confirmation dialog if delete was clicked
        if st.session_state.get(f"confirm_delete_{question.get('id')}", False):
            st.warning("Are you sure?")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes, Delete", key=f"{question_key}_confirm_yes", type="primary"):
                    if questionnaire_id and question.get('id'):
                        db = get_db_for_streamlit()
                        try:
                            # Delete from database
                            success = delete_question(db, question.get('id'))
                            if success:
                                # Reorder remaining questions
                                reorder_questions(db, questionnaire_id)
                                st.success(f"Question {question_index + 1} deleted from database! Questions have been renumbered.")
                            else:
                                st.error("Failed to delete question from database")
                        except Exception as e:
                            st.error(f"Error deleting question: {e}")
                        finally:
                            close_db(db)
                    else:
                        st.error("Cannot delete: Missing questionnaire ID or question ID")
                    
                    # Clean up confirmation state
                    if f"confirm_delete_{question.get('id')}" in st.session_state:
                        del st.session_state[f"confirm_delete_{question.get('id')}"]
                    st.rerun()
            with col_no:
                if st.button("❌ Cancel", key=f"{question_key}_confirm_no"):
                    # Clean up confirmation state
                    if f"confirm_delete_{question.get('id')}" in st.session_state:
                        del st.session_state[f"confirm_delete_{question.get('id')}"]
                    st.rerun()


def create_new_question_form(pdf_bytes: bytes, questionnaire_id: int = None) -> None:
    """
    Create a form for adding a new question from scratch
    
    Args:
        pdf_bytes: PDF file content
        questionnaire_id: ID of the questionnaire to add the question to
    """
    st.info("Create a new question based on information you found in the PDF that the AI agent might have missed.")
    
    # New question text
    new_question_text = st.text_area(
        "Question Text", 
        placeholder="Enter your new question here...",
        key="new_question_text",
        height=100
    )
    
    col1, col2 = st.columns(2)
    with col1:
        # Question type selection
        type_options = ["yes_no", "yes_no_unknown", "numeric", "text", "date", "multiple_choice"]
        new_question_type = st.selectbox(
            "Question Type",
            type_options,
            key="new_question_type"
        )
    
    with col2:
        # Criterion type selection
        criterion_options = ["eligibility", "medical_necessity", "documentation", "procedural", "exclusion"]
        new_criterion_type = st.selectbox(
            "Category",
            criterion_options,
            key="new_question_criterion"
        )
    
    # Page number reference
    new_page_number = st.text_input(
        "Source Page Reference (where you found this information)",
        value="1",
        key="new_question_page",
        help="Enter page number or range (e.g., '1', '2-3', '5-7')"
    )
    
    # Answer options based on question type
    st.write("**Answer Options:**")
    answer_options = None
    
    if new_question_type in ["yes_no", "yes_no_unknown"]:
        if new_question_type == "yes_no":
            options_text = "Yes, No, N/A"
            answer_options = ["Yes", "No", "N/A"]
        else:
            options_text = "Yes, No, Unknown, N/A"
            answer_options = ["Yes", "No", "Unknown", "N/A"]
        st.text_input("Options (comma-separated)", value=options_text, key="new_question_options", disabled=True)
    elif new_question_type == "multiple_choice":
        options_text = st.text_input("Options (comma-separated)", value="Option 1, Option 2, Option 3", key="new_question_options")
        if options_text:
            answer_options = [opt.strip() for opt in options_text.split(",")]
    elif new_question_type == "numeric":
        col_min, col_max, col_unit = st.columns(3)
        with col_min:
            min_val = st.number_input("Min Value", value=0, key="new_question_min")
        with col_max:
            max_val = st.number_input("Max Value", value=100, key="new_question_max")
        with col_unit:
            unit_val = st.text_input("Unit", value="", key="new_question_unit")
        answer_options = {"type": "numeric", "min": min_val, "max": max_val, "unit": unit_val}
    
    # Source text from PDF
    new_source_text = st.text_area(
        "Source Text Snippet (copy relevant text from the PDF)",
        placeholder="Paste or type the relevant text from the PDF that supports this question...",
        key="new_question_source",
        height=100
    )
    
    # Additional notes
    notes = st.text_area(
        "Additional Notes (optional)",
        placeholder="Any additional context or reasoning for this question...",
        key="new_question_notes",
        height=60
    )
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add Question", key="add_new_question", type="primary", disabled=not new_question_text.strip()):
            if questionnaire_id:
                # Save to database
                db = get_db_for_streamlit()
                try:
                    # Get the next order index
                    from backend.database.db_crud import get_questions_by_questionnaire
                    existing_questions = get_questions_by_questionnaire(db, questionnaire_id)
                    next_order_index = len(existing_questions) + 1
                    
                    # Convert page_number to int if it's a single number, otherwise keep as string
                    page_num = None
                    if new_page_number.strip():
                        try:
                            page_num = int(new_page_number.strip())
                        except ValueError:
                            # Keep as string for ranges like "1-5"
                            page_num = new_page_number.strip()
                    
                    # Create question in database
                    new_db_question = create_question(
                        db=db,
                        questionnaire_id=questionnaire_id,
                        question_text=new_question_text,
                        question_type=new_question_type,
                        criterion_type=new_criterion_type,
                        page_number=page_num,
                        source_text_snippet=new_source_text,
                        answer_options=answer_options,
                        order_index=next_order_index,
                        question_id=f"q_{next_order_index}"
                    )
                    
                    st.success(f"New question added to database! Question ID: {new_db_question.id}")
                    
                    # Clear the form
                    keys_to_clear = [
                        'new_question_text', 'new_question_type', 'new_question_criterion',
                        'new_question_page', 'new_question_source', 'new_question_notes',
                        'new_question_options', 'new_question_min', 'new_question_max', 'new_question_unit'
                    ]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    # Clear any session-based question tracking since we're now using DB
                    if 'new_questions' in st.session_state:
                        del st.session_state['new_questions']
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error adding question to database: {e}")
                finally:
                    close_db(db)
            else:
                st.error("Cannot add question: No questionnaire ID provided")
    
    with col2:
        if st.button("Clear Form", key="clear_new_question"):
            # Clear all form fields
            keys_to_clear = [
                'new_question_text', 'new_question_type', 'new_question_criterion',
                'new_question_page', 'new_question_source', 'new_question_notes',
                'new_question_options', 'new_question_min', 'new_question_max', 'new_question_unit'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Questions are now automatically available in the dropdown after being added to the database