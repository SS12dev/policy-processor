"""
Timing utilities for ETA and elapsed time calculations.
"""
import time
import streamlit as st
from typing import Callable, Dict, Any
import re


class SimpleTiming:
    """Simple timing utility that integrates with existing orchestrator messages."""
    
    def __init__(self):
        self.start_time = None
        self.progress_bar = None
        self.status_text = None
        
    def start(self, initial_message: str = "Starting..."):
        """Initialize timing."""
        self.start_time = time.time()
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.status_text.text(initial_message)
        
    def update(self, progress_percent: int, message: str):
        """Update progress with clean timing."""
        if self.progress_bar is None or self.status_text is None or self.start_time is None:
            return
            
        # Calculate elapsed time
        elapsed_seconds = int(time.time() - self.start_time)
        elapsed_str = f"{elapsed_seconds//60}m {elapsed_seconds%60}s" if elapsed_seconds >= 60 else f"{elapsed_seconds}s"
        
        # Clean the message - remove any existing timing info
        clean_message = re.sub(r'\(Elapsed:.*?\)', '', message)
        clean_message = re.sub(r'\(ETA:.*?\)', '', clean_message)
        clean_message = re.sub(r'\(.*?minutes?\)', '', clean_message)
        clean_message = clean_message.strip()
        
        # Create final message with elapsed time
        final_message = f"{clean_message} (Elapsed: {elapsed_str})"
        
        self.progress_bar.progress(progress_percent)
        self.status_text.text(final_message)
        
    def finish(self, success_message: str = "Completed"):
        """Finish with total time."""
        if self.start_time is None:
            return
            
        total_seconds = int(time.time() - self.start_time)
        total_time_str = f"{total_seconds//60}m {total_seconds%60}s" if total_seconds >= 60 else f"{total_seconds}s"
        
        if self.progress_bar:
            self.progress_bar.progress(100)
        if self.status_text:
            self.status_text.text(f"{success_message} (Total time: {total_time_str})")
            
    def clear(self):
        """Clear progress indicators."""
        if self.progress_bar:
            self.progress_bar.empty()
        if self.status_text:
            self.status_text.empty()


def create_simple_progress_callback(timing: SimpleTiming) -> Callable:
    """Create a simple progress callback that works with existing orchestrator messages."""
    def progress_callback(percent: int, message: str):
        timing.update(percent, message)
    return progress_callback


def estimate_processing_time(doc_type: str, content_size: int, num_items: int = 1) -> Dict[str, Any]:
    """Estimate processing time based on document type and size."""
    
    # More realistic base estimates in seconds
    base_estimates = {
        "policy": {
            "per_chunk": 35,  # Reduced from 45
            "setup_time": 10,
            "consolidation_time": 15
        },
        "application": {
            "per_page": 25,   # Reduced from 30
            "setup_time": 8,
            "consolidation_time": 12
        },
        "decision": {
            "per_question": 2,  # Reduced from 3
            "setup_time": 3,
            "consolidation_time": 8
        }
    }
    
    if doc_type not in base_estimates:
        doc_type = "application"  # fallback
        
    estimates = base_estimates[doc_type]
    
    if doc_type == "decision":
        # For decision making, estimate based on number of questions
        total_time = (estimates["setup_time"] + 
                     (num_items * estimates["per_question"]) + 
                     estimates["consolidation_time"])
    elif doc_type == "application":
        # For application processing
        total_time = (estimates["setup_time"] + 
                     (num_items * estimates["per_page"]) + 
                     estimates["consolidation_time"])
    else:
        # For policy processing
        total_time = (estimates["setup_time"] + 
                     (num_items * estimates["per_chunk"]) + 
                     estimates["consolidation_time"])
    
    return {
        "estimated_seconds": total_time,
        "estimated_minutes": total_time / 60,
        "formatted": f"{total_time//60}m {total_time%60}s" if total_time >= 60 else f"{total_time}s"
    }