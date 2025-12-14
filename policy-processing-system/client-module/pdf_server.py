"""
Simple HTTP server to serve PDFs for viewing in Streamlit.
Runs on port 8502 alongside Streamlit (port 8501).

This avoids browser data URI size limits by serving PDFs via HTTP.
"""

from flask import Flask, Response, request
from flask_cors import CORS
import base64
import sqlite3
from pathlib import Path
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for Streamlit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent / "data" / "policy_processor.db"


@app.route('/pdf/<job_id>')
def serve_pdf(job_id):
    """Serve PDF for a given job ID."""
    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get PDF from policy_documents table
        cursor.execute(
            "SELECT content_base64, filename FROM policy_documents WHERE job_id = ?",
            (job_id,)
        )
        result = cursor.fetchone()
        conn.close()

        if not result:
            logger.error(f"PDF not found for job_id: {job_id}")
            return Response("PDF not found", status=404)

        content_base64, filename = result

        # Decode base64 PDF
        pdf_bytes = base64.b64decode(content_base64)

        logger.info(f"Serving PDF for job {job_id}: {len(pdf_bytes)} bytes")

        # Return PDF with proper headers
        response = Response(pdf_bytes, mimetype='application/pdf')
        response.headers['Content-Disposition'] = f'inline; filename="{filename or "document.pdf"}"'
        response.headers['Cache-Control'] = 'public, max-age=3600'  # Cache for 1 hour

        return response

    except Exception as e:
        logger.error(f"Error serving PDF for job {job_id}: {e}")
        return Response(f"Error: {str(e)}", status=500)


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "pdf-server"}


if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("Starting PDF Server")
    logger.info("=" * 80)
    logger.info("Server: http://localhost:8502")
    logger.info("Usage: http://localhost:8502/pdf/{job_id}")
    logger.info("=" * 80)

    # Run on port 8502 (Streamlit uses 8501)
    app.run(host='0.0.0.0', port=8502, debug=False)
