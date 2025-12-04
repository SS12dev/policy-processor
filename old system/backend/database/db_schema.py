"""Database schema definitions for the Prior Authorization System."""

from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, LargeBinary, String, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class FileStorage(Base):
    """Store uploaded PDF files as BLOB."""
    
    __tablename__ = 'file_storage'

    id = Column(Integer, primary_key=True)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)  # 'policy' or 'application'
    content = Column(LargeBinary, nullable=False)  # Raw file bytes
    uploaded_at = Column(DateTime, default=datetime.utcnow)

class Policy(Base):
    """Represents a medical policy document."""
    
    __tablename__ = 'policy'

    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('file_storage.id'), nullable=False)
    name = Column(String(255), nullable=False)

    file = relationship("FileStorage")
    questionnaires = relationship("Questionnaire", back_populates="policy")

class Questionnaire(Base):
    """Generated medical questionnaire from policy (Agent 1 output)"""
    __tablename__ = 'questionnaire'

    id = Column(Integer, primary_key=True)
    policy_id = Column(Integer, ForeignKey('policy.id'), nullable=False)
    name = Column(String(255), nullable=False)
    total_questions = Column(Integer)
    complexity_level = Column(String(20))  # simple, moderate, complex
    coverage_areas = Column(JSON)  # medical_necessity, clinical_eligibility, documentation, etc.
    procedure_type = Column(String(100))  # bariatric_surgery, cardiac, oncology, etc.
    target_population = Column(Text)  # Description of patient population
    is_approved = Column(Boolean, default=False)  # HITL approval status
    created_at = Column(DateTime, default=datetime.utcnow)
    approved_at = Column(DateTime, nullable=True)
    # LLM Cost tracking fields
    llm_input_tokens = Column(Integer, default=0)
    llm_output_tokens = Column(Integer, default=0)
    llm_total_cost = Column(Float, default=0.0)
    llm_model = Column(String(100), nullable=True)

    policy = relationship("Policy", back_populates="questionnaires")
    questions = relationship("Question", back_populates="questionnaire")
    review_sessions = relationship("ReviewSession", back_populates="questionnaire")

class Question(Base):
    """Individual question in a questionnaire with enhanced medical metadata"""
    __tablename__ = 'question'

    id = Column(Integer, primary_key=True)
    questionnaire_id = Column(Integer, ForeignKey('questionnaire.id'), nullable=False)
    question_id = Column(String(100))  # Unique question identifier (e.g., BMI_001, AGE_001)
    question_text = Column(Text, nullable=False)
    question_type = Column(String(50))  # yes_no, numeric, text, date, multiple_choice, duration
    answer_options = Column(JSON)  # Enhanced medical answer options
    source_text_snippet = Column(Text)  # Exact policy text for highlighting
    page_number = Column(Integer)  # Source page number
    line_reference = Column(String(100))  # Line or section reference
    criterion_type = Column(String(50))  # medical_necessity, clinical_eligibility, documentation, procedural_requirements, exclusions
    priority_level = Column(String(20))  # high, medium, low
    approval_impact = Column(String(50))  # required_for_approval, preferred, informational
    validation_rules = Column(JSON)  # Medical validation logic
    order_index = Column(Integer)  # Question ordering
    chunk_id = Column(Integer)  # For traceability to processing chunks
    consolidation_notes = Column(Text)  # Notes from consolidation process

    questionnaire = relationship("Questionnaire", back_populates="questions")
    answers = relationship("Answer", back_populates="question")

class Application(Base):
    """Prior authorization application submitted by patient with enhanced clinical data"""
    __tablename__ = 'application'

    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('file_storage.id'), nullable=False)
    patient_name = Column(String(255))
    patient_dob = Column(String(50))
    patient_id = Column(String(100))
    insurance_info = Column(Text)
    primary_diagnosis = Column(Text)
    bmi = Column(String(50))
    weight = Column(String(50))
    requesting_physician = Column(String(255))
    clinical_details = Column(JSON)  # comorbidities, medications, treatments, lab_results, specialist_evaluations
    patient_summary = Column(Text)  # Comprehensive medical summary
    submission_date = Column(DateTime, default=datetime.utcnow)
    # LLM Cost tracking fields
    llm_input_tokens = Column(Integer, default=0)
    llm_output_tokens = Column(Integer, default=0)
    llm_total_cost = Column(Float, default=0.0)
    llm_model = Column(String(100), nullable=True)

    file = relationship("FileStorage")
    review_sessions = relationship("ReviewSession", back_populates="application")

class ReviewSession(Base):
    """Clinical decision-making session for prior authorization (Agent 3)"""
    __tablename__ = 'review_session'

    id = Column(Integer, primary_key=True)
    application_id = Column(Integer, ForeignKey('application.id'), nullable=False)
    questionnaire_id = Column(Integer, ForeignKey('questionnaire.id'), nullable=False)
    status = Column(String(50), default='Pending')  # Pending, In Review, Completed
    final_recommendation = Column(String(50))  # APPROVE, PEND, DENY
    confidence_score = Column(Float)  # Confidence in decision (0.0-1.0)
    primary_clinical_rationale = Column(Text)  # Primary medical reason
    medical_necessity_assessment = Column(Text)  # Clinical evaluation of medical necessity
    supporting_clinical_factors = Column(JSON)  # List of supporting clinical factors
    clinical_concerns = Column(JSON)  # List of medical concerns
    required_actions = Column(JSON)  # List of required actions
    review_priority = Column(String(20))  # Routine, Urgent, Expedited
    clinical_notes = Column(Text)  # Additional clinical considerations
    overall_risk_level = Column(String(20))  # Low, Medium, High
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    # Enhanced reasoning and cost tracking fields
    detailed_reasoning = Column(Text)  # Comprehensive reasoning for the decision
    decision_factors = Column(JSON)  # Structured factors leading to decision
    llm_input_tokens = Column(Integer, default=0)
    llm_output_tokens = Column(Integer, default=0)
    llm_total_cost = Column(Float, default=0.0)
    llm_model = Column(String(100), nullable=True)

    application = relationship("Application", back_populates="review_sessions")
    questionnaire = relationship("Questionnaire", back_populates="review_sessions")
    answers = relationship("Answer", back_populates="session")

class Answer(Base):
    """Extracted clinical answer for a question (Agent 2 output)"""
    __tablename__ = 'answer'

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('review_session.id'), nullable=False)
    question_id = Column(Integer, ForeignKey('question.id'), nullable=False)
    answer_text = Column(Text)  # Extracted clinical data with specific values/units
    confidence_score = Column(Float)  # Confidence in extraction accuracy
    source_page_number = Column(Integer)  # Page where answer was found
    source_text_snippet = Column(Text)  # Exact text from document
    source_type = Column(String(20))  # text, visual, both
    reasoning = Column(Text)  # Detailed explanation of extraction and medical rationale
    
    # Clinical decision fields (Agent 3 evaluation)
    meets_requirement = Column(String(20))  # Yes, No, Partial, Missing
    clinical_significance = Column(String(20))  # Critical, Important, Supportive, Informational
    risk_assessment = Column(String(20))  # Low, Medium, High
    clinical_rationale = Column(Text)  # Evidence-based medical reasoning
    policy_impact = Column(String(50))  # Supports approval, Neutral, Concerns for approval, Contraindication
    additional_info_needed = Column(Text)  # Specific clinical information required

    session = relationship("ReviewSession", back_populates="answers")
    question = relationship("Question", back_populates="answers")