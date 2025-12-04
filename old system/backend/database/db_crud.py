from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select
from backend.database.db_schema import (
    FileStorage, Policy, Questionnaire, Question,
    Application, ReviewSession, Answer
)
from typing import List, Optional
from datetime import datetime

# FileStorage CRUD
def create_file_storage(db: Session, filename: str, file_data: bytes, file_type: str = 'application/pdf') -> FileStorage:
    """Store file in database"""
    file_storage = FileStorage(
        file_name=filename,
        file_type=file_type,
        content=file_data
    )
    db.add(file_storage)
    db.commit()
    db.refresh(file_storage)
    return file_storage

def get_file_storage(db: Session, file_id: int) -> Optional[FileStorage]:
    """Get file by ID"""
    return db.execute(select(FileStorage).where(FileStorage.id == file_id)).scalar_one_or_none()

# Policy CRUD
def create_policy(db: Session, name: str, file_storage_id: int, status: str = 'processed') -> Policy:
    """Create policy from uploaded file"""
    policy = Policy(file_id=file_storage_id, name=name)
    db.add(policy)
    db.commit()
    db.refresh(policy)
    return policy

def get_all_policies(db: Session) -> List[Policy]:
    """Get all policies"""
    return db.execute(select(Policy)).scalars().all()

def get_policies(db: Session) -> List[Policy]:
    """Get all policies (alias)"""
    return get_all_policies(db)

def get_policy(db: Session, policy_id: int) -> Optional[Policy]:
    """Get policy by ID"""
    return db.execute(select(Policy).where(Policy.id == policy_id)).scalar_one_or_none()

# Questionnaire CRUD
def create_questionnaire(db: Session, policy_id: int, approved: bool = False, name: str = "Generated Questionnaire",
                        total_questions: int = None, complexity_level: str = None, coverage_areas: list = None,
                        procedure_type: str = None, target_population: str = None, 
                        llm_input_tokens: int = 0, llm_output_tokens: int = 0, 
                        llm_total_cost: float = 0.0, llm_model: str = None) -> Questionnaire:
    """Create enhanced medical questionnaire for policy with cost tracking"""
    questionnaire = Questionnaire(
        policy_id=policy_id, 
        name=name, 
        is_approved=approved,
        total_questions=total_questions,
        complexity_level=complexity_level,
        coverage_areas=coverage_areas,
        procedure_type=procedure_type,
        target_population=target_population,
        llm_input_tokens=llm_input_tokens,
        llm_output_tokens=llm_output_tokens,
        llm_total_cost=llm_total_cost,
        llm_model=llm_model
    )
    db.add(questionnaire)
    db.commit()
    db.refresh(questionnaire)
    return questionnaire

def get_questionnaires(db: Session, approved_only: bool = False) -> List[Questionnaire]:
    """Get questionnaires, optionally filtered by approval status"""
    query = select(Questionnaire)
    if approved_only:
        query = query.where(Questionnaire.is_approved == True)
    return db.execute(query).scalars().all()

def get_questionnaires_by_policy(db: Session, policy_id: int) -> List[Questionnaire]:
    """Get all questionnaires for a policy"""
    return db.execute(
        select(Questionnaire).where(Questionnaire.policy_id == policy_id)
    ).scalars().all()

def get_questionnaire(db: Session, questionnaire_id: int) -> Optional[Questionnaire]:
    """Get questionnaire by ID"""
    return db.execute(select(Questionnaire).where(Questionnaire.id == questionnaire_id)).scalar_one_or_none()

def get_questionnaire_by_id(db: Session, questionnaire_id: int) -> Optional[Questionnaire]:
    """Get questionnaire by ID (alias)"""
    return get_questionnaire(db, questionnaire_id)

def approve_questionnaire(db: Session, questionnaire_id: int) -> bool:
    """Approve questionnaire"""
    questionnaire = get_questionnaire(db, questionnaire_id)
    if questionnaire:
        questionnaire.is_approved = True
        questionnaire.approved_at = datetime.utcnow()
        db.commit()
        return True
    return False

# Question CRUD
def create_question(db: Session, questionnaire_id: int, question_text: str,
                   source_text_snippet: str = None, order_index: int = None,
                   question_id: str = None, question_type: str = None,
                   answer_options: dict = None, page_number: int = None,
                   line_reference: str = None, criterion_type: str = None,
                   priority_level: str = None, approval_impact: str = None,
                   validation_rules: dict = None,
                   chunk_id: int = None, consolidation_notes: str = None) -> Question:
    """Create enhanced medical question for questionnaire"""
    question = Question(
        questionnaire_id=questionnaire_id,
        question_id=question_id,
        question_text=question_text,
        question_type=question_type,
        answer_options=answer_options,
        source_text_snippet=source_text_snippet,
        page_number=page_number,
        line_reference=line_reference,
        criterion_type=criterion_type,
        priority_level=priority_level,
        approval_impact=approval_impact,
        validation_rules=validation_rules,
        order_index=order_index,
        chunk_id=chunk_id,
        consolidation_notes=consolidation_notes
    )
    db.add(question)
    db.commit()
    db.refresh(question)
    return question

def get_questions_by_questionnaire(db: Session, questionnaire_id: int) -> List[Question]:
    """Get all questions for a questionnaire"""
    return db.execute(
        select(Question)
        .where(Question.questionnaire_id == questionnaire_id)
        .order_by(Question.order_index)
    ).scalars().all()

# Application CRUD
def create_application(db: Session, patient_name: str, file_storage_id: int,
                      status: str = 'processed', patient_dob: str = None,
                      patient_id: str = None, insurance_info: str = None,
                      primary_diagnosis: str = None, bmi: str = None,
                      weight: str = None, requesting_physician: str = None,
                      clinical_details: dict = None, patient_summary: str = None,
                      llm_input_tokens: int = 0, llm_output_tokens: int = 0,
                      llm_total_cost: float = 0.0, llm_model: str = None) -> Application:
    """Create enhanced medical application from uploaded file with cost tracking"""
    application = Application(
        file_id=file_storage_id,
        patient_name=patient_name,
        patient_dob=patient_dob,
        patient_id=patient_id,
        insurance_info=insurance_info,
        primary_diagnosis=primary_diagnosis,
        bmi=bmi,
        weight=weight,
        requesting_physician=requesting_physician,
        clinical_details=clinical_details,
        patient_summary=patient_summary,
        llm_input_tokens=llm_input_tokens,
        llm_output_tokens=llm_output_tokens,
        llm_total_cost=llm_total_cost,
        llm_model=llm_model
    )
    db.add(application)
    db.commit()
    db.refresh(application)
    return application

def get_applications(db: Session) -> List[Application]:
    """Get all applications"""
    return db.execute(select(Application)).scalars().all()

def get_application(db: Session, application_id: int) -> Optional[Application]:
    """Get application by ID"""
    return db.execute(select(Application).where(Application.id == application_id)).scalar_one_or_none()

# ReviewSession CRUD
def create_review_session(db: Session, application_id: int, questionnaire_id: int, status: str = 'Pending',
                         final_recommendation: str = None, confidence_score: float = None,
                         primary_clinical_rationale: str = None, medical_necessity_assessment: str = None,
                         supporting_clinical_factors: list = None, clinical_concerns: list = None,
                         required_actions: list = None, review_priority: str = None,
                         clinical_notes: str = None, overall_risk_level: str = None,
                         detailed_reasoning: str = None, decision_factors: dict = None,
                         llm_input_tokens: int = 0, llm_output_tokens: int = 0,
                         llm_total_cost: float = 0.0, llm_model: str = None) -> ReviewSession:
    """Create enhanced clinical review session with detailed reasoning and cost tracking"""
    session = ReviewSession(
        application_id=application_id,
        questionnaire_id=questionnaire_id,
        status=status,
        final_recommendation=final_recommendation,
        confidence_score=confidence_score,
        primary_clinical_rationale=primary_clinical_rationale,
        medical_necessity_assessment=medical_necessity_assessment,
        supporting_clinical_factors=supporting_clinical_factors,
        clinical_concerns=clinical_concerns,
        required_actions=required_actions,
        review_priority=review_priority,
        clinical_notes=clinical_notes,
        overall_risk_level=overall_risk_level,
        detailed_reasoning=detailed_reasoning,
        decision_factors=decision_factors,
        llm_input_tokens=llm_input_tokens,
        llm_output_tokens=llm_output_tokens,
        llm_total_cost=llm_total_cost,
        llm_model=llm_model
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

def get_all_review_sessions(db: Session) -> List[ReviewSession]:
    """Get all review sessions"""
    return db.execute(select(ReviewSession)).scalars().all()

def get_review_sessions(db: Session) -> List[ReviewSession]:
    """Get all review sessions (alias)"""
    return get_all_review_sessions(db)

def get_review_session(db: Session, session_id: int) -> Optional[ReviewSession]:
    """Get review session by ID"""
    return db.execute(select(ReviewSession).where(ReviewSession.id == session_id)).scalar_one_or_none()

def update_review_session_status(db: Session, session_id: int, status: str, recommendation: str = None) -> bool:
    """Update review session status and recommendation"""
    session = get_review_session(db, session_id)
    if session:
        session.status = status
        if recommendation:
            session.final_recommendation = recommendation
        if status == 'Completed' or status == 'completed':
            session.completed_at = datetime.utcnow()
        db.commit()
        return True
    return False

def update_review_session_recommendation(db: Session, session_id: int,
                                       recommendation: str, rationale: str) -> bool:
    """Update review session with final recommendation"""
    session = get_review_session(db, session_id)
    if session:
        session.final_recommendation = recommendation
        session.final_rationale = rationale
        session.status = "In Review"
        db.commit()
        return True
    return False

# Answer CRUD
def create_answer(db: Session, review_session_id: int, question_id: int,
                 answer_text: str = None, source_page_number: int = None,
                 confidence_score: float = None, source_text_snippet: str = None,
                 source_type: str = None, reasoning: str = None,
                 meets_requirement: str = None, clinical_significance: str = None,
                 risk_assessment: str = None, clinical_rationale: str = None,
                 policy_impact: str = None, additional_info_needed: str = None) -> Answer:
    """Create enhanced clinical answer for question in review session"""
    answer = Answer(
        session_id=review_session_id,
        question_id=question_id,
        answer_text=answer_text,
        source_page_number=source_page_number,
        confidence_score=confidence_score,
        source_text_snippet=source_text_snippet,
        source_type=source_type,
        reasoning=reasoning,
        meets_requirement=meets_requirement,
        clinical_significance=clinical_significance,
        risk_assessment=risk_assessment,
        clinical_rationale=clinical_rationale,
        policy_impact=policy_impact,
        additional_info_needed=additional_info_needed
    )
    db.add(answer)
    db.commit()
    db.refresh(answer)
    return answer

def get_answers_by_session(db: Session, session_id: int) -> List[Answer]:
    """Get all answers for a review session"""
    return db.execute(
        select(Answer).where(Answer.session_id == session_id)
    ).scalars().all()

# Additional Question CRUD operations for dynamic editing
def delete_question(db: Session, question_id: int) -> bool:
    """Delete a question from the database"""
    question = db.execute(select(Question).where(Question.id == question_id)).scalar_one_or_none()
    if question:
        db.delete(question)
        db.commit()
        return True
    return False

def update_question(db: Session, question_id: int, question_text: str = None,
                   question_type: str = None, criterion_type: str = None,
                   answer_options: dict = None, source_text_snippet: str = None,
                   page_number: int = None) -> Question:
    """Update an existing question"""
    question = db.execute(select(Question).where(Question.id == question_id)).scalar_one_or_none()
    if question:
        if question_text is not None:
            question.question_text = question_text
        if question_type is not None:
            question.question_type = question_type
        if criterion_type is not None:
            question.criterion_type = criterion_type
        if answer_options is not None:
            question.answer_options = answer_options
        if source_text_snippet is not None:
            question.source_text_snippet = source_text_snippet
        if page_number is not None:
            question.page_number = page_number
        
        db.commit()
        db.refresh(question)
        return question
    return None

def reorder_questions(db: Session, questionnaire_id: int) -> None:
    """Reorder questions after deletion to maintain sequential order_index"""
    questions = db.execute(
        select(Question)
        .where(Question.questionnaire_id == questionnaire_id)
        .order_by(Question.order_index)
    ).scalars().all()
    
    for i, question in enumerate(questions, 1):
        question.order_index = i
    
    db.commit()

def get_question_by_id(db: Session, question_id: int) -> Optional[Question]:
    """Get a single question by ID"""
    return db.execute(select(Question).where(Question.id == question_id)).scalar_one_or_none()

def update_answer(db: Session, answer_id: int, answer_text: str = None, clinical_rationale: str = None, 
                 reasoning: str = None, confidence_score: float = None, source_page_number: int = None) -> bool:
    """Update an existing answer"""
    try:
        answer = db.execute(select(Answer).where(Answer.id == answer_id)).scalar_one_or_none()
        if not answer:
            return False
            
        if answer_text is not None:
            answer.answer_text = answer_text
        if clinical_rationale is not None:
            answer.clinical_rationale = clinical_rationale
        if reasoning is not None:
            answer.reasoning = reasoning
        if confidence_score is not None:
            answer.confidence_score = confidence_score
        if source_page_number is not None:
            answer.source_page_number = source_page_number
            
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error updating answer: {e}")
        return False

def update_review_session(db: Session, session_id: int, final_recommendation: str = None, 
                         primary_clinical_rationale: str = None, confidence_score: float = None,
                         review_priority: str = None, status: str = None) -> bool:
    """Update a review session with decision details"""
    try:
        session = db.execute(select(ReviewSession).where(ReviewSession.id == session_id)).scalar_one_or_none()
        if not session:
            return False
            
        if final_recommendation is not None:
            session.final_recommendation = final_recommendation
        if primary_clinical_rationale is not None:
            session.primary_clinical_rationale = primary_clinical_rationale
        if confidence_score is not None:
            session.confidence_score = confidence_score
        if review_priority is not None:
            session.review_priority = review_priority
        if status is not None:
            session.status = status
            
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error updating review session: {e}")
        return False

# Additional enhanced CRUD functions for UI components
def get_all_applications(db: Session) -> List[Application]:
    """Get all applications with related data loaded efficiently"""
    return db.execute(
        select(Application)
        .options(selectinload(Application.file))
        .order_by(Application.submission_date.desc())
    ).scalars().all()

def get_application_by_id(db: Session, application_id: int) -> Optional[Application]:
    """Get a specific application by ID with related data loaded"""
    return db.execute(
        select(Application)
        .options(selectinload(Application.file))
        .where(Application.id == application_id)
    ).scalar_one_or_none()

def get_review_sessions_by_application(db: Session, application_id: int) -> List[ReviewSession]:
    """Get all review sessions for a specific application with related data"""
    return db.execute(
        select(ReviewSession)
        .options(
            selectinload(ReviewSession.questionnaire),
            selectinload(ReviewSession.answers)
        )
        .where(ReviewSession.application_id == application_id)
        .order_by(ReviewSession.created_at.desc())
    ).scalars().all()