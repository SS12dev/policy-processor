"""
Document Verifier - Phase 7: Verification & Quality Check

This module compares generated outputs (policies, decision trees) against the source document
to detect quality issues that need refinement:
1. Duplicate policies (same content, different IDs)
2. Missing policies (chunks not extracted)
3. Incomplete decision trees (missing critical questions)
4. Hierarchy structure issues (too many roots, insufficient depth)
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from models.schemas import SubPolicy, DecisionTree, PolicyCondition
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DuplicatePair:
    """Represents a pair of duplicate policies."""
    policy_a_id: str
    policy_b_id: str
    title_similarity: float
    condition_similarity: float
    same_parent: bool
    same_level: bool
    confidence: float
    reason: str


@dataclass
class CoverageIssue:
    """Represents a SubPolicy coverage issue."""
    chunk_id: str
    issue_type: str  # 'missing', 'over_extracted', 'under_extracted'
    policy_count: int
    expected_count: int
    policies: List[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class TreeCompletenessIssue:
    """Represents a decision tree completeness issue."""
    policy_id: str
    tree_id: str
    total_conditions: int
    conditions_covered: int
    completeness_score: float
    missing_conditions: List[Dict[str, Any]] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class HierarchyIssue:
    """Represents a hierarchy structure issue."""
    issue_type: str  # 'too_many_policies', 'too_many_roots', 'insufficient_depth', 'orphaned_policies'
    severity: str  # 'high', 'medium', 'low'
    current_value: int
    target_value: int
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


@dataclass
class VerificationReport:
    """Complete verification report with all detected issues."""
    total_policies: int
    total_trees: int
    total_issues: int
    needs_refinement: bool
    confidence: float
    
    duplicate_policies: List[DuplicatePair] = field(default_factory=list)
    coverage_issues: List[CoverageIssue] = field(default_factory=list)
    tree_completeness_issues: List[TreeCompletenessIssue] = field(default_factory=list)
    hierarchy_issues: List[HierarchyIssue] = field(default_factory=list)
    
    summary: Dict[str, Any] = field(default_factory=dict)


class DocumentVerifier:
    """
    Verifies the quality of extracted policies and generated decision trees
    by comparing against the source document.
    """
    
    def __init__(self, target_metrics: Optional[Dict[str, int]] = None):
        """
        Initialize the document verifier.
        
        Args:
            target_metrics: Target metrics for validation (optional)
        """
        self.target_metrics = target_metrics or {
            'min_policies': 5,
            'max_policies': 7,
            'min_roots': 3,
            'max_roots': 4,
            'min_depth': 2,
            'max_depth': 3,
            'min_tree_completeness': 0.80,
            'max_validation_issues': 40
        }
        
        # Stop words for similarity calculation
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'
        }
    
    def verify_all(
        self,
        document: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        policies: List[SubPolicy],
        trees: List[DecisionTree],
        validation_results: Dict[str, Any]
    ) -> VerificationReport:
        """
        Run all verification checks and return a comprehensive report.
        
        Args:
            document: Original parsed document
            chunks: Document chunks
            policies: Extracted policies
            trees: Generated decision trees
            validation_results: Results from validation node
            
        Returns:
            VerificationReport with all detected issues
        """
        logger.info("=" * 80)
        logger.info("STARTING VERIFICATION & QUALITY CHECK")
        logger.info("=" * 80)
        
        report = VerificationReport(
            total_policies=len(policies),
            total_trees=len(trees),
            total_issues=0,
            needs_refinement=False,
            confidence=0.0
        )
        
        # Check 1: Duplicate SubPolicy Detection
        logger.info("\n[CHECK 1] Detecting duplicate policies...")
        report.duplicate_policies = self._detect_duplicate_policies(policies)
        logger.info(f"Found {len(report.duplicate_policies)} duplicate SubPolicy pairs")
        
        # Check 2: SubPolicy Coverage Analysis
        logger.info("\n[CHECK 2] Analyzing SubPolicy coverage...")
        report.coverage_issues = self._analyze_policy_coverage(chunks, policies)
        logger.info(f"Found {len(report.coverage_issues)} coverage issues")
        
        # Check 3: Decision Tree Completeness
        logger.info("\n[CHECK 3] Verifying decision tree completeness...")
        report.tree_completeness_issues = self._verify_tree_completeness(policies, trees)
        logger.info(f"Found {len(report.tree_completeness_issues)} tree completeness issues")
        
        # Check 4: Hierarchy Validation
        logger.info("\n[CHECK 4] Validating hierarchy structure...")
        report.hierarchy_issues = self._validate_hierarchy(policies)
        logger.info(f"Found {len(report.hierarchy_issues)} hierarchy issues")
        
        # Calculate total issues and confidence
        report.total_issues = (
            len(report.duplicate_policies) +
            len(report.coverage_issues) +
            len(report.tree_completeness_issues) +
            len(report.hierarchy_issues)
        )
        
        # Calculate confidence score (0-1)
        report.confidence = self._calculate_confidence(report, validation_results)
        
        # Determine if refinement needed
        report.needs_refinement = self._should_refine(report)
        
        # Create summary
        report.summary = self._create_summary(report)
        
        # Log summary
        self._log_verification_summary(report)
        
        return report
    
    def _detect_duplicate_policies(self, policies: List[SubPolicy]) -> List[DuplicatePair]:
        """
        Detect policies that are duplicates based on title and condition similarity.
        
        Uses multi-signal approach:
        - High title similarity (>90%)
        - Same parent_id
        - High condition overlap (>80%)
        - Same level
        """
        duplicates = []
        
        for i, policy_a in enumerate(policies):
            for j, policy_b in enumerate(policies[i+1:], i+1):
                # Calculate title similarity
                title_sim = self._calculate_text_similarity(
                    policy_a.title,
                    policy_b.title
                )
                
                # Check parent and level
                same_parent = policy_a.parent_id == policy_b.parent_id
                same_level = policy_a.level == policy_b.level
                
                # Calculate condition similarity
                condition_sim = self._calculate_condition_similarity(
                    policy_a.conditions,
                    policy_b.conditions
                )
                
                # Determine if duplicate based on multiple signals
                is_duplicate = False
                reason = ""
                
                if title_sim > 0.95 and same_parent:
                    is_duplicate = True
                    reason = "exact_title_match_same_parent"
                elif title_sim > 0.85 and same_parent and same_level:
                    is_duplicate = True
                    reason = "high_title_similarity_same_parent_level"
                elif title_sim > 0.80 and condition_sim > 0.80 and same_parent:
                    is_duplicate = True
                    reason = "high_title_and_condition_similarity"
                
                if is_duplicate:
                    duplicates.append(DuplicatePair(
                        policy_a_id=policy_a.policy_id,
                        policy_b_id=policy_b.policy_id,
                        title_similarity=title_sim,
                        condition_similarity=condition_sim,
                        same_parent=same_parent,
                        same_level=same_level,
                        confidence=(title_sim + condition_sim) / 2,
                        reason=reason
                    ))
                    
                    logger.warning(
                        f"Duplicate detected: '{policy_a.policy_id}' <-> '{policy_b.policy_id}' "
                        f"(title_sim={title_sim:.2f}, cond_sim={condition_sim:.2f}, reason={reason})"
                    )
        
        return duplicates
    
    def _analyze_policy_coverage(
        self,
        chunks: List[Dict[str, Any]],
        policies: List[SubPolicy]
    ) -> List[CoverageIssue]:
        """
        Analyze whether all significant chunks were extracted into policies.
        Detect over-extraction, under-extraction, or missing chunks.
        """
        issues = []
        
        # Count policies per chunk
        chunk_policy_map = {}
        for SubPolicy in policies:
            chunk_id = getattr(SubPolicy, 'source_chunk_id', None)
            if chunk_id:
                if chunk_id not in chunk_policy_map:
                    chunk_policy_map[chunk_id] = []
                chunk_policy_map[chunk_id].append(SubPolicy.policy_id)
        
        # Analyze each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            policy_count = len(chunk_policy_map.get(chunk_id, []))
            
            # Estimate expected SubPolicy count based on chunk content
            expected_count = self._estimate_expected_policies(chunk)
            
            # Check for issues
            if policy_count == 0 and expected_count > 0:
                issues.append(CoverageIssue(
                    chunk_id=chunk_id,
                    issue_type='missing',
                    policy_count=0,
                    expected_count=expected_count,
                    recommendation=f"Re-extract chunk {chunk_id} with lower confidence threshold"
                ))
            elif policy_count > expected_count * 1.5:
                issues.append(CoverageIssue(
                    chunk_id=chunk_id,
                    issue_type='over_extracted',
                    policy_count=policy_count,
                    expected_count=expected_count,
                    policies=chunk_policy_map[chunk_id],
                    recommendation=f"Consider merging policies from chunk {chunk_id}"
                ))
        
        return issues
    
    def _verify_tree_completeness(
        self,
        policies: List[SubPolicy],
        trees: List[DecisionTree]
    ) -> List[TreeCompletenessIssue]:
        """
        Verify that decision trees cover all SubPolicy conditions.
        Check if questions correspond to conditions.
        """
        issues = []
        
        # Create SubPolicy map for quick lookup
        policy_map = {p.policy_id: p for p in policies}
        
        for tree in trees:
            SubPolicy = policy_map.get(tree.policy_id)
            if not SubPolicy:
                continue
            
            # Get all conditions
            total_conditions = len(SubPolicy.conditions)
            if total_conditions == 0:
                continue
            
            # Get all questions from tree
            tree_questions = self._extract_tree_questions(tree)
            
            # Count covered conditions
            conditions_covered = 0
            missing_conditions = []
            
            for condition in SubPolicy.conditions:
                covered = self._is_condition_covered(condition, tree_questions)
                if covered:
                    conditions_covered += 1
                else:
                    missing_conditions.append({
                        'condition': condition.description,
                        'confidence': getattr(condition, 'confidence_score', 0.0),
                        'suggested_question': self._generate_question_from_condition(condition)
                    })
            
            # Calculate completeness score
            completeness_score = conditions_covered / total_conditions if total_conditions > 0 else 1.0
            
            # Check if below threshold
            if completeness_score < self.target_metrics['min_tree_completeness']:
                issues.append(TreeCompletenessIssue(
                    policy_id=SubPolicy.policy_id,
                    tree_id=tree.tree_id,
                    total_conditions=total_conditions,
                    conditions_covered=conditions_covered,
                    completeness_score=completeness_score,
                    missing_conditions=missing_conditions,
                    recommendation=f"Add {len(missing_conditions)} missing question(s) to improve completeness"
                ))
                
                logger.warning(
                    f"Tree '{tree.tree_id}' for SubPolicy '{SubPolicy.policy_id}' has low completeness: "
                    f"{completeness_score:.1%} ({conditions_covered}/{total_conditions} conditions covered)"
                )
        
        return issues
    
    def _validate_hierarchy(self, policies: List[SubPolicy]) -> List[HierarchyIssue]:
        """
        Validate hierarchy structure against target metrics.
        Check total policies, root count, depth, orphans.
        """
        issues = []
        
        # Count metrics
        total_policies = len(policies)
        root_policies = len([p for p in policies if p.parent_id is None])
        max_depth = self._calculate_max_depth(policies)
        
        # Check 1: Total SubPolicy count
        if total_policies > self.target_metrics['max_policies']:
            issues.append(HierarchyIssue(
                issue_type='too_many_policies',
                severity='high',
                current_value=total_policies,
                target_value=self.target_metrics['max_policies'],
                details={'excess': total_policies - self.target_metrics['max_policies']},
                recommendation=f"Merge similar policies to reduce from {total_policies} to {self.target_metrics['max_policies']}"
            ))
        
        # Check 2: Root SubPolicy count
        if root_policies > self.target_metrics['max_roots']:
            issues.append(HierarchyIssue(
                issue_type='too_many_roots',
                severity='high',
                current_value=root_policies,
                target_value=self.target_metrics['max_roots'],
                details={'excess': root_policies - self.target_metrics['max_roots']},
                recommendation=f"Strengthen semantic grouping to reduce roots from {root_policies} to {self.target_metrics['max_roots']}"
            ))
        
        # Check 3: Hierarchy depth
        if max_depth < self.target_metrics['min_depth']:
            issues.append(HierarchyIssue(
                issue_type='insufficient_depth',
                severity='medium',
                current_value=max_depth,
                target_value=self.target_metrics['min_depth'],
                details={'deficit': self.target_metrics['min_depth'] - max_depth},
                recommendation=f"Create deeper hierarchy by identifying more parent-child relationships (depth {max_depth} → {self.target_metrics['min_depth']})"
            ))
        
        # Check 4: Orphaned policies
        orphans = self._find_orphaned_policies(policies)
        if orphans:
            issues.append(HierarchyIssue(
                issue_type='orphaned_policies',
                severity='low',
                current_value=len(orphans),
                target_value=0,
                details={'orphan_ids': [p.policy_id for p in orphans]},
                recommendation=f"Assign parent policies or promote to root for {len(orphans)} orphaned policies"
            ))
        
        return issues
    
    # ========================
    # Helper Methods
    # ========================
    
    def _calculate_text_similarity(self, text_a: str, text_b: str) -> float:
        """Calculate Jaccard similarity between two text strings."""
        if not text_a or not text_b:
            return 0.0
        
        # Tokenize and normalize
        tokens_a = set(self._tokenize(text_a.lower()))
        tokens_b = set(self._tokenize(text_b.lower()))
        
        # Remove stop words
        tokens_a = tokens_a - self.stop_words
        tokens_b = tokens_b - self.stop_words
        
        if not tokens_a or not tokens_b:
            return 0.0
        
        # Jaccard similarity
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_condition_similarity(
        self,
        conditions_a: List[PolicyCondition],
        conditions_b: List[PolicyCondition]
    ) -> float:
        """Calculate similarity between two condition lists."""
        if not conditions_a or not conditions_b:
            return 0.0
        
        # Extract condition texts
        texts_a = [c.description for c in conditions_a if hasattr(c, 'description')]
        texts_b = [c.description for c in conditions_b if hasattr(c, 'description')]
        
        if not texts_a or not texts_b:
            return 0.0
        
        # Calculate pairwise similarities and take max average
        similarities = []
        for text_a in texts_a:
            max_sim = max(
                [self._calculate_text_similarity(text_a, text_b) for text_b in texts_b],
                default=0.0
            )
            similarities.append(max_sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return re.findall(r'\w+', text)
    
    def _estimate_expected_policies(self, chunk: Dict[str, Any]) -> int:
        """Estimate expected number of policies from a chunk."""
        # Simple heuristic based on chunk content
        content = chunk.get('content', '')
        
        # Count section markers
        section_markers = len(re.findall(r'\n[A-Z][A-Z\s]+\n', content))
        
        # Count bullet points / numbered lists
        list_markers = len(re.findall(r'\n\s*[\d\-\•]\s+', content))
        
        # Estimate: 1 SubPolicy per major section, plus 0.5 for each 5 list items
        estimated = section_markers + (list_markers // 5) * 0.5
        
        return max(1, int(estimated))
    
    def _extract_tree_questions(self, tree: DecisionTree) -> List[str]:
        """Extract all question texts from a decision tree."""
        questions = []
        
        def traverse(node):
            if hasattr(node, 'question') and node.question:
                questions.append(node.question.text if hasattr(node.question, 'text') else str(node.question))
            if hasattr(node, 'children'):
                for child in node.children:
                    traverse(child)
        
        # Start from root
        if hasattr(tree, 'root_node'):
            traverse(tree.root_node)
        
        return questions
    
    def _is_condition_covered(self, condition: PolicyCondition, questions: List[str]) -> bool:
        """Check if a condition is covered by tree questions."""
        if not questions:
            return False
        
        # Extract keywords from condition
        condition_keywords = set(self._tokenize(condition.description.lower())) - self.stop_words
        
        # Check if any question covers this condition (>50% keyword overlap)
        for question in questions:
            question_keywords = set(self._tokenize(question.lower())) - self.stop_words
            if not question_keywords:
                continue
            
            overlap = len(condition_keywords & question_keywords) / len(condition_keywords)
            if overlap > 0.5:
                return True
        
        return False
    
    def _generate_question_from_condition(self, condition: PolicyCondition) -> str:
        """Generate a suggested question from a condition."""
        # Simple heuristic: convert condition to question format
        text = condition.description.strip()
        
        # If already a question, return as-is
        if text.endswith('?'):
            return text
        
        # Try to convert to question
        if 'BMI' in text or 'age' in text.lower():
            return f"What is your {text.lower().split()[0]}?"
        elif 'have' in text.lower() or 'has' in text.lower():
            return f"Do you {text.lower().replace('has ', 'have ').replace('the patient ', '')}?"
        else:
            return f"Does the patient meet the following requirement: {text}?"
    
    def _calculate_max_depth(self, policies: List[SubPolicy]) -> int:
        """Calculate maximum depth of SubPolicy hierarchy."""
        if not policies:
            return 0
        
        # Build parent-child map
        children_map = {}
        for policy in policies:
            if policy.parent_id:
                if policy.parent_id not in children_map:
                    children_map[policy.parent_id] = []
                children_map[policy.parent_id].append(policy)
        
        # Find max depth from roots
        roots = [p for p in policies if p.parent_id is None]
        max_depth = 0
        
        def get_depth(policy: SubPolicy, current_depth: int) -> int:
            children = children_map.get(policy.policy_id, [])
            if not children:
                return current_depth
            return max(get_depth(child, current_depth + 1) for child in children)
        
        for root in roots:
            depth = get_depth(root, 0)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _find_orphaned_policies(self, policies: List[SubPolicy]) -> List[SubPolicy]:
        """Find policies that reference non-existent parents."""
        policy_ids = {p.policy_id for p in policies}
        orphans = []
        
        for SubPolicy in policies:
            if SubPolicy.parent_id and SubPolicy.parent_id not in policy_ids:
                orphans.append(SubPolicy)
        
        return orphans
    
    def _calculate_confidence(
        self,
        report: VerificationReport,
        validation_results: Dict[str, Any]
    ) -> float:
        """
        Calculate overall confidence score based on issues found.
        
        Confidence decreases with:
        - Number of duplicate policies
        - Number of hierarchy issues
        - Number of tree completeness issues
        - Severity of issues
        """
        base_confidence = validation_results.get('confidence', 0.9)
        
        # Penalties
        duplicate_penalty = len(report.duplicate_policies) * 0.05
        hierarchy_penalty = sum(
            0.10 if issue.severity == 'high' else 0.05 if issue.severity == 'medium' else 0.02
            for issue in report.hierarchy_issues
        )
        tree_penalty = len(report.tree_completeness_issues) * 0.03
        coverage_penalty = len(report.coverage_issues) * 0.02
        
        total_penalty = duplicate_penalty + hierarchy_penalty + tree_penalty + coverage_penalty
        
        confidence = max(0.0, base_confidence - total_penalty)
        
        return confidence
    
    def _should_refine(self, report: VerificationReport) -> bool:
        """
        Determine if refinement is needed based on issues found.
        
        Refinement needed if:
        - Any duplicate policies found
        - High-severity hierarchy issues
        - Confidence < 0.95
        """
        # Critical issues that always trigger refinement
        if report.duplicate_policies:
            return True
        
        # High-severity hierarchy issues
        high_severity_issues = [
            issue for issue in report.hierarchy_issues
            if issue.severity == 'high'
        ]
        if high_severity_issues:
            return True
        
        # Low confidence
        if report.confidence < 0.95:
            return True
        
        return False
    
    def _create_summary(self, report: VerificationReport) -> Dict[str, Any]:
        """Create a summary dictionary for the report."""
        return {
            'total_issues': report.total_issues,
            'duplicate_count': len(report.duplicate_policies),
            'coverage_issues': len(report.coverage_issues),
            'tree_issues': len(report.tree_completeness_issues),
            'hierarchy_issues': len(report.hierarchy_issues),
            'confidence': report.confidence,
            'needs_refinement': report.needs_refinement,
            'critical_issues': len([
                issue for issue in report.hierarchy_issues
                if issue.severity == 'high'
            ]) + len(report.duplicate_policies)
        }
    
    def _log_verification_summary(self, report: VerificationReport):
        """Log a comprehensive verification summary."""
        logger.info("=" * 80)
        logger.info("VERIFICATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Policies: {report.total_policies}")
        logger.info(f"Total Trees: {report.total_trees}")
        logger.info(f"Total Issues: {report.total_issues}")
        logger.info(f"Confidence: {report.confidence:.1%}")
        logger.info(f"Needs Refinement: {'YES' if report.needs_refinement else 'NO'}")
        
        if report.duplicate_policies:
            logger.warning(f"\n⚠️  DUPLICATE POLICIES: {len(report.duplicate_policies)}")
            for dup in report.duplicate_policies:
                logger.warning(f"  - {dup.policy_a_id} <-> {dup.policy_b_id} (confidence: {dup.confidence:.1%})")
        
        if report.hierarchy_issues:
            logger.warning(f"\n⚠️  HIERARCHY ISSUES: {len(report.hierarchy_issues)}")
            for issue in report.hierarchy_issues:
                logger.warning(f"  - {issue.issue_type}: {issue.current_value} (target: {issue.target_value}) [{issue.severity}]")
        
        if report.tree_completeness_issues:
            logger.warning(f"\n⚠️  TREE COMPLETENESS ISSUES: {len(report.tree_completeness_issues)}")
            for issue in report.tree_completeness_issues:
                logger.warning(f"  - {issue.policy_id}: {issue.completeness_score:.1%} completeness")
        
        if report.coverage_issues:
            logger.warning(f"\n⚠️  COVERAGE ISSUES: {len(report.coverage_issues)}")
            for issue in report.coverage_issues:
                logger.warning(f"  - {issue.chunk_id}: {issue.issue_type}")
        
        logger.info("=" * 80)
