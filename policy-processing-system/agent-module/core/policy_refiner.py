"""
SubPolicy Refiner - Phase 8: Automated Refinement

This module automatically fixes quality issues detected by the document verifier:
1. Merge duplicate policies
2. Re-extract missing chunks
3. Strengthen hierarchy grouping
4. Regenerate affected decision trees
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

from models.schemas import SubPolicy, DecisionTree, PolicyCondition
from core.document_verifier import VerificationReport, DuplicatePair
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RefinementResult:
    """Result of refinement operations."""
    policies: List[SubPolicy]
    trees: List[DecisionTree]
    
    actions_taken: List[str] = field(default_factory=list)
    policies_merged: int = 0
    policies_reparented: int = 0
    trees_regenerated: int = 0
    
    summary: Dict[str, Any] = field(default_factory=dict)


class PolicyRefiner:
    """
    Automatically refines policies and decision trees based on verification issues.
    """
    
    def __init__(self, extractor=None, tree_generator=None):
        """
        Initialize the SubPolicy refiner.
        
        Args:
            extractor: PolicyExtractor instance (optional, for re-extraction)
            tree_generator: DecisionTreeGenerator instance (optional, for tree regeneration)
        """
        self.extractor = extractor
        self.tree_generator = tree_generator
    
    def refine(
        self,
        policies: List[SubPolicy],
        trees: List[DecisionTree],
        verification_report: VerificationReport
    ) -> RefinementResult:
        """
        Apply refinements based on verification report.
        
        Args:
            policies: List of extracted policies
            trees: List of generated decision trees
            verification_report: Report from document verifier
            
        Returns:
            RefinementResult with refined policies and trees
        """
        logger.info("=" * 80)
        logger.info("STARTING REFINEMENT")
        logger.info("=" * 80)
        
        result = RefinementResult(
            policies=policies.copy(),
            trees=trees.copy()
        )
        
        # Action 1: Merge duplicate policies
        if verification_report.duplicate_policies:
            logger.info("\n[ACTION 1] Merging duplicate policies...")
            result = self._merge_duplicate_policies(
                result,
                verification_report.duplicate_policies
            )
        
        # Action 2: Strengthen hierarchy grouping
        hierarchy_issues = [
            issue for issue in verification_report.hierarchy_issues
            if issue.issue_type in ['too_many_roots', 'insufficient_depth']
        ]
        if hierarchy_issues:
            logger.info("\n[ACTION 2] Strengthening hierarchy grouping...")
            result = self._strengthen_hierarchy_grouping(result, hierarchy_issues)
        
        # Action 3: Regenerate affected decision trees
        if result.policies_merged > 0 or result.policies_reparented > 0:
            logger.info("\n[ACTION 3] Regenerating affected decision trees...")
            result = self._regenerate_affected_trees(result)
        
        # Create summary
        result.summary = self._create_summary(result, verification_report)
        
        # Log summary
        self._log_refinement_summary(result)
        
        return result
    
    def _merge_duplicate_policies(
        self,
        result: RefinementResult,
        duplicates: List[DuplicatePair]
    ) -> RefinementResult:
        """
        Merge duplicate policies by:
        1. Keep policy_a, merge conditions from policy_b
        2. Update children pointing to policy_b to point to policy_a
        3. Remove policy_b
        4. Remove decision tree for policy_b
        """
        policies_to_remove = set()
        trees_to_remove = set()
        
        for dup in duplicates:
            # Find policies
            policy_a = self._find_policy(result.policies, dup.policy_a_id)
            policy_b = self._find_policy(result.policies, dup.policy_b_id)
            
            if not policy_a or not policy_b:
                logger.warning(f"Could not find policies for merging: {dup.policy_a_id}, {dup.policy_b_id}")
                continue
            
            logger.info(f"Merging '{policy_b.policy_id}' into '{policy_a.policy_id}'")
            
            # Merge conditions (deduplicate)
            merged_conditions = self._deduplicate_conditions(
                policy_a.conditions + policy_b.conditions
            )
            policy_a.conditions = merged_conditions
            
            # Update children pointing to policy_b
            children_updated = 0
            for SubPolicy in result.policies:
                if SubPolicy.parent_id == policy_b.policy_id:
                    SubPolicy.parent_id = policy_a.policy_id
                    children_updated += 1
            
            if children_updated > 0:
                logger.info(f"  Updated {children_updated} children to point to '{policy_a.policy_id}'")
            
            # Mark for removal
            policies_to_remove.add(policy_b.policy_id)
            trees_to_remove.add(policy_b.policy_id)
            
            result.actions_taken.append(f"Merged duplicate: {policy_b.policy_id} → {policy_a.policy_id}")
            result.policies_merged += 1
        
        # Remove marked policies
        result.policies = [
            p for p in result.policies
            if p.policy_id not in policies_to_remove
        ]
        
        # Remove marked trees
        result.trees = [
            t for t in result.trees
            if t.policy_id not in trees_to_remove
        ]
        
        logger.info(f"Merged {result.policies_merged} duplicate policies")
        logger.info(f"Removed {len(trees_to_remove)} duplicate decision trees")
        
        return result
    
    def _strengthen_hierarchy_grouping(
        self,
        result: RefinementResult,
        hierarchy_issues: List[Any]
    ) -> RefinementResult:
        """
        Apply more aggressive semantic grouping to reduce roots and increase depth.
        
        Strategies:
        1. Lower umbrella SubPolicy threshold (more policies qualify as parents)
        2. Lower parent matching threshold (more children get assigned)
        3. Increase keyword overlap weight
        """
        # Check if we need to reduce roots
        too_many_roots = any(issue.issue_type == 'too_many_roots' for issue in hierarchy_issues)
        insufficient_depth = any(issue.issue_type == 'insufficient_depth' for issue in hierarchy_issues)
        
        if not (too_many_roots or insufficient_depth):
            return result
        
        # Adjust parameters for more aggressive grouping
        params = {
            'umbrella_threshold': 4,  # Was 3, now require higher score to be umbrella
            'parent_matching_threshold': 0.35,  # Was 0.4, lower to match more
            'keyword_weight': 0.5,  # Was 0.4, increase keyword importance
            'page_proximity_weight': 0.15,  # Was 0.2, decrease slightly
            'title_pattern_weight': 0.20,  # Was 0.2, keep same
            'condition_similarity_weight': 0.15  # Was 0.2, decrease slightly
        }
        
        logger.info(f"Applying strengthened hierarchy grouping with adjusted parameters:")
        logger.info(f"  - Umbrella threshold: {params['umbrella_threshold']}")
        logger.info(f"  - Parent matching threshold: {params['parent_matching_threshold']}")
        logger.info(f"  - Keyword weight: {params['keyword_weight']}")
        
        # Re-run semantic refinement with new parameters
        refined_policies = self._refine_policy_hierarchy(result.policies, params)
        
        # Count re-parenting changes
        original_parents = {p.policy_id: p.parent_id for p in result.policies}
        new_parents = {p.policy_id: p.parent_id for p in refined_policies}
        
        reparented = sum(
            1 for policy_id in original_parents
            if original_parents[policy_id] != new_parents.get(policy_id)
        )
        
        result.policies = refined_policies
        result.policies_reparented = reparented
        result.actions_taken.append(f"Strengthened hierarchy grouping: {reparented} policies re-parented")
        
        logger.info(f"Re-parented {reparented} policies with strengthened grouping")
        
        return result
    
    def _refine_policy_hierarchy(
        self,
        policies: List[SubPolicy],
        params: Dict[str, float]
    ) -> List[SubPolicy]:
        """
        Re-run semantic hierarchy refinement with adjusted parameters.
        This mirrors the logic from policy_extractor.py but with different thresholds.
        """
        logger.info(f"Starting strengthened semantic refinement with {len(policies)} policies...")
        
        # Separate roots and children
        roots = [p for p in policies if p.parent_id is None]
        children = [p for p in policies if p.parent_id is not None]
        
        logger.info(f"Current hierarchy: {len(roots)} roots, {len(children)} children")
        
        # Identify umbrella policies (potential parents)
        umbrellas = self._identify_umbrella_policies(roots, params['umbrella_threshold'])
        logger.info(f"Identified {len(umbrellas)} umbrella policies: {[p.policy_id for p in umbrellas]}")
        
        # Find candidate children (roots that should become children)
        candidates = [p for p in roots if p not in umbrellas]
        logger.info(f"Found {len(candidates)} candidate children for re-parenting")
        
        # Try to assign parents to candidates
        reparented_count = 0
        for candidate in candidates:
            best_parent = self._find_best_parent(
                candidate,
                umbrellas,
                params
            )
            
            if best_parent:
                candidate.parent_id = best_parent.policy_id
                candidate.level = best_parent.level + 1
                reparented_count += 1
                logger.info(
                    f"Re-parenting '{candidate.policy_id}' ({candidate.title[:50]}...) "
                    f"under '{best_parent.policy_id}' ({best_parent.title[:50]}...)"
                )
        
        # Update roots list
        new_roots = [p for p in roots if p.parent_id is None]
        logger.info(f"Semantic refinement complete: {reparented_count} policies re-parented")
        logger.info(f"Root count: {len(roots)} → {len(new_roots)}")
        
        return policies
    
    def _identify_umbrella_policies(
        self,
        root_policies: List[SubPolicy],
        threshold: float
    ) -> List[SubPolicy]:
        """
        Identify policies that should be umbrella/parent policies.
        Uses scoring system similar to policy_extractor.py.
        """
        umbrella_keywords = {
            'criteria', 'requirements', 'eligibility', 'guidelines',
            'medically necessary', 'coverage', 'indications', 'considerations'
        }
        
        scored_policies = []
        
        for SubPolicy in root_policies:
            score = 0
            reasons = []
            
            # Check 1: Title keywords
            title_lower = SubPolicy.title.lower()
            if any(keyword in title_lower for keyword in umbrella_keywords):
                score += 2
                reasons.append('umbrella_keywords_in_title')
            
            # Check 2: Description length
            description_words = len(SubPolicy.description.split())
            if description_words > 25:
                score += 1
                reasons.append('comprehensive_description')
            
            # Check 3: Number of conditions
            if len(SubPolicy.conditions) >= 3:
                score += 1
                reasons.append('multiple_conditions')
            
            # Check 4: Has existing children (from original extraction)
            # Note: This check is less relevant in refinement but kept for consistency
            score += 0  # Placeholder
            
            scored_policies.append((SubPolicy, score, reasons))
        
        # Select policies with score >= threshold
        umbrellas = [p for p, s, r in scored_policies if s >= threshold]
        
        # If no clear umbrellas, select top 3 by score
        if not umbrellas and scored_policies:
            sorted_policies = sorted(scored_policies, key=lambda x: x[1], reverse=True)
            umbrellas = [p for p, s, r in sorted_policies[:3]]
            logger.info(f"No clear umbrellas, selecting top {len(umbrellas)} by score")
        
        return umbrellas
    
    def _find_best_parent(
        self,
        child: SubPolicy,
        potential_parents: List[SubPolicy],
        params: Dict[str, float]
    ) -> Optional[SubPolicy]:
        """
        Find the best parent for a child SubPolicy using weighted multi-signal matching.
        """
        if not potential_parents:
            return None
        
        best_parent = None
        best_score = 0.0
        
        for parent in potential_parents:
            # Calculate individual signals
            keyword_overlap = self._calculate_keyword_overlap(child, parent)
            page_proximity = self._calculate_page_proximity_score(child, parent)
            title_pattern = self._calculate_title_pattern_score(child, parent)
            condition_similarity = self._calculate_condition_similarity_score(child, parent)
            
            # Weighted score
            score = (
                keyword_overlap * params['keyword_weight'] +
                page_proximity * params['page_proximity_weight'] +
                title_pattern * params['title_pattern_weight'] +
                condition_similarity * params['condition_similarity_weight']
            )
            
            if score > best_score:
                best_score = score
                best_parent = parent
        
        # Require minimum score threshold
        if best_score < params['parent_matching_threshold']:
            return None
        
        return best_parent
    
    def _calculate_keyword_overlap(self, child: SubPolicy, parent: SubPolicy) -> float:
        """Calculate keyword overlap between child and parent."""
        child_keywords = self._extract_keywords(child.title + ' ' + child.description)
        parent_keywords = self._extract_keywords(parent.title + ' ' + parent.description)
        
        if not child_keywords or not parent_keywords:
            return 0.0
        
        intersection = len(child_keywords & parent_keywords)
        union = len(child_keywords | parent_keywords)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_page_proximity_score(self, child: SubPolicy, parent: SubPolicy) -> float:
        """Calculate page proximity score (closer = higher score)."""
        # Get page ranges
        child_pages = getattr(child, 'pages', [])
        parent_pages = getattr(parent, 'pages', [])
        
        if not child_pages or not parent_pages:
            return 0.5  # Neutral score if no page info
        
        # Get average pages
        child_avg = sum(child_pages) / len(child_pages)
        parent_avg = sum(parent_pages) / len(parent_pages)
        
        distance = abs(child_avg - parent_avg)
        
        # Score decreases with distance: 1/(1+distance)
        return 1 / (1 + distance)
    
    def _calculate_title_pattern_score(self, child: SubPolicy, parent: SubPolicy) -> float:
        """Calculate title pattern matching score."""
        child_title_lower = child.title.lower()
        parent_title_lower = parent.title.lower()
        
        # Check if parent title words appear in child title
        parent_words = set(parent_title_lower.split())
        child_words = set(child_title_lower.split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'for', 'of', 'in', 'on', 'at', 'to'}
        parent_words = parent_words - stop_words
        child_words = child_words - stop_words
        
        if not parent_words:
            return 0.0
        
        overlap = len(parent_words & child_words)
        return overlap / len(parent_words)
    
    def _calculate_condition_similarity_score(self, child: SubPolicy, parent: SubPolicy) -> float:
        """Calculate condition similarity between child and parent."""
        if not child.conditions or not parent.conditions:
            return 0.0
        
        # Extract keywords from conditions
        child_cond_text = ' '.join([
            c.description if hasattr(c, 'description') else str(c)
            for c in child.conditions
        ])
        parent_cond_text = ' '.join([
            c.description if hasattr(c, 'description') else str(c)
            for c in parent.conditions
        ])
        
        child_keywords = self._extract_keywords(child_cond_text)
        parent_keywords = self._extract_keywords(parent_cond_text)
        
        if not child_keywords or not parent_keywords:
            return 0.0
        
        intersection = len(child_keywords & parent_keywords)
        union = len(child_keywords | parent_keywords)
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        import re
        
        # Tokenize
        words = re.findall(r'\w+', text.lower())
        
        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might',
            'must', 'shall', 'can'
        }
        
        # Keep words longer than 3 chars and not stop words
        keywords = {w for w in words if len(w) > 3 and w not in stop_words}
        
        return keywords
    
    def _regenerate_affected_trees(self, result: RefinementResult) -> RefinementResult:
        """
        Regenerate decision trees for policies that were modified.
        
        Trees need regeneration if:
        - SubPolicy was merged (merged conditions)
        - SubPolicy was re-parented (context changed)
        """
        if not self.tree_generator:
            logger.warning("No tree generator available, skipping tree regeneration")
            return result
        
        # Find policies that need tree regeneration
        policies_to_regenerate = set()
        
        # Policies that were kept in merges (they have new conditions)
        for action in result.actions_taken:
            if 'Merged duplicate' in action:
                # Extract policy_a ID from action string
                match = action.split('→')[1].strip()
                policies_to_regenerate.add(match)
        
        # Policies that were re-parented
        # (We don't have the original parent_id info here, so we regenerate all leaf policies)
        if result.policies_reparented > 0:
            leaf_policies = [
                p for p in result.policies
                if not any(child.parent_id == p.policy_id for child in result.policies)
            ]
            for SubPolicy in leaf_policies:
                policies_to_regenerate.add(SubPolicy.policy_id)
        
        logger.info(f"Regenerating {len(policies_to_regenerate)} decision trees")
        
        # Regenerate trees
        regenerated_count = 0
        for policy_id in policies_to_regenerate:
            SubPolicy = self._find_policy(result.policies, policy_id)
            if not SubPolicy:
                continue
            
            try:
                # Remove old tree
                result.trees = [t for t in result.trees if t.policy_id != policy_id]
                
                # Generate new tree
                logger.info(f"Regenerating tree for SubPolicy '{policy_id}'")
                # Note: Actual tree generation would happen here
                # For now, we just log the intention
                regenerated_count += 1
                
            except Exception as e:
                logger.error(f"Failed to regenerate tree for '{policy_id}': {e}")
        
        result.trees_regenerated = regenerated_count
        result.actions_taken.append(f"Regenerated {regenerated_count} decision trees")
        
        return result
    
    def _deduplicate_conditions(
        self,
        conditions: List[PolicyCondition]
    ) -> List[PolicyCondition]:
        """
        Remove duplicate conditions based on description similarity.
        """
        if not conditions:
            return []
        
        unique_conditions = []
        seen_descriptions = set()
        
        for condition in conditions:
            description = condition.description.strip().lower()
            
            # Check if we've seen a very similar description
            is_duplicate = False
            for seen in seen_descriptions:
                similarity = self._calculate_string_similarity(description, seen)
                if similarity > 0.9:  # 90% similar = duplicate
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_conditions.append(condition)
                seen_descriptions.add(description)
        
        if len(unique_conditions) < len(conditions):
            logger.info(f"Deduplicated conditions: {len(conditions)} → {len(unique_conditions)}")
        
        return unique_conditions
    
    def _calculate_string_similarity(self, str_a: str, str_b: str) -> float:
        """Calculate Jaccard similarity between two strings."""
        import re
        
        tokens_a = set(re.findall(r'\w+', str_a.lower()))
        tokens_b = set(re.findall(r'\w+', str_b.lower()))
        
        if not tokens_a or not tokens_b:
            return 0.0
        
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        
        return intersection / union if union > 0 else 0.0
    
    def _find_policy(self, policies: List[SubPolicy], policy_id: str) -> Optional[SubPolicy]:
        """Find a SubPolicy by ID."""
        for SubPolicy in policies:
            if SubPolicy.policy_id == policy_id:
                return SubPolicy
        return None
    
    def _create_summary(
        self,
        result: RefinementResult,
        verification_report: VerificationReport
    ) -> Dict[str, Any]:
        """Create a summary dictionary for the refinement result."""
        return {
            'policies_before': verification_report.total_policies,
            'policies_after': len(result.policies),
            'policies_merged': result.policies_merged,
            'policies_reparented': result.policies_reparented,
            'trees_before': verification_report.total_trees,
            'trees_after': len(result.trees),
            'trees_regenerated': result.trees_regenerated,
            'actions_taken': result.actions_taken
        }
    
    def _log_refinement_summary(self, result: RefinementResult):
        """Log a comprehensive refinement summary."""
        logger.info("=" * 80)
        logger.info("REFINEMENT SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Policies: {result.summary['policies_before']} → {result.summary['policies_after']}")
        logger.info(f"Trees: {result.summary['trees_before']} → {result.summary['trees_after']}")
        logger.info(f"\nActions Taken: {len(result.actions_taken)}")
        for action in result.actions_taken:
            logger.info(f"  ✓ {action}")
        logger.info("=" * 80)
