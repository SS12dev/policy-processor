"""
Performance and Quality Metrics Dashboard Component

Displays processing statistics and quality indicators in Streamlit UI.
"""
import streamlit as st
from typing import Dict, Any, Optional
from datetime import datetime


def render_processing_metrics(
    result: Dict[str, Any],
    show_detailed: bool = False
) -> None:
    """
    Render processing metrics dashboard.
    
    Args:
        result: Processing result dict from orchestrator
        show_detailed: Whether to show detailed breakdown
    """
    if not result:
        st.info("No processing results available yet.")
        return
    
    # Extract metrics
    policy_hierarchy = result.get("policy_hierarchy", {})
    decision_trees = result.get("decision_trees", [])
    validation = result.get("validation_result", {})
    metadata = result.get("enhanced_document_metadata", {})
    chunks = result.get("chunks", [])
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        policies_count = policy_hierarchy.get("total_policies", 0) if isinstance(policy_hierarchy, dict) else getattr(policy_hierarchy, "total_policies", 0)
        st.metric(
            "Policies Extracted",
            policies_count,
            help="Total number of policies extracted from document"
        )
    
    with col2:
        trees_count = len(decision_trees)
        st.metric(
            "Decision Trees",
            trees_count,
            help="Number of decision trees generated"
        )
    
    with col3:
        if validation:
            confidence = validation.get("overall_confidence", 0) if isinstance(validation, dict) else getattr(validation, "overall_confidence", 0)
            st.metric(
                "Overall Confidence",
                f"{confidence:.0%}",
                help="Confidence in extracted policies and trees"
            )
        else:
            st.metric("Overall Confidence", "N/A")
    
    with col4:
        processing_time = result.get("processing_time_seconds", 0)
        st.metric(
            "Processing Time",
            f"{processing_time:.1f}s",
            help="Total time to process document"
        )
    
    # Quality indicators
    if show_detailed and metadata:
        st.subheader("Document Quality")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            extractability = metadata.get("overall_extractability_score", 0) if isinstance(metadata, dict) else getattr(metadata, "overall_extractability_score", 0)
            st.progress(extractability, text=f"Extractability: {extractability:.0%}")
        
        with col2:
            structure_conf = metadata.get("confidence_in_structure", 0) if isinstance(metadata, dict) else getattr(metadata, "confidence_in_structure", 0)
            st.progress(structure_conf, text=f"Structure: {structure_conf:.0%}")
        
        with col3:
            if chunks:
                complete_chunks = sum(1 for c in chunks if c.get("has_complete_context", False))
                completeness = complete_chunks / len(chunks) if chunks else 0
                st.progress(completeness, text=f"Context: {completeness:.0%}")
        
        # Detailed statistics
        with st.expander("📊 Detailed Statistics"):
            st.markdown("### Document Analysis")
            
            total_pages = metadata.get("total_pages", 0) if isinstance(metadata, dict) else getattr(metadata, "total_pages", 0)
            complexity = metadata.get("complexity_score", 0) if isinstance(metadata, dict) else getattr(metadata, "complexity_score", 0)
            structure_type = metadata.get("structure_type", "unknown") if isinstance(metadata, dict) else getattr(metadata, "structure_type", "unknown")
            
            st.write(f"- **Pages**: {total_pages}")
            st.write(f"- **Complexity**: {complexity:.2f}")
            st.write(f"- **Structure**: {structure_type}")
            
            policy_pages = metadata.get("policy_pages_count", 0) if isinstance(metadata, dict) else getattr(metadata, "policy_pages_count", 0)
            admin_pages = metadata.get("admin_pages_count", 0) if isinstance(metadata, dict) else getattr(metadata, "admin_pages_count", 0)
            st.write(f"- **Policy Pages**: {policy_pages}/{total_pages}")
            st.write(f"- **Admin Pages**: {admin_pages}/{total_pages}")
            
            st.markdown("### Policy Extraction")
            root_policies = policy_hierarchy.get("root_policies", []) if isinstance(policy_hierarchy, dict) else getattr(policy_hierarchy, "root_policies", [])
            max_depth = policy_hierarchy.get("max_depth", 0) if isinstance(policy_hierarchy, dict) else getattr(policy_hierarchy, "max_depth", 0)
            definitions = policy_hierarchy.get("definitions", {}) if isinstance(policy_hierarchy, dict) else getattr(policy_hierarchy, "definitions", {})
            
            st.write(f"- **Root Policies**: {len(root_policies)}")
            st.write(f"- **Max Hierarchy Depth**: {max_depth}")
            st.write(f"- **Definitions Extracted**: {len(definitions)}")
            
            st.markdown("### Chunking")
            st.write(f"- **Total Chunks**: {len(chunks)}")
            if chunks:
                avg_tokens = sum(c.get("token_count", 0) for c in chunks) / len(chunks)
                st.write(f"- **Avg Tokens/Chunk**: {avg_tokens:.0f}")
                
                filtered_pages = len(metadata.get("pages_to_filter", [])) if isinstance(metadata, dict) else len(getattr(metadata, "pages_to_filter", []))
                st.write(f"- **Pages Filtered**: {filtered_pages}")
            
            st.markdown("### Decision Trees")
            if decision_trees:
                total_nodes = sum(t.get("total_nodes", 0) if isinstance(t, dict) else getattr(t, "total_nodes", 0) for t in decision_trees)
                total_paths = sum(t.get("total_paths", 0) if isinstance(t, dict) else getattr(t, "total_paths", 0) for t in decision_trees)
                avg_conf = sum(t.get("confidence_score", 0) if isinstance(t, dict) else getattr(t, "confidence_score", 0) for t in decision_trees) / len(decision_trees)
                
                st.write(f"- **Total Nodes**: {total_nodes}")
                st.write(f"- **Total Paths**: {total_paths}")
                st.write(f"- **Avg Confidence**: {avg_conf:.0%}")
                
                complete_routing = sum(1 for t in decision_trees if (t.get("has_complete_routing", False) if isinstance(t, dict) else getattr(t, "has_complete_routing", False)))
                st.write(f"- **Complete Routing**: {complete_routing}/{len(decision_trees)}")


def render_validation_issues(validation_result: Dict[str, Any]) -> None:
    """
    Render validation issues with severity indicators.
    
    Args:
        validation_result: Validation result dict
    """
    if not validation_result:
        return
    
    issues = validation_result.get("issues", []) if isinstance(validation_result, dict) else getattr(validation_result, "issues", [])
    
    if not issues:
        st.success("✅ No validation issues found!")
        return
    
    # Group by severity
    errors = [i for i in issues if (i.get("severity") if isinstance(i, dict) else getattr(i, "severity", "")) == "error"]
    warnings = [i for i in issues if (i.get("severity") if isinstance(i, dict) else getattr(i, "severity", "")) == "warning"]
    info = [i for i in issues if (i.get("severity") if isinstance(i, dict) else getattr(i, "severity", "")) == "info"]
    
    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        if errors:
            st.error(f"🚫 {len(errors)} Errors")
    with col2:
        if warnings:
            st.warning(f"⚠️ {len(warnings)} Warnings")
    with col3:
        if info:
            st.info(f"ℹ️ {len(info)} Info")
    
    # Detailed issues
    with st.expander(f"View All Issues ({len(issues)} total)"):
        for issue in issues:
            severity = issue.get("severity") if isinstance(issue, dict) else getattr(issue, "severity", "")
            issue_type = issue.get("issue_type") if isinstance(issue, dict) else getattr(issue, "issue_type", "")
            description = issue.get("description") if isinstance(issue, dict) else getattr(issue, "description", "")
            location = issue.get("location") if isinstance(issue, dict) else getattr(issue, "location", "")
            suggestion = issue.get("suggestion") if isinstance(issue, dict) else getattr(issue, "suggestion", "")
            
            severity_icon = {"error": "🚫", "warning": "⚠️", "info": "ℹ️"}.get(severity, "")
            
            st.markdown(f"**{severity_icon} {severity.upper()}: {issue_type}**")
            st.write(description)
            st.caption(f"Location: {location}")
            
            if suggestion:
                st.info(f"💡 Suggestion: {suggestion}")
            
            st.divider()


def render_cost_estimate(result: Dict[str, Any]) -> None:
    """
    Render estimated API cost for processing.
    
    Args:
        result: Processing result dict
    """
    chunks = result.get("chunks", [])
    decision_trees = result.get("decision_trees", [])
    
    # Rough cost estimates (adjust based on actual pricing)
    extraction_cost = len(chunks) * 0.001  # GPT-4o-mini for extraction
    tree_cost = len(decision_trees) * 0.005  # GPT-4 for trees
    total_cost = extraction_cost + tree_cost
    
    with st.expander("💰 Estimated API Cost"):
        st.write(f"- Policy Extraction: ${extraction_cost:.4f}")
        st.write(f"- Tree Generation: ${tree_cost:.4f}")
        st.write(f"**Total Estimated Cost**: ${total_cost:.4f}")
        st.caption("Actual costs may vary based on token usage and retries")
