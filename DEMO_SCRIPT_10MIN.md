# 🎯 10-MINUTE SPRINT DEMO SCRIPT
## Policy Processing System - Comprehensive Demo

**Presenter Notes:** This is a polished, rehearsed script. Practice timing each section. Total: 10 minutes.

---

## ⏱️ TIMING BREAKDOWN
- **Opening & Overview**: 1 minute
- **Architecture & Technology**: 2 minutes  
- **Live Demo - Processing Pipeline**: 5 minutes
- **Decision Tree Visualization**: 1.5 minutes
- **Closing & Key Takeaways**: 0.5 minutes

---

## 🎬 OPENING (0:00 - 1:00)

**[Start with Energy]**

> "Good morning/afternoon everyone! Today I'm excited to demo our **AI-Powered Policy Processing System** - a production-ready platform that transforms complex medical policy documents into structured data and interactive decision trees.

> **The Problem We're Solving:**
> Healthcare organizations deal with hundreds of policy documents - 50, 100, even 200 pages long. These contain critical prior authorization rules, coverage criteria, and decision workflows. Extracting this information manually is time-consuming, error-prone, and doesn't scale.

> **Our Solution:**
> We've built an intelligent system that automatically:
> - ✅ Processes any PDF format - typed, scanned, or mixed
> - ✅ Extracts structured policies with criteria and conditions  
> - ✅ Generates interactive decision trees for authorization workflows
> - ✅ Validates everything with multi-stage quality checks

> Let me show you how it works."

---

## 🏗️ ARCHITECTURE & TECHNOLOGY (1:00 - 3:00)

**[Switch to Architecture Diagram or README]**

> "First, let me quickly walk you through the architecture because we've built this with modern, scalable patterns.

### **System Architecture**

> **Two Core Modules:**

> **1. Agent Module** - This is our processing brain:
> - FastAPI server implementing **Google's A2A Protocol** - that's Agent-to-Agent communication standard
> - **LangGraph** state machine with 10 intelligent processing nodes
> - Stateless design using **Redis** for temporary storage
> - This means we can horizontally scale - spin up multiple agent instances behind a load balancer

> **2. Client Module** - This is our web interface:
> - **Streamlit** web UI - clean, interactive
> - **SQLite** database for processing history
> - Real-time streaming updates from the agent
> - Tree visualization and export capabilities

### **Technology Highlights**

> **Why this tech stack matters:**

> **A2A Protocol** - We're using Google's standardized agent communication protocol. This means our system can interoperate with any A2A-compliant agent. It's future-proof and enterprise-ready.

> **LangGraph State Machine** - This isn't just sequential processing. We have a sophisticated state machine with:
> - Conditional routing (smart decision-making)
> - Automatic retry logic for failed nodes
> - Multi-stage verification and refinement
> - Full error recovery

> **Multi-LLM Support** - We support OpenAI, Azure OpenAI, and LiteLLM proxy. The system intelligently chooses models:
> - Simple documents → GPT-4o-mini (cost-effective)
> - Complex documents → GPT-4o (high accuracy)
> - Decision trees → Always GPT-4o (quality matters)

> **Production-Ready Features:**
> - Comprehensive logging and metrics
> - Kubernetes-ready containers
> - Health monitoring
> - Database persistence

> Now let's see it in action."

---

## 🚀 LIVE DEMO - PROCESSING PIPELINE (3:00 - 8:00)

**[Switch to Browser - Streamlit UI]**

> "I have both servers already running - the agent module and the client module. Let's upload a policy document."

### **Upload Document (3:00 - 3:30)**

**[Navigate to Upload & Process tab]**

> "This is our upload interface. I'm going to select a real medical policy document - the **Bariatric Surgery Policy** from Elevance Health. This is a complex document with multiple coverage criteria, exclusions, and authorization requirements.

**[Upload the PDF - Drag & Drop or Select File]**

> "As soon as I upload, processing begins and we start streaming updates in real-time. Watch the progress bar and the detailed logs."

### **Node-by-Node Walkthrough (3:30 - 7:30)**

**[As each stage processes, explain it. This is the heart of the demo!]**

---

#### **NODE 1: PDF PARSING (Progress: 5%)**

**[Wait for this stage to appear]**

> "**Node 1 - PDF Parsing**: This is where we extract everything from the PDF using a multi-strategy approach:
> - **Text Extraction**: PyMuPDF for typed text
> - **OCR Processing**: Tesseract for scanned pages - runs in parallel with 4 workers
> - **Image Detection**: Extracts images and tables
> - **Structure Analysis**: Detects headings, sections, and table of contents
> - **Metadata Capture**: Page counts, fonts, document structure

> Why this matters: We can handle ANY PDF format - completely typed documents, fully scanned images, or mixed documents. The system automatically detects what's needed.

> Notice in the logs - it's showing us: X pages processed, Y tables detected, Z scanned pages with OCR, headings detected."

---

#### **NODE 2: DOCUMENT ANALYSIS (Progress: 15%)**

**[When this stage starts]**

> "**Node 2 - Document Analysis**: Now we're doing intelligent page-level classification:
> - **Content Classification**: Each page is categorized - is it policy content, administrative info, definitions, references, or bibliography?
> - **Policy Boundary Detection**: Where do policies start and end?
> - **Content Zones**: Map which pages contain main policies vs. supporting material
> - **Complexity Scoring**: Calculate document complexity (0.0 to 1.0)

> This is crucial because not all pages are relevant. We don't want to waste processing time on table of contents, disclaimers, or bibliography pages.

> The system also decides here: 'Is this document complex enough that I should use GPT-4 for extraction, or can I use GPT-4o-mini to save costs?'

> See the complexity score? If it's above 0.7, we automatically upgrade to GPT-4 for better accuracy."

---

#### **NODE 3: INTELLIGENT CHUNKING (Progress: 25%)**

**[When this stage starts]**

> "**Node 3 - Intelligent Chunking**: This is where we get smart about breaking up the document:
> - **Policy-Aware Chunking**: We respect policy boundaries - never split a policy across chunks
> - **Page Filtering**: Removes non-policy pages identified in analysis
> - **Context Preservation**: Maintains semantic continuity across multi-page policies
> - **Overlap Strategy**: Adds overlap between chunks to preserve context
> - **Duplicate Detection**: Identifies potential duplicate policies

> Traditional systems might just split every 5 pages or every 1000 tokens. We're doing semantic chunking based on document structure.

> Watch the logs - it's showing filtered pages, unique policies detected, and chunk statistics. We typically create 8-15 chunks from a 50-page document."

---

#### **NODE 4: POLICY EXTRACTION (Progress: 40%)**

**[When this stage starts]**

> "**Node 4 - Policy Extraction**: This is one of our most sophisticated nodes - **1,526 lines of code**:
> - **LLM-Powered Extraction**: Uses GPT to extract structured policies from each chunk
> - **Hierarchy Building**: Organizes policies into parent-child relationships
> - **Criteria Extraction**: Pulls out conditions, requirements, and exclusions
> - **Coverage Information**: Extracts what's covered and what's not
> - **Documentation Requirements**: Identifies required paperwork
> - **Provenance Tracking**: Maintains source page references for every piece of data

> What's being extracted:
> - Policy titles and descriptions
> - Eligibility criteria (age, BMI, diagnosis codes)
> - Coverage conditions (when it's approved)
> - Exclusions (when it's denied)
> - Documentation requirements
> - Prior authorization workflows

> This is where we go from unstructured text to structured, queryable data."

---

#### **NODE 5: DECISION TREE GENERATION (Progress: 60%)**

**[When this stage starts]**

> "**Node 5 - Decision Tree Generation**: Now we convert policies into interactive decision trees - **1,192 lines of code**:
> - **Hierarchical Tree Generation**: Creates trees for each major policy
> - **Question Node Creation**: Converts criteria into yes/no questions
> - **Decision Logic**: Builds branching paths based on answers
> - **Outcome Nodes**: Determines final outcomes (approved, denied, pending, etc.)
> - **Confidence Scoring**: Each tree gets a confidence score
> - **Always Uses GPT-4**: We want maximum quality here

> Example: For bariatric surgery, it might create questions like:
> - 'Is patient age ≥ 18 years?'
> - 'Is BMI ≥ 40 or BMI ≥ 35 with comorbidities?'
> - 'Has patient completed 6 months of supervised weight loss?'
> - Each answer branches to the next question or a final outcome

> These trees become interactive workflows that staff can use for prior authorization decisions."

---

#### **NODE 6: VALIDATION (Progress: 85%)**

**[When this stage starts]**

> "**Node 6 - Validation**: Multi-stage quality assurance:
> - **Completeness Check**: Are all tree paths complete? No dead ends?
> - **Routing Validation**: Does every question have proper yes/no branches?
> - **Structure Validation**: Is the hierarchy logical?
> - **Confidence Thresholds**: Are confidence scores acceptable?
> - **Cross-Validation**: Do policies match extracted criteria?

> If validation finds issues - trees with incomplete paths, low confidence scores, missing branches - it triggers automatic retry."

---

#### **NODE 7: RETRY LOGIC (If Triggered)**

**[If you see retry]**

> "**Node 7 - Retry Failed Trees**: This is our self-healing capability:
> - Identifies which specific trees failed validation
> - Re-generates only the failed trees with improved prompts
> - Uses GPT-4 with enhanced context
> - Re-validates after retry

> This ensures quality - we don't just accept poor results. The system keeps trying until it gets it right, up to a maximum number of retries."

---

#### **NODE 8: VERIFICATION (Progress: 92%)**

**[When this stage starts]**

> "**Node 8 - Verification & Quality Check**: Deep verification stage:
> - **Duplicate Detection**: Are there duplicate policies that should be merged?
> - **Coverage Analysis**: Did we miss any policies from the source document?
> - **Over-Extraction Check**: Did we hallucinate policies that don't exist?
> - **Tree Completeness**: Validates all decision paths are complete
> - **Hierarchy Structure**: Checks parent-child relationships make sense

> If verification detects quality issues, it triggers refinement."

---

#### **NODE 9: REFINEMENT (If Triggered)**

**[If you see refinement]**

> "**Node 9 - Refinement**: Automated quality improvement:
> - **Merge Duplicates**: Combines duplicate policies intelligently
> - **Fix Hierarchy**: Adjusts parent-child relationships
> - **Regenerate Trees**: Recreates decision trees for merged policies
> - **Re-Verification**: Checks results again after refinement

> This is like having an AI quality assurance analyst that catches issues before they reach production."

---

#### **NODE 10: COMPLETION (Progress: 100%)**

**[When processing completes]**

> "**Node 10 - Completion**: Final packaging:
> - Aggregates all results
> - Calculates final statistics
> - Stores to database
> - Prepares exports

> And we're done! Look at the summary:
> - **Processing Time**: X minutes Y seconds
> - **Policies Extracted**: Z policies
> - **Decision Trees Generated**: N trees
> - **Validation**: PASSED with XX% confidence
> - **Model Usage**: Shows which models were used

> That's the complete pipeline - 10 nodes working together as a state machine."

---

## 🌳 DECISION TREE VISUALIZATION (8:00 - 9:30)

**[Switch to "Review Decision Trees" tab]**

> "Now let's look at what we generated. Switch to the **Review Decision Trees** tab.

**[Select a Tree]**

> Here's the decision tree for **Bariatric Surgery Coverage**. Look at this clean visualization:

### **Tree Features**

> **Color Coding:**
> - 🔵 **Blue tiles** = Question nodes ('Is patient BMI ≥ 40?')
> - 🟠 **Orange tiles** = Decision nodes (intermediate decisions)
> - 🟢 **Green tiles** = Approved outcomes
> - 🔴 **Red tiles** = Denied outcomes
> - 🟣 **Purple tiles** = Pending review
> - 🟡 **Yellow tiles** = Requires additional documentation

> **Interactive Features:**
> - Collapsible branches - click to expand/collapse
> - Indentation shows hierarchy
> - Answer labels on connecting lines
> - Click any node to see details

### **Walk Through a Path**

**[Click through a decision path]**

> Let me walk through a typical authorization decision:

> **Start**: 'Is patient age 18 or older?' 
> → **YES** → Move to next question
> 
> 'Does patient have BMI ≥ 40 or BMI ≥ 35 with comorbidities?'
> → **YES** → Move to next question
> 
> 'Has patient completed 6 months of medically supervised weight loss program?'
> → **YES** → Move to next question
> 
> 'Does patient have documentation from mental health evaluation?'
> → **YES** → 
> 
> **OUTCOME**: ✅ **APPROVED** with documentation required

> This is a complete authorization workflow - extracted and structured automatically from that 50-page PDF.

### **Export Options**

> We can export this as:
> - **JSON** - for system integration with EMRs or authorization systems
> - **CSV** - for analysis in Excel
> - **Database** - it's already stored for historical tracking

> Organizations can now integrate these decision trees directly into their prior authorization systems."

---

## 🎯 CLOSING & KEY TAKEAWAYS (9:30 - 10:00)

**[Return to Overview or Closing Slide]**

> "Let me wrap up with the key takeaways:

### **What Makes This Special:**

> **1. Production-Ready Architecture**
> - A2A Protocol for standardized agent communication
> - Stateless, scalable design ready for Kubernetes
> - Real-time streaming for live progress updates

> **2. Intelligent Processing**
> - 10-node state machine with conditional routing
> - Automatic model selection (cost vs. quality)
> - Self-healing with retry and refinement logic
> - Multi-stage validation for quality assurance

> **3. Real Business Value**
> - Processes 50-100 page policies in 2-3 minutes
> - Extracts structured data from unstructured documents
> - Creates usable decision workflows automatically
> - Handles any PDF format - typed, scanned, or mixed

> **4. Enterprise-Grade Quality**
> - Over 3,500 lines of processing logic
> - Comprehensive error handling
> - Full audit trails and provenance tracking
> - Database persistence for compliance

### **Next Steps:**

> We're ready for:
> - Multi-tenant deployment
> - Batch processing for document libraries
> - API authentication and role-based access
> - Integration with existing prior authorization systems
> - Azure/AWS cloud deployment

> **Questions?**"

---

## 📋 QUICK REFERENCE: ALL 10 NODES SUMMARY

**Use this for Q&A or if asked about specific nodes:**

| Node | Name | Purpose | Key Features |
|------|------|---------|--------------|
| **1** | **PDF Parsing** | Extract all content from PDF | Multi-strategy extraction, OCR, image/table detection, structure analysis |
| **2** | **Document Analysis** | Understand document structure | Page classification, policy boundaries, complexity scoring, model selection |
| **3** | **Intelligent Chunking** | Break into semantic chunks | Policy-aware splitting, page filtering, context preservation, duplicate detection |
| **4** | **Policy Extraction** | Extract structured policies | LLM-powered extraction (1,526 lines), hierarchy building, criteria extraction |
| **5** | **Decision Tree Generation** | Create interactive trees | Question generation (1,192 lines), branching logic, outcome nodes, confidence scoring |
| **6** | **Validation** | Quality assurance | Completeness checks, routing validation, confidence thresholds, structure validation |
| **7** | **Retry Logic** | Fix failed components | Automatic retry, improved prompts, selective regeneration, re-validation |
| **8** | **Verification** | Deep quality check | Duplicate detection, coverage analysis, completeness validation, hierarchy checks |
| **9** | **Refinement** | Automated improvement | Merge duplicates, fix hierarchy, regenerate trees, re-verify results |
| **10** | **Completion** | Finalize & package | Aggregate results, calculate stats, database storage, prepare exports |

---

## 🎭 PRESENTATION TIPS

### **Energy & Pacing**
- ✅ Start strong - this is impressive work
- ✅ Speak clearly and maintain energy during processing
- ✅ Use the streaming time to explain features (don't just stand silently)
- ✅ Point to the screen as things happen
- ✅ Smile and make eye contact with stakeholders

### **Handling Technical Depth**
- ✅ Adjust depth based on audience:
  - **Executive Audience**: Focus on business value, less technical detail
  - **Technical Audience**: Dive into architecture, LangGraph, A2A protocol
  - **Mixed Audience**: Balance both - highlight tech briefly, focus on outcomes

### **If Processing Takes Time**
- ✅ Use the wait time to explain architecture
- ✅ Talk about each node as it processes
- ✅ Highlight the real-time streaming capability
- ✅ Discuss scalability and production readiness

### **If Something Goes Wrong**
- ✅ Stay calm - tech demos have hiccups
- ✅ Have the "Review Decision Trees" tab open with pre-processed results
- ✅ Say: "Let me show you results from an earlier run while this processes"
- ✅ Focus on the visualization and features instead
- ✅ Blame it on "demo gods" with humor

### **Strong Close**
- ✅ Summarize key achievements
- ✅ Emphasize production readiness
- ✅ Open for questions with confidence
- ✅ Thank the team if applicable

---

## ⚡ PRE-DEMO CHECKLIST

**30 Minutes Before:**
- [ ] Start Agent Server: `python server.py`
- [ ] Start Client UI: `streamlit run app.py`
- [ ] Test with one document to warm up
- [ ] Open browser to `http://localhost:8501`
- [ ] Have sample PDFs ready (Bariatric Surgery recommended)
- [ ] Clear any old processing jobs from UI
- [ ] Open README.md for architecture reference
- [ ] Test internet connection (LLM API calls)
- [ ] Close unnecessary apps to free resources

**5 Minutes Before:**
- [ ] Refresh browser page
- [ ] Check both servers are running (health status green)
- [ ] Have backup pre-processed result ready
- [ ] Take a deep breath - you've got this! 🚀

---

## 🎯 AUDIENCE-SPECIFIC ADJUSTMENTS

### **For Executive/Business Stakeholders:**
- Focus on: Time savings, accuracy, scalability, ROI
- Minimize: Technical jargon, code references
- Emphasize: "Processes 100-page documents in 2-3 minutes automatically"

### **For Technical Team/Engineers:**
- Focus on: Architecture, LangGraph, A2A protocol, stateless design
- Include: Code complexity (3,500+ lines), technology choices, scalability patterns
- Emphasize: "Production-ready with Kubernetes support, full observability"

### **For Healthcare/Domain Experts:**
- Focus on: Policy extraction accuracy, decision tree usability, audit trails
- Show: Actual policy criteria, documentation requirements, authorization workflows
- Emphasize: "Maintains provenance - every decision traceable to source document"

---

## 💬 ANTICIPATED QUESTIONS & ANSWERS

**Q: How accurate is the extraction?**
> "We have multi-stage validation with confidence scoring. Typical confidence is 85-95%. Failed extractions are automatically retried with GPT-4. We also have verification and refinement stages that catch duplicates and quality issues before completion."

**Q: How long does processing take?**
> "30-50 page documents typically process in 2-3 minutes. 100+ page documents might take 5-8 minutes depending on complexity and whether GPT-4 is used. We can optimize with model selection and parallel processing."

**Q: Can it handle scanned PDFs?**
> "Absolutely. Our multi-strategy extraction includes OCR with Tesseract. We process scanned PDFs in parallel with 4 workers. The system automatically detects which pages need OCR and handles typed/scanned/mixed documents seamlessly."

**Q: How does it scale?**
> "The agent module is completely stateless - uses Redis for temporary storage. We can run multiple agent instances behind a load balancer. It's containerized and ready for Kubernetes deployment with horizontal pod autoscaling."

**Q: What about costs?**
> "We optimize costs through intelligent model selection. Simple documents use GPT-4o-mini ($0.15/1M tokens) while complex documents use GPT-4o ($5/1M tokens). Decision trees always use GPT-4o for quality. A typical 50-page document costs $0.20-0.40 in LLM API calls."

**Q: Can we customize the decision trees?**
> "Yes. The system extracts what's in the policy, but we can add custom validation rules, outcome types, and business logic. The tree structure is stored as JSON, making it easy to modify or extend."

**Q: What about security and compliance?**
> "PDFs are not stored in state - only processed in memory. We maintain full audit trails with provenance tracking. All processing is logged. For HIPAA compliance, we can deploy in a private cloud with encrypted storage and access controls."

**Q: What happens if the LLM API is down?**
> "We have comprehensive error handling. Failed API calls trigger retries with exponential backoff. If retries fail, the job is marked as failed with detailed error logs. We can implement fallback models or queue-based processing for resilience."

**Q: Can it handle multiple languages?**
> "Currently optimized for English. The OCR supports multiple languages (configurable), and GPT-4 has strong multilingual capabilities. We'd need to validate prompts and output quality for other languages, but the architecture supports it."

---

**Good luck with your demo! You've built something genuinely impressive. Go show it off! 🚀**
