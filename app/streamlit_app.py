import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import yaml
import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from pathlib import Path
import sys
from dotenv import load_dotenv
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Reduce noise from other libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)

def log_analysis_step(message: str, level: str = "info"):
    """Helper function to log analysis steps with consistent formatting"""
    log_func = getattr(logger, level)
    log_func(f"[ANALYSIS] {message}")

# Add the report-analyst directory to the Python path
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))
logger.info(f"Added {current_dir} to Python path")

from core.analyzer import DocumentAnalyzer
from core.prompt_manager import PromptManager

# Load environment variables
load_dotenv()
logger.info("Loaded environment variables")

class TCFDAnalyzer:
    def __init__(self):
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize the real document analyzer
        self.analyzer = DocumentAnalyzer()
        self.prompt_manager = PromptManager()
        
        # Get questions from the analyzer
        self.questions = self.analyzer.questions
        if self.questions:
            self.name = "TCFD Questions"
            self.description = "Task Force on Climate-related Financial Disclosures (TCFD) question set for analyzing sustainability reports"
            logger.debug(f"Loaded {len(self.questions)} questions")
        else:
            logger.error("Failed to load questions from analyzer")
            self.questions = {}
    
    async def analyze_document(self, file_path: str, selected_questions: List[int], use_llm_scoring: bool = False, single_call: bool = True) -> AsyncGenerator[Dict, None]:
        """Analyze a document for TCFD compliance using the real analyzer"""
        try:
            log_analysis_step(f"Starting analysis of document: {file_path}")
            log_analysis_step(f"Selected questions: {selected_questions}")
            log_analysis_step(f"LLM scoring enabled: {use_llm_scoring}")
            
            results = {
                "answers": {},
                "sources": {},
                "page_numbers": {}
            }
            
            # Store raw responses for debugging
            if not hasattr(st.session_state, 'raw_responses'):
                st.session_state.raw_responses = {}
            
            # Pass use_llm_scoring to process_document
            async for result in self.analyzer.process_document(file_path, selected_questions, use_llm_scoring, single_call):
                yield result
            
        except Exception as e:
            log_analysis_step(f"Critical error during analysis: {str(e)}", "error")
            st.error(f"Error analyzing document: {str(e)}")
            yield {"error": f"Error analyzing document: {str(e)}"}

def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Save uploaded file to temp directory"""
    try:
        if uploaded_file is None:
            logger.warning("No file was uploaded")
            return None
            
        file_path = Path("temp") / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Successfully saved file: {file_path}")
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        st.error(f"Error saving file: {str(e)}")
        return None

def display_results(results: Dict[str, Any], questions: Dict[str, Dict], question_id: int):
    """Display results for a single question in a clean, organized layout"""
    try:
        # Get result components
        result = results.get(question_id, {})
        answer = result.get("ANSWER", "No answer provided")
        score = result.get("SCORE", 0)
        evidence = result.get("EVIDENCE", [])
        gaps = result.get("GAPS", [])
        sources = result.get("SOURCES", [])
        
        # Display question header with score
        st.markdown(f"### {questions[question_id]['text']}")
        st.markdown(
            f"""
            <div style="background-color: {'#4CAF50' if score >= 7 else '#FF9800' if score >= 4 else '#F44336'}; 
                        color: white; 
                        padding: 0.5rem 1rem; 
                        border-radius: 0.5rem; 
                        display: inline-block;
                        margin-bottom: 1rem;">
                Score: {score}/10
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Analysis section in an accordion (closed by default)
        with st.expander("Analysis", expanded=False):
            st.markdown(
                f"""
                <div style="padding: 1rem; 
                            border-radius: 0.5rem; 
                            background-color: #1e1e1e; 
                            color: #ffffff;
                            margin-bottom: 1rem;">
                    {answer}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Evidence and Gaps in a cleaner layout
            if evidence or gaps:
                col1, col2 = st.columns(2)
                
                with col1:
                    if evidence:
                        st.markdown("##### Key Evidence")
                        for e in evidence:
                            st.markdown(f"✓ {e}")
                
                with col2:
                    if gaps:
                        st.markdown("##### Areas for Improvement")
                        for gap in gaps:
                            st.markdown(f"○ {gap}")
            
            # Sources at the bottom of the accordion
            if sources:
                st.markdown("---")
                st.markdown("**Referenced Sources**")
                sources_text = ", ".join([f"Source {s}" for s in sources])
                st.markdown(
                    f"""
                    <div style="padding: 0.5rem; 
                                border-radius: 0.5rem; 
                                background-color: #2d2d2d;
                                color: #ffffff;">
                        {sources_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Debug section if needed
        if "Error:" in answer and hasattr(st.session_state, 'raw_responses'):
            raw_response = st.session_state.raw_responses.get(question_id)
            if raw_response:
                with st.expander("Debug Information", expanded=False):
                    st.code(raw_response)
        
        st.markdown("---")
    
    except Exception as e:
        logger.error(f"Error displaying results for question {question_id}: {str(e)}")
        st.error(f"Error displaying results for question {question_id}")

async def analyze_document_and_display(analyzer, file_path: str, selected_questions: List[int], use_llm_scoring: bool = False, single_call: bool = True):
    """Analyze document and display results as they come in"""
    try:
        results = {"answers": {}}  # Initialize results structure
        results_placeholder = st.empty()
        results_container = st.container()
        
        # Use the async generator directly
        async for result in analyzer.analyze_document(file_path, selected_questions, use_llm_scoring, single_call):
            if "error" in result:
                log_analysis_step(f"Error received from analyzer: {result['error']}", "error")
                st.error(f"Analysis error: {result['error']}")
                continue
            
            if "status" in result:
                log_analysis_step(f"Status update: {result['status']}", "debug")
                results_placeholder.write(result["status"])
                continue
            
            try:
                # Process the result
                q_id = f"tcfd_{result['question_number']}"
                log_analysis_step(f"Processing result for question {q_id}")
                
                # Store raw response for debugging
                if not hasattr(st.session_state, 'raw_responses'):
                    st.session_state.raw_responses = {}
                st.session_state.raw_responses[q_id] = result["result"]
                
                # Parse the result - it's already in the correct format from the analyzer
                result_json = json.loads(result["result"])
                
                # Store results
                results["answers"][q_id] = result_json
                
                # Display the updated results immediately
                with results_container:
                    st.empty()  # Clear previous content
                    for display_id in results["answers"]:
                        display_results(results["answers"], analyzer.questions, display_id)
                        
                        # Display chunks in a dataframe if available
                        result_data = results["answers"][display_id]
                        if "CHUNKS" in result_data:
                            log_analysis_step(f"Processing chunks data for display (question {display_id})")
                            st.subheader(f"Retrieved Context Chunks for Question {display_id}")
                            chunks_df = pd.DataFrame(result_data["CHUNKS"])
                            
                            if not chunks_df.empty:
                                log_analysis_step(f"Found {len(chunks_df)} chunks to display")
                                
                                # Check for computed scores
                                has_computed_scores = "computed_score" in chunks_df.columns
                                log_analysis_step(f"Computed scores present: {has_computed_scores}")
                                
                                # Clean up the display by selecting relevant columns
                                columns = ["text", "metadata", "relevance_score"]
                                if has_computed_scores:
                                    columns.append("computed_score")
                                    log_analysis_step("Including computed scores in display")
                                display_df = chunks_df[columns].copy()
                                
                                # Convert metadata dict to string for better display
                                display_df["metadata"] = display_df["metadata"].apply(lambda x: json.dumps(x, indent=2))
                                
                                # Format scores
                                display_df["relevance_score"] = display_df["relevance_score"].apply(lambda x: f"{x:.4f}")
                                if has_computed_scores:
                                    display_df["computed_score"] = display_df["computed_score"].apply(lambda x: f"{x:.4f}")
                                    # Sort by computed score if available
                                    display_df = display_df.sort_values("computed_score", ascending=False)
                                    log_analysis_step("Sorted chunks by computed score")
                                else:
                                    # Sort by vector similarity score
                                    display_df = display_df.sort_values("relevance_score", ascending=False)
                                    log_analysis_step("Sorted chunks by vector similarity")
                                
                                # Rename columns for better display
                                display_df = display_df.rename(columns={
                                    "relevance_score": "Vector Similarity",
                                    "computed_score": "LLM Computed Score",
                                    "text": "Content",
                                    "metadata": "Metadata"
                                })
                                
                                log_analysis_step(f"Displaying dataframe with columns: {display_df.columns.tolist()}")
                                st.dataframe(display_df, use_container_width=True)
                            else:
                                log_analysis_step("Warning: Empty chunks dataframe", "warning")
            
            except json.JSONDecodeError as e:
                log_analysis_step(f"JSON decode error for {q_id}: {str(e)}", "error")
                st.error(f"Error processing result for question {q_id}")
            except Exception as e:
                log_analysis_step(f"Unexpected error processing result for {q_id}: {str(e)}", "error")
                st.error(f"Error processing result: {str(e)}")
        
        # Log final results summary
        log_analysis_step(f"Analysis complete. Processed {len(results['answers'])} questions successfully")
        
        # Store final results in session state
        st.session_state.results = results
        st.session_state.analysis_complete = True
        
        # Clear the status placeholder
        results_placeholder.empty()
        
    except Exception as e:
        log_analysis_step(f"Critical error during analysis: {str(e)}", "error")
        st.error(f"Error during analysis: {str(e)}")

def main():
    st.set_page_config(
        page_title="TCFD Report Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("TCFD Report Analyzer")
    st.write("Upload a PDF report and select questions for TCFD compliance analysis.")
    
    try:
        # Initialize analyzer
        analyzer = TCFDAnalyzer()
        
        # File upload
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file:
            try:
                logger.info(f"File uploaded: {uploaded_file.name}")
                st.write("File uploaded successfully!")
                
                # Display available questions
                if not analyzer.questions:
                    questions_path = Path(__file__).parent / "questionsets" / "tcfd_questions.yaml"
                    st.error(f"No questions loaded. Please check if the questions file exists at: {questions_path}")
                    return
                    
                st.subheader("Select Questions for Analysis")
                selected_questions = []
                for q_id, q_data in analyzer.questions.items():
                    if st.checkbox(q_data['text'], key=q_id):
                        selected_questions.append(int(q_id.split('_')[1]))  # Convert to integer ID immediately
                
                # Add configuration options
                with st.expander("Advanced Options"):
                    use_llm_scoring = st.checkbox("Use LLM for relevance scoring", value=False)
                    if use_llm_scoring:
                        single_call = st.checkbox(
                            "Score all chunks in single LLM call", 
                            value=True,
                            help="More efficient but may be less accurate with large numbers of chunks"
                        )
                    else:
                        single_call = True
                
                if st.button("Analyze Document"):
                    with st.spinner("Analyzing document..."):
                        # Save uploaded file
                        file_path = save_uploaded_file(uploaded_file)
                        if not file_path:
                            return
                        
                        log_analysis_step(f"Starting analysis with LLM scoring: {use_llm_scoring}")
                        log_analysis_step(f"Processing questions: {selected_questions}")
                        
                        # Run analysis using the selected questions
                        asyncio.run(analyze_document_and_display(
                            analyzer, 
                            file_path, 
                            selected_questions,  # Already integer IDs
                            use_llm_scoring,
                            single_call
                        ))
            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")
    except Exception as e:
        st.error(f"Error initializing analyzer: {str(e)}")

if __name__ == "__main__":
    main() 