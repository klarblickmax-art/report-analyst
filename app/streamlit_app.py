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
import time

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
from core.dataframe_manager import create_analysis_dataframes, create_combined_dataframe

# Load environment variables
load_dotenv()
logger.info("Loaded environment variables")

class ReportAnalyzer:
    def __init__(self):
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize the real document analyzer
        self.analyzer = DocumentAnalyzer()
        self.prompt_manager = PromptManager()
        
    def load_question_set(self, question_set: str) -> Dict:
        """Load questions from the specified question set file"""
        question_file = Path(__file__).parent / "questionsets" / f"{question_set}_questions.yaml"
        try:
            with open(question_file, 'r') as f:
                data = yaml.safe_load(f)
                # Create questions with proper IDs
                questions = {}
                for i, q in enumerate(data['questions'], 1):  # Start from 1
                    q_id = f"{question_set}_{i}"
                    questions[q_id] = q
                    # Add the ID to the question data for reference
                    q['id'] = q_id
                    # Add numeric ID for easier mapping
                    q['number'] = i
                return {
                    "questions": questions,
                    "name": data.get('name', f"{question_set.upper()} Questions"),
                    "description": data.get('description', '')
                }
        except Exception as e:
            logger.error(f"Failed to load questions from {question_file}: {str(e)}")
            return {
                "questions": {},
                "name": "",
                "description": ""
            }
    
    async def analyze_document(self, file_path: str, questions: Dict, selected_questions: List[str], use_llm_scoring: bool = False, single_call: bool = True) -> AsyncGenerator[Dict, None]:
        """Analyze a document using the provided questions"""
        try:
            log_analysis_step(f"Starting analysis of document: {file_path}")
            log_analysis_step(f"Selected questions: {selected_questions}")
            log_analysis_step(f"LLM scoring enabled: {use_llm_scoring}")
            
            # Update analyzer with the current questions
            self.analyzer.questions = questions
            
            # Convert selected question IDs to numbers for the analyzer
            selected_numbers = [questions[q_id]['number'] for q_id in selected_questions]
            
            # Get the question set prefix from the first selected question
            question_set = selected_questions[0].split('_')[0] if selected_questions else "tcfd"
            self.analyzer.question_set_prefix = question_set
            
            # Pass use_llm_scoring to process_document
            async for result in self.analyzer.process_document(file_path, selected_numbers, use_llm_scoring, single_call):
                # Convert question number back to ID if needed
                if 'question_number' in result:
                    result['question_id'] = f"{question_set}_{result['question_number']}"
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

def display_dataframes(analysis_df: pd.DataFrame, chunks_df: pd.DataFrame):
    """Display only the dataframes without download buttons"""
    # Main Analysis Table (only once)
    st.subheader("Analysis Results")
    st.dataframe(
        analysis_df,
        use_container_width=True,
        column_config={
            "Score": st.column_config.NumberColumn(
                "Score",
                help="Analysis score out of 10",
                min_value=0,
                max_value=10,
                format="%.1f"
            ),
            "Analysis": st.column_config.TextColumn(
                "Analysis",
                width="large"
            ),
            "Key Evidence": st.column_config.TextColumn(
                "Key Evidence",
                width="medium"
            )
        }
    )
    
    # Document Chunks Table (only once)
    st.subheader("Document Chunks")
    st.dataframe(
        chunks_df,
        use_container_width=True,
        column_config={
            "Vector Similarity": st.column_config.NumberColumn(
                "Vector Similarity",
                format="%.3f"
            ),
            "LLM Score": st.column_config.NumberColumn(
                "LLM Score",
                format="%.3f"
            ),
            "Chunk Text": st.column_config.TextColumn(
                "Chunk Text",
                width="large"
            )
        }
    )

def display_download_buttons(analysis_df: pd.DataFrame, chunks_df: pd.DataFrame):
    """Display download buttons in a separate section"""
    st.subheader("Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "Download Complete Analysis",
            data=analysis_df.to_csv(index=False).encode('utf-8'),
            file_name="complete_analysis.csv",
            mime="text/csv",
            key='download_analysis'
        )
    
    with col2:
        st.download_button(
            "Download All Chunks",
            data=chunks_df.to_csv(index=False).encode('utf-8'),
            file_name="all_chunks.csv",
            mime="text/csv",
            key='download_chunks'
        )

async def analyze_document_and_display(analyzer, file_path: str, questions: Dict, selected_questions: List[str], use_llm_scoring: bool = False, single_call: bool = True):
    """Analyze document and display results as they come in"""
    try:
        results = {"answers": {}}
        status_placeholder = st.empty()
        
        async for result in analyzer.analyze_document(file_path, questions, selected_questions, use_llm_scoring, single_call):
            if "error" in result:
                log_analysis_step(f"Error received from analyzer: {result['error']}", "error")
                st.error(f"Analysis error: {result['error']}")
                continue
            
            if "status" in result:
                log_analysis_step(f"Status update: {result['status']}", "debug")
                status_placeholder.write(result["status"])
                continue
                
            try:
                question_id = result.get('question_id')
                if question_id is None:
                    continue
                
                # Store results
                results["answers"][question_id] = result
                
                # Create DataFrames from results
                analysis_df, chunks_df = create_analysis_dataframes(
                    results["answers"], 
                    Path(file_path).name
                )
                
                # Store in session state for display
                st.session_state.analysis_df = analysis_df
                st.session_state.chunks_df = chunks_df
                        
            except Exception as e:
                log_analysis_step(f"Unexpected error processing result: {str(e)}", "error")
                log_analysis_step(traceback.format_exc(), "error")
                st.error(f"Error processing result: {str(e)}")
                continue
        
        # Store final results in session state
        st.session_state.results = results
        st.session_state.analysis_complete = True
        
        # Clear the status placeholder
        status_placeholder.empty()
        
    except Exception as e:
        log_analysis_step(f"Critical error during analysis: {str(e)}", "error")
        log_analysis_step(traceback.format_exc(), "error")
        st.error(f"Error during analysis: {str(e)}")

def display_final_results(analysis_df: pd.DataFrame, chunks_df: pd.DataFrame):
    """Display the final results including both tables"""
    # Analysis Results
    st.subheader("Analysis Results")
    st.dataframe(
        analysis_df,
        column_config={
            "Score": st.column_config.NumberColumn(
                "Score",
                help="Analysis score out of 10",
                min_value=0,
                max_value=10,
                format="%.1f"
            ),
            "Analysis": st.column_config.TextColumn(
                "Analysis",
                width="large"
            ),
            "Key Evidence": st.column_config.TextColumn(
                "Key Evidence",
                width="medium"
            )
        },
        use_container_width=True,
        hide_index=True
    )
    
    # Document Chunks
    st.subheader("Document Chunks")
    st.dataframe(
        chunks_df,
        column_config={
            "Question ID": st.column_config.SelectboxColumn(
                "Question ID",
                help="The question this chunk belongs to",
                width="medium",
                options=chunks_df["Question ID"].unique().tolist()
            ),
            "Vector Similarity": st.column_config.NumberColumn(
                "Vector Similarity",
                help="Similarity score between chunk and question",
                min_value=0,
                max_value=1,
                format="%.3f"
            ),
            "LLM Score": st.column_config.NumberColumn(
                "LLM Score",
                help="LLM-computed relevance score",
                min_value=0,
                max_value=1,
                format="%.3f"
            ),
            "Chunk Text": st.column_config.TextColumn(
                "Chunk Text",
                help="Text content of the chunk",
                width="large"
            ),
            "Evidence Reference": st.column_config.CheckboxColumn(
                "Used as Evidence",
                help="Whether this chunk was referenced in the analysis"
            ),
            "Position in Question": st.column_config.NumberColumn(
                "Position",
                help="Position of chunk within question results",
                min_value=0
            )
        },
        use_container_width=True,
        hide_index=False,
        filters=True  # Enable filtering
    )

def main():
    st.set_page_config(
        page_title="Report Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    # Initialize session state variables if they don't exist
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'selected_questions' not in st.session_state:
        st.session_state.selected_questions = []
    if 'analysis_triggered' not in st.session_state:
        st.session_state.analysis_triggered = False
    
    st.title("Report Analyzer")
    st.write("Upload a PDF report and select questions for sustainability report analysis.")
    
    try:
        # Initialize analyzer
        analyzer = ReportAnalyzer()
        
        # Advanced options in expander
        with st.expander("Advanced Options"):
            # Use session state to persist advanced options
            if 'use_llm_scoring' not in st.session_state:
                st.session_state.use_llm_scoring = False
            if 'single_call' not in st.session_state:
                st.session_state.single_call = True
            if 'use_cache' not in st.session_state:
                st.session_state.use_cache = True
                
            st.session_state.use_llm_scoring = st.checkbox(
                "Use LLM for relevance scoring",
                value=st.session_state.use_llm_scoring
            )
            
            if st.session_state.use_llm_scoring:
                st.session_state.single_call = st.checkbox(
                    "Score all chunks in single LLM call",
                    value=st.session_state.single_call,
                    help="More efficient but may be less accurate with large numbers of chunks"
                )
            
            st.session_state.use_cache = st.checkbox(
                "Use LLM Cache",
                value=st.session_state.use_cache,
                help="Cache LLM responses to improve performance for repeated queries"
            )
            
            # Cache clear button
            if st.button("Clear Cache"):
                if hasattr(st.session_state, 'llm_cache'):
                    del st.session_state.llm_cache
                st.success("Cache cleared!")
        
        # Question set selection using session state
        if 'current_question_set' not in st.session_state:
            st.session_state.current_question_set = "tcfd"
            
        # File upload handling with session state
        if 'uploaded_file' not in st.session_state:
            st.session_state.uploaded_file = None
            
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
        
        if st.session_state.uploaded_file:
            try:
                # Load questions and handle selection
                question_set_data = analyzer.load_question_set(st.session_state.current_question_set)
                questions = question_set_data["questions"]
                
                if question_set_data["description"]:
                    st.write(question_set_data["description"])
                
                # Question selection with session state
                st.subheader("Select Questions for Analysis")
                selected_questions = []
                for q_id, q_data in questions.items():
                    if st.checkbox(
                        q_data['text'],
                        key=f"question_{q_id}",
                        value=q_id in st.session_state.selected_questions
                    ):
                        selected_questions.append(q_id)
                st.session_state.selected_questions = selected_questions
                
                # Create a single results container
                results_container = st.container()
                
                # Analysis trigger with session state control
                if st.button("Analyze Document") or st.session_state.analysis_triggered:
                    st.session_state.analysis_triggered = True
                    
                    if not st.session_state.analysis_complete:
                        with st.spinner("Analyzing document..."):
                            file_path = save_uploaded_file(st.session_state.uploaded_file)
                            if file_path:
                                asyncio.run(analyze_document_and_display(
                                    analyzer,
                                    file_path,
                                    questions,
                                    st.session_state.selected_questions,
                                    st.session_state.use_llm_scoring,
                                    st.session_state.single_call
                                ))
                
                # Display results if available
                if hasattr(st.session_state, 'analysis_df'):
                    with results_container:
                        display_final_results(st.session_state.analysis_df, st.session_state.chunks_df)
                        
                        if st.session_state.analysis_complete:
                            display_download_buttons(st.session_state.analysis_df, st.session_state.chunks_df)
                            
                            if st.button("Clear Results", key="clear_results_button"):
                                st.session_state.analysis_complete = False
                                st.session_state.analysis_triggered = False
                                st.session_state.selected_questions = []
                                if hasattr(st.session_state, 'analysis_df'):
                                    del st.session_state.analysis_df
                                if hasattr(st.session_state, 'chunks_df'):
                                    del st.session_state.chunks_df
                                st.experimental_rerun()

            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")
                
    except Exception as e:
        st.error(f"Error initializing analyzer: {str(e)}")

if __name__ == "__main__":
    main() 