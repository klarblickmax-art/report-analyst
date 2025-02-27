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
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import glob  # Add at the top of the file

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

LLM_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-3.5-turbo",
]

question_sets = {
    "tcfd": {
        "name": "TCFD Questions",
        "description": "Task Force on Climate-related Financial Disclosures questions"
    },
    "s4m": {
        "name": "S4M Questions",
        "description": "Sustainability for Management questions"
    }
}

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
    
    async def analyze_document(self, file_path: str, questions: Dict, selected_questions: List[str], use_llm_scoring: bool = False, single_call: bool = True, force_recompute: bool = False) -> AsyncGenerator[Dict, None]:
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
            self.analyzer.question_set = question_set
            
            # Pass use_llm_scoring to process_document
            async for result in self.analyzer.process_document(
                file_path, 
                selected_numbers, 
                use_llm_scoring, 
                single_call,
                force_recompute
            ):
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
            
        # If it's already a path, just return it
        if isinstance(uploaded_file, (str, Path)):
            return str(uploaded_file)
            
        # Check if file was already saved in this session
        file_key = f"saved_file_{uploaded_file.name}"
        if file_key in st.session_state:
            return st.session_state[file_key]
            
        # Otherwise, handle it as an UploadedFile
        file_path = Path("temp") / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Successfully saved file: {file_path}")
        
        # Store the path in session state
        st.session_state[file_key] = str(file_path)
        # Reset file processing flag when a new file is saved
        st.session_state.file_processed = False
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

def convert_df(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to CSV bytes"""
    return df.to_csv(index=False).encode('utf-8')

def display_download_buttons(analysis_df: pd.DataFrame, chunks_df: pd.DataFrame, file_key: str):
    """Display download buttons for analysis results"""
    # Generate unique timestamp for this render
    timestamp = int(time.time() * 1000)
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = convert_df(analysis_df)
        st.download_button(
            label="Download Analysis Results",
            data=csv,
            file_name=f"analysis_{file_key}.csv",
            mime="text/csv",
            key=f"download_analysis_{file_key}_{timestamp}"
        )
    
    with col2:
        csv = convert_df(chunks_df)
        st.download_button(
            label="Download Chunks Data",
            data=csv,
            file_name=f"chunks_{file_key}.csv",
            mime="text/csv",
            key=f"download_chunks_{file_key}_{timestamp}"
        )

def generate_file_key(file_path: str, st) -> str:
    """Generate a cache file key from file path and settings"""
    return (f"{Path(file_path).name}_"
            f"cs{st.session_state.new_chunk_size}_"
            f"ov{st.session_state.new_overlap}_"
            f"tk{st.session_state.new_top_k}_"
            f"m{st.session_state.new_llm_model}")

async def analyze_document_and_display(analyzer, file_path: str, questions: Dict, selected_questions: List[str], use_llm_scoring: bool = False, single_call: bool = True, force_recompute: bool = False):
    """Analyze document and display results as they come in"""
    try:
        selected_questions_list = list(selected_questions) if selected_questions else []
        question_set = selected_questions_list[0].split('_')[0] if selected_questions_list else "tcfd"
        
        # Use the helper function to generate file key
        file_key = generate_file_key(file_path, st)
        
        # Initialize or clear results if question set changed
        if ('results' not in st.session_state or 
            'current_question_set' not in st.session_state or 
            st.session_state.current_question_set != question_set):
            st.session_state.results = {"answers": {}}
            st.session_state.current_question_set = question_set
            st.session_state.analyzed_files = set()
            
        # Create status placeholder
        status_placeholder = st.empty()
            
        log_analysis_step(f"Starting analysis with question set: {question_set}")
        
        # Load cached answers first
        cached_answers = {} if force_recompute else analyzer.analyzer._load_cached_answers(file_path)
        
        if cached_answers:
            log_analysis_step(f"Found {len(cached_answers)} cached answers for {file_key}")
            # Show cache info
            st.info(f"📁 Loading results from cache: {file_key}")
            
            # Filter cached answers to only include current question set
            cached_answers = {k: v for k, v in cached_answers.items() 
                            if k.startswith(question_set)}
            
            if cached_answers:
                # Immediately update results with cached answers
                for q_id, answer in cached_answers.items():
                    st.session_state.results["answers"][q_id] = answer
                
                # Update display with cached results
                analysis_df, chunks_df = create_analysis_dataframes(
                    st.session_state.results["answers"], 
                    file_key
                )
                st.session_state.analysis_df = analysis_df
                st.session_state.chunks_df = chunks_df
        
        # Determine which questions need processing
        questions_to_process = [q_id for q_id in selected_questions_list 
                              if force_recompute or q_id not in cached_answers]
        
        if questions_to_process:
            log_analysis_step(f"Processing {len(questions_to_process)} uncached questions...")
            
            # Update analyzer with question set
            analyzer.analyzer.question_set = question_set
            
            # Process only uncached questions
            async for result in analyzer.analyze_document(
                file_path, 
                questions,
                questions_to_process,
                use_llm_scoring, 
                single_call,
                force_recompute
            ):
                if "error" in result:
                    log_analysis_step(f"Error received from analyzer: {result['error']}", "error")
                    st.error(f"Analysis error: {result['error']}")
                    continue
                
                if "status" in result:
                    status_placeholder.write(result["status"])
                    continue
                    
                question_id = result.get('question_id')
                if question_id is None:
                    continue
                
                # Store results
                st.session_state.results["answers"][question_id] = result
                
                # Update display
                analysis_df, chunks_df = create_analysis_dataframes(
                    st.session_state.results["answers"], 
                    file_key
                )
                
                st.session_state.analysis_df = analysis_df
                st.session_state.chunks_df = chunks_df
        else:
            log_analysis_step("All selected questions have cached answers")
            # Show success message for cached results
            st.success(f"✓ All {len(selected_questions_list)} selected questions loaded from cache")
        
        # Mark this file as analyzed with current configuration
        if 'analyzed_files' not in st.session_state:
            st.session_state.analyzed_files = set()
        st.session_state.analyzed_files.add(file_key)
        
        # Clear status and mark as complete
        status_placeholder.empty()
        st.session_state.analysis_complete = True
        
    except Exception as e:
        log_analysis_step(f"Critical error during analysis: {str(e)}", "error")
        log_analysis_step(traceback.format_exc(), "error")
        st.error(f"Error during analysis: {str(e)}")

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

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
        filter_dataframe(chunks_df),  # Apply the filter function
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
        hide_index=False
    )

def load_question_sets() -> Dict[str, str]:
    """Load all available question sets and their descriptions"""
    question_sets = {}
    question_sets_dir = Path(__file__).parent / "questionsets"
    
    for yaml_file in question_sets_dir.glob("*_questions.yaml"):
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                # Get set name from filename (e.g., 'tcfd' from 'tcfd_questions.yaml')
                set_id = yaml_file.stem.replace('_questions', '')
                question_sets[set_id] = {
                    'name': data.get('name', set_id.upper()),
                    'description': data.get('description', '')
                }
        except Exception as e:
            logger.error(f"Error loading question set {yaml_file}: {e}")
    
    return question_sets

def get_uploaded_files_history() -> List[Dict]:
    """Get list of previously uploaded files from temp directory"""
    temp_dir = Path("temp")
    if not temp_dir.exists():
        return []
    
    files = []
    for file in temp_dir.glob("*.pdf"):
        # Verify file exists and is not empty
        if file.exists() and file.stat().st_size > 0:
            files.append({
                'name': file.name,
                'path': str(file.resolve()),  # Get absolute path
                'date': file.stat().st_mtime,
                'size': file.stat().st_size
            })
            logger.info(f"Found file: {file.name}, size: {file.stat().st_size} bytes")
    
    # Sort by most recent first
    return sorted(files, key=lambda x: x['date'], reverse=True)

def get_all_cached_answers(question_set: str) -> Dict:
    """Get all cached answers for a given question set"""
    # Use relative path from where the app is run
    cache_path = Path("storage/cache")
    
    if not cache_path.exists():
        log_analysis_step(f"Cache directory not found: {cache_path}")
        return {}
        
    log_analysis_step(f"Looking for cached answers in: {cache_path}")
    all_answers = {}
    
    # Look for all cache files for this question set
    pattern = f"*_{question_set}_answers.json"
    log_analysis_step(f"Searching for files matching: {pattern}")
    
    for cache_file in cache_path.glob(pattern):
        try:
            log_analysis_step(f"Found cache file: {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                answers = json.load(f)
                # Extract report name from filename (first part before underscore)
                report_name = cache_file.stem.split('_')[0]
                all_answers[report_name] = answers
                log_analysis_step(f"Loaded {len(answers)} answers for {report_name}")
        except Exception as e:
            log_analysis_step(f"Error loading cache file {cache_file}: {str(e)}", "warning")
    
    if not all_answers:
        log_analysis_step(f"No cached answers found for question set {question_set}")
    else:
        log_analysis_step(f"Found cached answers for {len(all_answers)} reports")
    
    return all_answers

def create_coverage_matrix(question_set: str) -> pd.DataFrame:
    """Create a matrix showing which reports have answers for which questions"""
    all_answers = get_all_cached_answers(question_set)
    
    # Get all unique question IDs
    all_questions = set()
    for answers in all_answers.values():
        all_questions.update(answers.keys())
    
    # Create matrix
    matrix_data = []
    for report_name, answers in all_answers.items():
        row = {'Report': report_name}
        for q_id in sorted(all_questions):
            row[q_id] = '✓' if q_id in answers else ''
        matrix_data.append(row)
    
    return pd.DataFrame(matrix_data)

def display_consolidated_results(question_set: str):
    """Display consolidated results for all reports in a question set"""
    all_answers = get_all_cached_answers(question_set)
    
    if not all_answers:
        st.warning(f"No cached answers found for question set {question_set}")
        return
        
    # Create consolidated DataFrames
    all_analysis = []
    all_chunks = []
    
    for report_name, answers in all_answers.items():
        analysis_df, chunks_df = create_analysis_dataframes(answers, report_name)
        all_analysis.append(analysis_df)
        all_chunks.append(chunks_df)
    
    consolidated_analysis = pd.concat(all_analysis, ignore_index=True)
    consolidated_chunks = pd.concat(all_chunks, ignore_index=True)
    
    # Display coverage matrix
    st.subheader("Analysis Coverage")
    coverage_matrix = create_coverage_matrix(question_set)
    st.dataframe(coverage_matrix, use_container_width=True)
    
    # Display consolidated results
    st.subheader("All Analysis Results")
    st.dataframe(consolidated_analysis, use_container_width=True)
    
    st.subheader("All Document Chunks")
    st.dataframe(consolidated_chunks, use_container_width=True)
    
    # Add download buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "Download Coverage Matrix (CSV)",
            coverage_matrix.to_csv(index=False),
            f"{question_set}_coverage_matrix.csv",
            "text/csv"
        )
    with col2:
        st.download_button(
            "Download All Analysis (CSV)",
            consolidated_analysis.to_csv(index=False),
            f"{question_set}_all_analysis.csv",
            "text/csv"
        )
    with col3:
        st.download_button(
            "Download All Chunks (CSV)",
            consolidated_chunks.to_csv(index=False),
            f"{question_set}_all_chunks.csv",
            "text/csv"
        )

def get_available_caches(base_filename: str) -> List[str]:
    """Get list of available cache files for a given base filename"""
    # Look for cache files matching the pattern
    cache_pattern = f"*{Path(base_filename).stem}*_answers.json"
    cache_files = glob.glob(f"cache/{cache_pattern}")
    # Extract just the configuration part of the filename
    return [Path(f).stem.replace('_answers', '') for f in cache_files]

def display_cache_selector(file_path: str):
    """Display dropdown with available cache files"""
    available_caches = get_available_caches(file_path)
    if available_caches:
        st.subheader("Available Cache Files")
        selected_cache = st.selectbox(
            "Select cached analysis to load",
            options=available_caches,
            format_func=lambda x: f"Cache: {x}",
            key=f"cache_selector_{int(time.time() * 1000)}"
        )
        if st.button("Load Selected Cache"):
            # Extract parameters from cache filename
            parts = selected_cache.split('_')
            # Store parameters to use as default values for the UI widgets
            st.session_state.new_chunk_size = 500  # default value
            st.session_state.new_overlap = 20      # default value
            st.session_state.new_top_k = 5         # default value
            st.session_state.new_llm_model = LLM_MODELS[0]  # default value
            
            for part in parts:
                if part.startswith('cs'):
                    st.session_state.new_chunk_size = int(part[2:])
                elif part.startswith('ov'):
                    st.session_state.new_overlap = int(part[2:])
                elif part.startswith('tk'):
                    st.session_state.new_top_k = int(part[2:])
                elif part.startswith('m'):
                    st.session_state.new_llm_model = part[1:]
            
            # Trigger rerun to update the UI widgets with new values
            st.experimental_rerun()
    else:
        st.info("No cached analyses available for this file")

def get_current_settings(st) -> dict:
    """Get all current settings from the UI widgets"""
    # Get first question set as default
    default_set = list(question_sets.keys())[0]
    
    return {
        'chunk_size': st.session_state.get('new_chunk_size', 500),
        'overlap': st.session_state.get('new_overlap', 20),
        'top_k': st.session_state.get('new_top_k', 5),
        'llm_model': st.session_state.get('new_llm_model', LLM_MODELS[0]),
        'use_llm_scoring': st.session_state.get('new_llm_scoring', False),
        'batch_scoring': st.session_state.get('new_batch_scoring', True),
        'selected_set': st.session_state.get('new_question_set', default_set)
    }

def main():
    try:
        st.set_page_config(page_title="Report Analyst", layout="wide")
        
        # Initialize analyzer with default question set
        try:
            analyzer = ReportAnalyzer()
        except Exception as e:
            st.error(f"Error initializing analyzer: {str(e)}")
            st.exception(e)
            return

        st.title("Report Analyst")
        
        # Settings section - moved below the title
        with st.expander("Analysis Configuration", expanded=True):
            # Question set selection
            col1, col2 = st.columns([1, 2])
            with col1:
                selected_set = st.selectbox(
                    "Select Question Set",
                    options=list(question_sets.keys()),
                    format_func=lambda x: question_sets[x]['name'],
                    key="new_question_set",
                    index=0  # Ensure a default is selected
                )
            
            # Show question set description
            with col2:
                if selected_set in question_sets:
                    st.info(question_sets[selected_set]['description'])
            
            # Update analyzer's question set
            analyzer.analyzer.update_question_set(selected_set)
            
            # Clear results if question set changed
            if ('last_question_set' not in st.session_state or 
                st.session_state.last_question_set != selected_set):
                if 'results' in st.session_state:
                    del st.session_state.results
                st.session_state.last_question_set = selected_set
            
            # LLM settings
            col1, col2, col3 = st.columns(3)
            with col1:
                new_llm_scoring = st.checkbox("Use LLM Scoring", value=False, key="new_llm_scoring")
            with col2:
                new_batch_scoring = st.checkbox("Batch Scoring", value=True, key="new_batch_scoring")
            with col3:
                new_llm_model = st.selectbox(
                    "LLM Model", 
                    options=LLM_MODELS,
                    index=0,  # Ensure a default is selected
                    key="new_llm_model"
                )
            
            # Chunking parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                new_chunk_size = st.number_input(
                    "Chunk Size",
                    min_value=100,
                    max_value=2000,
                    value=500,  # Default value
                    key="new_chunk_size"
                )
            with col2:
                new_overlap = st.number_input(
                    "Overlap",
                    min_value=0,
                    max_value=100,
                    value=20,  # Default value
                    key="new_overlap"
                )
            with col3:
                new_top_k = st.number_input(
                    "Top K",
                    min_value=1,
                    max_value=20,
                    value=5,  # Default value
                    key="new_top_k"
                )

        # Now these values will be available in session_state for the rest of the app
        # We can access them as st.session_state.new_chunk_size, etc.

        # Use the selected settings from session state
        selected_set = st.session_state.get('selected_set', selected_set)
        
        # File selection/upload
        st.subheader("Select Report")
        file_tab, upload_tab, consolidated_tab = st.tabs(["Previous Reports", "Upload New", "Consolidated Results"])
        
        # Reset analysis state when switching files
        if 'current_file' not in st.session_state:
            st.session_state.current_file = None
        
        with file_tab:
            previous_files = get_uploaded_files_history()
            if previous_files:
                    
                selected_file = st.selectbox(
                    "Select a previously analyzed report",
                    options=previous_files,
                    format_func=lambda x: x['name'],
                    key="previous_file"
                )
                if selected_file:
                    file_path = Path(selected_file['path'])
                    if file_path.exists():
                        # Update current file and trigger analysis
                        if str(file_path) != st.session_state.current_file:
                            st.session_state.current_file = str(file_path)
                            st.session_state.uploaded_file = file_path
                            # Reset analysis state but keep results
                            st.session_state.analysis_complete = False
                            st.session_state.analysis_triggered = True  # Set to True to trigger analysis
                            st.experimental_rerun()
                    else:
                        st.error(f"File not found: {file_path}")
            else:
                st.info("No previously analyzed reports found")
        
        with upload_tab:
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="file_uploader")
            if uploaded_file:
                # Save the uploaded file and update session state
                file_path = save_uploaded_file(uploaded_file)
                if file_path and file_path != st.session_state.get('current_file'):
                    st.session_state.current_file = file_path
                    st.session_state.uploaded_file = uploaded_file
                    # Reset analysis state
                    st.session_state.analysis_complete = False
                    st.session_state.analysis_triggered = False
                    if 'results' in st.session_state:
                        del st.session_state.results
                    st.success(f"File uploaded successfully: {uploaded_file.name}")
                    # Only rerun if we haven't processed this file yet
                    if not st.session_state.get('file_processed'):
                        st.session_state.file_processed = True
                        st.rerun()
        
        if st.session_state.get('uploaded_file'):
            try:
                # Load questions and handle selection
                question_set_data = analyzer.load_question_set(selected_set)
                questions = question_set_data["questions"]
                
                if question_set_data["description"]:
                    st.write(question_set_data["description"])
                
                # Add question selection UI with Select All toggle
                st.subheader("Select Questions")
                
                # Initialize selected_questions in session state if not present
                if 'selected_questions' not in st.session_state:
                    st.session_state.selected_questions = list(questions.keys())
                
                # Select All toggle
                select_all = st.checkbox("Select All Questions", 
                                       value=len(st.session_state.selected_questions) == len(questions),
                                       key="select_all")
                
                # Handle select all / deselect all
                if select_all:
                    st.session_state.selected_questions = list(questions.keys())
                else:
                    st.session_state.selected_questions = []
                
                # Individual question checkboxes
                selected_questions = []
                for q_id, q_data in questions.items():
                    if st.checkbox(
                        f"{q_id}: {q_data['text']}", 
                        value=q_id in st.session_state.selected_questions,
                        key=f"question_{q_id}"
                    ):
                        selected_questions.append(q_id)
                
                # Update session state
                st.session_state.selected_questions = selected_questions
                
                # Create a single results container
                results_container = st.container()
                
                # Track configuration changes
                current_config = get_current_settings(st)
                
                # Check if configuration has changed
                if 'last_config' not in st.session_state or st.session_state.last_config != current_config:
                    # Clear results and trigger reanalysis
                    if 'results' in st.session_state:
                        del st.session_state.results
                    if 'analysis_df' in st.session_state:
                        del st.session_state.analysis_df
                    if 'chunks_df' in st.session_state:
                        del st.session_state.chunks_df
                    if 'analyzed_files' in st.session_state:
                        del st.session_state.analyzed_files
                    st.session_state.analysis_complete = False
                    st.session_state.analysis_triggered = True
                    st.session_state.last_config = current_config
                    st.experimental_rerun()
                
                # Add analysis trigger with proper state handling
                if st.button("Analyze Selected Questions", key="analyze_button"):
                    if not selected_questions:
                        st.warning("Please select at least one question to analyze.")
                    else:
                        st.session_state.analysis_triggered = True
                        
                        # Show progress
                        progress_text = st.empty()
                        progress_text.info("Starting analysis...")
                        
                        # Generate current file key based on configuration
                        file_key = generate_file_key(st.session_state.current_file, st)
                        
                        # Check cache first
                        cached = analyzer.analyzer._load_cached_answers(st.session_state.current_file)
                        
                        # Filter cached answers to only include current question set
                        if cached:
                            cached = {k: v for k, v in cached.items() 
                                    if k in selected_questions}
                            
                            if cached:
                                progress_text.info(f"Found {len(cached)} cached answers")
                                # Update results with cached answers
                                st.session_state.results = {"answers": cached}
                                
                                # Create dataframes from cached results
                                analysis_df, chunks_df = create_analysis_dataframes(
                                    cached,
                                    file_key
                                )
                                st.session_state.analysis_df = analysis_df
                                st.session_state.chunks_df = chunks_df
                        
                        # Determine which questions need processing
                        questions_to_process = [q_id for q_id in selected_questions 
                                              if q_id not in cached]
                        
                        if questions_to_process:
                            progress_text.info(f"Processing {len(questions_to_process)} uncached questions...")
                            
                            # Update analyzer parameters
                            analyzer.analyzer.update_parameters(
                                chunk_size=st.session_state.new_chunk_size,
                                chunk_overlap=st.session_state.new_overlap,
                                top_k=st.session_state.new_top_k
                            )
                            analyzer.analyzer.update_llm_model(st.session_state.new_llm_model)
                            
                            try:
                                # Run analysis for uncached questions
                                asyncio.run(analyze_document_and_display(
                                    analyzer=analyzer,
                                    file_path=st.session_state.current_file,
                                    questions=questions,
                                    selected_questions=questions_to_process,
                                    use_llm_scoring=st.session_state.new_llm_scoring,
                                    single_call=st.session_state.new_batch_scoring
                                ))
                                
                                progress_text.success("Analysis complete!")
                                
                            except Exception as e:
                                st.error(f"Error during analysis: {str(e)}")
                                st.exception(e)
                        else:
                            progress_text.success("All selected questions loaded from cache!")
                        
                        # Display results if available
                        if hasattr(st.session_state, 'analysis_df'):
                            display_final_results(
                                st.session_state.analysis_df,
                                st.session_state.chunks_df
                            )

            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")
            
        with consolidated_tab:
            st.subheader("View All Results")
            st.write("View and export consolidated results for all analyzed reports")
            
            # Question set selection for consolidated view
            selected_set = st.selectbox(
                "Select Question Set",
                options=list(question_sets.keys()),
                format_func=lambda x: question_sets[x]['name'],
                key="consolidated_set"
            )
            
            if selected_set:
                display_consolidated_results(selected_set)

        # Add Climate+Tech footer at the end
        footer = """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            color: black;
            text-align: center;
            padding: 10px;
            font-size: 14px;
        }
        .footer img {
            height: 30px;
            vertical-align: middle;
            margin-right: 10px;
        }
        </style>
        <div class="footer">
            <img src="https://www.climateandtech.com/climateandtech.png" alt="Climate+Tech Logo">
            <p>Climate+Tech Sustainability Report Analysis Tool</p>
            <p>For custom tool development contact us at <a href="https://www.climateandtech.com" target="_blank">www.climateandtech.com</a></p>
        </div>
        """
        st.markdown(footer, unsafe_allow_html=True)

        # Add this to the main function where results are displayed
        if hasattr(st.session_state, 'current_file'):
            with st.expander("Available Cache Files", expanded=False):
                display_cache_selector(st.session_state.current_file)

    except Exception as e:
        st.error("Error during analysis:")
        st.exception(e)  # This will show the full traceback

if __name__ == "__main__":
    main() 