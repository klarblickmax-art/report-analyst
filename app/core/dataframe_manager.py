import pandas as pd
import json
from typing import Dict, Tuple, Any, List

def format_list_field(field: Any) -> str:
    """Format list fields for better display"""
    if isinstance(field, str):
        try:
            # Try to parse if it's a string representation of a list
            field = eval(field)
        except:
            return field
            
    if isinstance(field, list):
        formatted_items = []
        for item in field:
            if isinstance(item, dict):
                # Handle evidence items with text and chunk info
                text = item.get('text', '')
                chunk = item.get('chunk', 'Unknown')
                formatted_items.append(f"• {text} [Chunk {chunk}]")
            else:
                formatted_items.append(f"• {str(item)}")
        return "\n".join(formatted_items)
    return str(field)

def extract_evidence_text(evidence: Any) -> str:
    """Extract text from evidence items"""
    if isinstance(evidence, dict):
        text = evidence.get('text', '')
        chunk = evidence.get('chunk', 'Unknown')
        return f"{text} [Chunk {chunk}]"
    return str(evidence)

def create_analysis_dataframes(answers: Dict, report_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create analysis and chunks dataframes from the answers dictionary"""
    analysis_rows = []
    chunks_rows = []
    
    for question_id, data in answers.items():
        try:
            result = json.loads(data['result'])
            
            # Add to analysis dataframe
            analysis_rows.append({
                'Report Name': report_name,
                'Question ID': question_id,
                'Question Text': data.get('question_text', ''),
                'Score': result.get('SCORE', 0),
                'Analysis': result.get('ANSWER', ''),
                'Key Evidence': format_list_field(result.get('EVIDENCE', [])),
                'Gaps': format_list_field(result.get('GAPS', [])),
                'Areas for Improvement': format_list_field(result.get('AREAS_FOR_IMPROVEMENT', []))
            })
            
            # Add chunks with position within question
            chunks = result.get('CHUNKS', [])
            for idx, chunk in enumerate(chunks):
                chunks_rows.append({
                    'Report Name': report_name,
                    'Question ID': question_id,
                    'Position in Question': idx,  # Track position within question
                    'Chunk Text': chunk['text'],
                    'Vector Similarity': chunk['relevance_score'],
                    'LLM Score': chunk.get('computed_score', 0.0),
                    'Evidence Reference': is_chunk_referenced(idx, result.get('EVIDENCE', []))
                })
    
        except Exception as e:
            logger.error(f"Error processing answer for question {question_id}: {str(e)}")
            continue

    # Create DataFrames
    analysis_df = pd.DataFrame(analysis_rows)
    chunks_df = pd.DataFrame(chunks_rows)
    
    # Sort chunks by Question ID and Position to maintain order
    chunks_df = chunks_df.sort_values(['Question ID', 'Position in Question'])
    
    # Add a global index that maintains the sorting
    chunks_df = chunks_df.reset_index(drop=True)
    chunks_df.index.name = 'Global ID'
    
    return analysis_df, chunks_df

def is_chunk_referenced(position: int, evidence_list: List[Dict]) -> bool:
    """Check if a chunk is referenced in the evidence list"""
    for evidence in evidence_list:
        if evidence.get('chunk') == position:
            return True
    return False

def create_combined_dataframe(analysis_df: pd.DataFrame, chunks_df: pd.DataFrame) -> pd.DataFrame:
    """Create a combined multi-index dataframe"""
    if analysis_df.empty or chunks_df.empty:
        return pd.DataFrame()
        
    # Create multi-index dataframes
    analysis_indexed = analysis_df.set_index(['Report Name', 'Question ID'])
    chunks_indexed = chunks_df.set_index(['Report Name', 'Question ID'])
    
    # Combine the dataframes
    combined_df = pd.concat([analysis_indexed, chunks_indexed], axis=1)
    
    # Remove duplicate columns
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    
    return combined_df 