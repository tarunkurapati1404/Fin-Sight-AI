"""
MLFLOW EXPERIMENT TRACKER

Logs every RAG query as an experiment for monitoring and debugging.

WHAT GETS LOGGED:
- Parameters: User question, model used, number of sources retrieved
- Metrics: Timing (embedding, search, LLM, total), token usage, costs, similarity scores
- Artifacts: Generated answer saved as text file
- Tags: Category of FAQs used, success/failure status

WHY THIS MATTERS:
Interviewer: "How do you monitor your AI system in production?"
You: "I use MLflow to track every query - parameters, metrics, artifacts.
     I can debug issues, compare performance over time, and show stakeholders
     usage statistics via the MLflow dashboard."
"""

import mlflow
import os
import time
import json
from datetime import datetime
from config import MLFLOW_TRACKING_URI


def setup_mlflow():
    """
    Initialize MLflow experiment tracking.
    Creates local directory to store experiment data.
    """
    # Set tracking URI (where data is stored)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # "./mlruns" from config
    
    # Set experiment name (groups related runs together)
    experiment_name = "fin_sight_ai_rag_pipeline"
    
    # Create/get experiment (returns experiment ID)
    mlflow.set_experiment(experiment_name)
    
    print(f"📊 MLflow initialized")
    print(f"   Experiment: {experiment_name}")
    print(f"   Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"   View UI: mlflow ui")
    
    return experiment_name


def log_rag_query(result):
    """
    Log a single RAG query result to MLflow.
    
    This function should be called AFTER ask_question() completes.
    
    Args:
        result (dict): The complete result dictionary from rag_pipeline.ask_question()
    
    Returns:
        str: The MLflow run ID (for reference)
    """
    if not result.get('success'):
        # Don't log failed queries (or log them separately)
        return None
    
    # Start MLflow run (auto-generates unique run ID)
    with mlflow.start_run(run_name=f"query_{int(time.time())}") as run:
        run_id = run.info.run_id
        
        # ============================================
        # LOG PARAMETERS (inputs/configuration)
        # ============================================
        mlflow.log_param("question", result['question'])
        mlflow.log_param("model_used", result.get('model_used', 'unknown'))
        mlflow.log_param("num_sources_retrieved", len(result.get('sources', [])))
        mlflow.log_param("top_k_results", len(result.get('sources', [])))
        
        # Log source categories used
        categories = [s['category'] for s in result.get('sources', [])]
        mlflow.log_param("source_categories", str(categories))
        
        # Log similarity scores of sources
        scores = [s['score'] for s in result.get('sources', [])]
        mlflow.log_param("similarity_scores", str(scores))
        
        # ============================================
        # LOG METRICS (outputs/performance)
        # ============================================
        timing = result.get('timing', {})
        
        # Timing metrics (in seconds)
        mlflow.log_metric("timing_embedding_sec", timing.get('embedding_generation', 0))
        mlflow.log_metric("timing_vector_search_sec", timing.get('vector_search', 0))
        mlflow.log_metric("timing_context_assembly_sec", timing.get('context_assembly', 0))
        mlflow.log_metric("timing_llm_generation_sec", timing.get('llm_generation', 0))
        mlflow.log_metric("timing_total_sec", timing.get('total_pipeline', 0))
        
        # Token usage metrics
        usage = result.get('token_usage')
        if usage:
            mlflow.log_metric("tokens_prompt", usage.get('prompt_tokens', 0))
            mlflow.log_metric("tokens_completion", usage.get('completion_tokens', 0))
            mlflow.log_metric("tokens_total", usage.get('total_tokens', 0))
            
            # Calculate cost (approximate)
            cost_input = usage.get('prompt_tokens', 0) * 0.0000015  # GPT-3.5 input price
            cost_output = usage.get('completion_tokens', 0) * 0.000002  # GPT-3.5 output price
            total_cost = cost_input + cost_output
            
            mlflow.log_metric("cost_usd", total_cost)
        
        # Quality metrics (based on similarity scores)
        if scores:
            mlflow.log_metric("avg_similarity_score", sum(scores) / len(scores))
            mlflow.log_metric("max_similarity_score", max(scores))
            mlflow.log_metric("min_similarity_score", min(scores))
        
        # Answer length metric
        answer_length = len(result.get('answer', ''))
        mlflow.log_metric("answer_length_chars", answer_length)
        
        # ============================================
        # LOG ARTIFACTS (files)
        # ============================================
        
        # Save generated answer as text file
        artifact_dir = "artifacts"
        os.makedirs(artifact_dir, exist_ok=True)
        
        answer_file = f"{artifact_dir}/answer_{run_id[:8]}.txt"
        with open(answer_file, 'w', encoding='utf-8') as f:
            f.write(f"QUESTION:\n{result['question']}\n\n")
            f.write(f"ANSWER:\n{result['answer']}\n\n")
            f.write(f"SOURCES:\n")
            for i, src in enumerate(result.get('sources', []), 1):
                f.write(f"\n--- Source {i} [Score: {src['score']}] ---\n")
                f.write(f"Q: {src['question']}\n")
                f.write(f"A: {src['answer']}\n")
            f.write(f"\n\nTIMING:\n{json.dumps(timing, indent=2)}\n")
        
        mlflow.log_artifact(answer_file)
        
        # Save full result as JSON for debugging
        json_file = f"{artifact_dir}/result_{run_id[:8]}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        
        mlflow.log_artifact(json_file)
        
        # ============================================
        # LOG TAGS (metadata for filtering)
        # ============================================
        mlflow.set_tag("project", "fin-sight-ai")
        mlflow.set_tag("pipeline_stage", "production")
        mlflow.set_tag("timestamp", datetime.now().isoformat())
        mlflow.set_tag("primary_category", categories[0] if categories else "unknown")
        
        print(f"✅ Logged to MLflow: {run_id[:8]}...")
    
    return run_id


def print_experiment_summary():
    """Print summary of all logged experiments."""
    client = mlflow.tracking.MlflowClient()
    
    experiment = client.get_experiment_by_name("fin_sight_ai_rag_pipeline")
    if not experiment:
        print("No experiments found yet.")
        return
    
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    
    print(f"\n📊 EXPERIMENT SUMMARY")
    print(f"=" * 50)
    print(f"Total runs: {len(runs)}")
    
    if runs:
        print(f"\nRecent runs:")
        for run in runs[-5:]:  # Last 5 runs
            run_id = run.info.run_id[:8]
            question = run.data.params.get('question', 'N/A')[:50] + "..."
            total_time = run.data.metrics.get('timing_total_sec', 0)
            cost = run.data.metrics.get('cost_usd', 0)
            
            print(f"  • {run_id}: '{question}' ({total_time}s, ${cost:.4f})")


# ============================================
# MAIN TEST
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MLFLOW TRACKER TEST")
    print("=" * 60)
    
    # Setup
    setup_mlflow()
    
    # Simulate a RAG result (for testing without full pipeline)
    test_result = {
        'success': True,
        'question': 'What documents do I need for a loan?',
        'answer': 'You need ID proof, address proof, income proof...',
        'sources': [
            {'faq_id': 1, 'question': 'Documents needed?', 'answer': 'ID, address...', 
             'category': 'Loan Application', 'score': 0.7444},
            {'faq_id': 2, 'question': 'Statement request?', 'answer': 'Download from app...', 
             'category': 'Statements', 'score': 0.4915}
        ],
        'model_used': 'gpt-3.5-turbo',
        'token_usage': {
            'prompt_tokens': 689,
            'completion_tokens': 115,
            'total_tokens': 804
        },
        'timing': {
            'embedding_generation': 2.942,
            'vector_search': 0.148,
            'context_assembly': 0.0,
            'llm_generation': 3.557,
            'total_pipeline': 6.647
        }
    }
    
    # Log it
    run_id = log_rag_query(test_result)
    
    print(f"\n✅ Test complete! Run ID: {run_id}")
    print(f"\nTo view in UI, run: mlflow ui")
    print(f"Then open: http://localhost:5000")