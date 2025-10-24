"""
Run all experiment steps sequentially
"""
import sys
import time

def run_step(step_name, module_name):
    """Run a single step and handle errors"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {step_name}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # Import and run the step
        module = __import__(module_name)
        module.main()
        
        elapsed = time.time() - start_time
        print(f"\n✓ {step_name} completed in {elapsed:.1f}s")
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {step_name} failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        return False

def main():
    print("\n" + "="*80)
    print("MOOC FORUM TOPIC ANALYSIS - FULL PIPELINE")
    print("="*80)
    
    steps = [
        ("Step 1: Preprocess Data", "step1_preprocess_data"),
        ("Step 2: Text Preprocessing", "step2_text_preprocessing"),
        ("Step 3: Generate Embeddings", "step3_generate_embeddings"),
        ("Step 4: Train BERTopic", "step4_train_bertopic"),
        ("Step 5: Evaluate Models", "step5_evaluate"),
        ("Step 6: Visualize Results", "step6_visualize"),
    ]
    
    total_start = time.time()
    
    for step_name, module_name in steps:
        success = run_step(step_name, module_name)
        if not success:
            print(f"\n✗ Pipeline stopped at: {step_name}")
            print("Fix the error and run the failed step individually:")
            print(f"  python {module_name}.py")
            sys.exit(1)
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*80)
    print("✓ ALL STEPS COMPLETED SUCCESSFULLY")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print("="*80)
    
    print("\nGenerated files:")
    print("  data/groups_raw.pkl")
    print("  data/groups_preprocessed.pkl")
    print("  data/embeddings.pkl")
    print("  models/bertopic_*/")
    print("  results_summary.csv")
    print("  results_comparison.png")
    print("  topic_visualization_all.html")

if __name__ == '__main__':
    main()

