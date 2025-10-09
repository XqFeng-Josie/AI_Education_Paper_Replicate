import os
import pandas as pd

'''
Get the best/avg scores from the k-fold cross-validation results.
Usage:
python analyze_kfold_best_models.py --kfold_dir outputs/roberta/ --prefix student_on_task_k 
python analyze_kfold_best_models.py --kfold_dir outputs/roberta/ --all
'''

def analyze_kfold_best_models(kfold_dir, prefix=""):
    best_scores = []
    for fold in os.listdir(kfold_dir):
        if fold.startswith(prefix):
            sub_dir = [r for r in os.listdir(os.path.join(kfold_dir, fold)) if os.path.isdir(os.path.join(kfold_dir, fold, r))][0]
            scores_file = os.path.join(kfold_dir, fold, sub_dir, "training_progress_scores.csv")
            scores = pd.read_csv(scores_file)
            # float -> round to 3 decimal places
            scores = scores.round(3)
            # sort by f1
            scores = scores.sort_values(by="f1", ascending=False)
            best_score = scores.iloc[0]
            print(fold,  best_score['accuracy'], best_score['precision'], best_score['recall'],best_score['f1'])
            best_scores.append(best_score)
    if len(best_scores) == 0:
        return None
    best_scores = pd.DataFrame(best_scores)
    return best_scores


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--kfold_dir", type=str, default="outputs/roberta/")
    parser.add_argument("--prefix", type=str, default="", choices=["student_on_task_k", "teacher_on_task_k", "high_uptake_k", "focusing_question_k","student_reasoning_k"])
    parser.add_argument("--all", action='store_true', help="Print all avg scores")
    args = parser.parse_args()
    result_csv = "results.csv"
    result_df = []
    if args.all:
        prefixes = ["student_on_task_k", "teacher_on_task_k", "high_uptake_k", "focusing_question_k","student_reasoning_k"]
    else:
        prefixes = [args.prefix]
    for prefix in prefixes:
        print(f"prefix: {prefix}")
        best_scores = analyze_kfold_best_models(args.kfold_dir, prefix)
        if best_scores is not None:
            avg_scores = best_scores.mean().round(3)
            print(f"avg_scores: accuracy: {avg_scores['accuracy']}, precision: {avg_scores['precision']}, recall: {avg_scores['recall']}, f1: {avg_scores['f1']}")
            final_best_score = best_scores.sort_values(by="f1", ascending=False).iloc[0].round(3)
            print(f"best scores: accuracy: {final_best_score['accuracy']}, precision: {final_best_score['precision']}, recall: {final_best_score['recall']}, f1: {final_best_score['f1']}")
            result_df.append({
                'prefix': prefix,
                'accuracy': avg_scores['accuracy'],
                'precision': avg_scores['precision'],
                'recall': avg_scores['recall'],
                'f1': avg_scores['f1']
            })
        else:
            print(f"No best scores found for prefix: {prefix}")
    if len(result_df) > 0:
        result_df = pd.DataFrame(result_df)
        result_df.to_csv(result_csv, index=False, sep="\t")
        print(f"Results saved to {result_csv}")