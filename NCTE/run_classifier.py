"""
Classroom Transcript Analysis - Text Classification Tool

Usage Examples:
1. Training:
   python run_classifier.py --train --train_data=data.csv --text_cols=text --label_col=label --predict_index_col=index

2. Cross-validation:
   python run_classifier.py --cv --train_data=data.csv --text_cols=text --label_col=label --predict_index_col=index --balance_labels

3. Prediction with existing model:
   python run_classifier.py --predict --predict_data=data.csv --model_path=outputs/roberta/best_model --text_cols=text --predict_index_col=index

4. Train and predict in one run:
   python run_classifier.py --train --predict --train_data=train.csv --predict_data=test.csv --text_cols=text --label_col=label --predict_index_col=index
"""

import os
import logging
import warnings
from sys import exit
from argparse import ArgumentParser

import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr, spearmanr
from simpletransformers.classification import ClassificationModel, ClassificationArgs

warnings.filterwarnings("ignore")


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def pearson_corr(preds, labels):
    """Calculate Pearson correlation coefficient."""
    return pearsonr(preds, labels)[0]

def spearman_corr(preds, labels):
    """Calculate Spearman correlation coefficient."""
    return spearmanr(preds, labels)[0]

def accuracy(preds, labels):
    """Calculate accuracy score."""
    return sum([p == l for p, l in zip(preds, labels)]) / len(labels)

def precision(preds, labels):
    """Calculate precision score."""
    return precision_score(y_true=labels, y_pred=preds)

def recall(preds, labels):
    """Calculate recall score."""
    return recall_score(y_true=labels, y_pred=preds)

def f1(preds, labels):
    """Calculate F1 score."""
    return f1_score(y_true=labels, y_pred=preds)


# =============================================================================
# DATA PREPROCESSING FUNCTIONS
# =============================================================================

def load_and_preprocess_data(data_path, text_cols, label_col, predict_index_col=None, is_prediction_data=False):
    """Load and preprocess data for training or prediction."""
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path).sample(frac=1)
    
    if not is_prediction_data:
        # Remove rows with null labels for training data
        data = data[~data[label_col].isnull()]
        print(f"Loaded {len(data)} training examples.")
    else:
        print(f"Loaded {len(data)} prediction examples.")
    
    # Prepare column mapping and select relevant columns
    if len(text_cols) == 1:
        column_mapping = {text_cols[0]: 'text'}
        required_cols = ['text']
        if not is_prediction_data:
            column_mapping[label_col] = 'labels'
            required_cols.append('labels')
        if predict_index_col:
            required_cols.insert(0, predict_index_col)
            
    elif len(text_cols) == 2:
        column_mapping = {text_cols[0]: 'text_a', text_cols[1]: 'text_b'}
        required_cols = ['text_a', 'text_b']
        if not is_prediction_data:
            column_mapping[label_col] = 'labels'
            required_cols.append('labels')
        if predict_index_col:
            required_cols.insert(0, predict_index_col)
    else:
        raise ValueError("You can have up to 2 text columns to classify!")
    
    # Apply column renaming and select required columns
    data = data.rename(columns=column_mapping)[required_cols].dropna()
    
    # Convert labels to integers for classification (only for training data)
    if not is_prediction_data:
        print(f"Label types before conversion: {data['labels'].dtype}")
        print(f"Unique labels before conversion: {sorted(data['labels'].unique())}")
        data['labels'] = data['labels'].astype(int)
        print(f"Label types after conversion: {data['labels'].dtype}")
        print(f"Unique labels after conversion: {sorted(data['labels'].unique())}")
    
    return data

def balance_training_data(train_df):
    """Balance training data by upsampling minority classes."""
    most_common = train_df["labels"].value_counts().idxmax()
    print(f"Most common label is: {most_common}")
    most_common_df = train_df[train_df["labels"] == most_common]
    concat_list = [most_common_df]
    
    for label, group in train_df[train_df["labels"] != most_common].groupby("labels"):
        concat_list.append(group.sample(replace=True, n=len(most_common_df)))
    
    balanced_df = pd.concat(concat_list).sample(frac=1)  # Shuffle after balancing
    print(f"Balanced train size: {len(balanced_df)}")
    print(balanced_df["labels"].value_counts())
    return balanced_df


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================

def create_model_args(colname, output_dir, train_df, text_cols, num_train_epochs=10, 
                     train_batch_size=8, gradient_accumulation_steps=2, max_seq_length=512,
                     cross_validate=False, num_labels=2):
    """Create and configure model arguments."""
    save_dir = os.path.join(output_dir, f"{colname}_train_size={len(train_df)}")
    
    model_args = ClassificationArgs()
    model_args.reprocess_input_data = True
    model_args.overwrite_output_dir = True
    model_args.evaluate_during_training = True
    model_args.max_seq_length = int(max_seq_length / len(text_cols))
    model_args.num_train_epochs = num_train_epochs
    model_args.evaluate_during_training_steps = max(1, int(len(train_df) / train_batch_size))
    model_args.save_eval_checkpoints = False
    model_args.save_model_every_epoch = False
    model_args.wandb_project = colname
    model_args.train_batch_size = train_batch_size
    model_args.output_dir = save_dir
    model_args.best_model_dir = os.path.join(save_dir, "best_model")
    model_args.cache_dir = os.path.join(save_dir, "cache")
    model_args.tensorboard_dir = os.path.join(save_dir, "tensorboard")
    model_args.regression = num_labels == 1
    model_args.gradient_accumulation_steps = gradient_accumulation_steps
    model_args.wandb_kwargs = {"reinit": True}
    model_args.fp16 = False
    model_args.fp16_opt_level = "O0"
    model_args.no_cache = False
    model_args.no_save = False  # Always save models (changed from cross_validate)
    model_args.save_optimizer_and_scheduler = True
    model_args.save_best_model_at_end = True
    model_args.metric_for_best_model = "f1"
    
    return model_args, save_dir

def train_model(colname, train_df, eval_df, text_cols, output_dir, model_type="roberta-base", 
                num_labels=2, num_train_epochs=10, train_batch_size=8, 
                gradient_accumulation_steps=2, max_seq_length=512, 
                cross_validate=False, balance_labels=True):
    """Train a classification model."""
    print(f"Train size: {len(train_df)}")
    print(f"Eval size: {len(eval_df)}")
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    
    # Balance labels if requested
    if balance_labels:
        train_df = balance_training_data(train_df)
    
    # Shuffle training data
    train_df = train_df.sample(frac=1)
    
    # Create model arguments
    model_args, save_dir = create_model_args(
        colname, output_dir, train_df, text_cols, num_train_epochs, 
        train_batch_size, gradient_accumulation_steps, max_seq_length,
        cross_validate, num_labels
    )
    
    # Initialize model
    model = ClassificationModel(
        model_type.split("-")[0], model_type,
        use_cuda=torch.cuda.is_available(),
        num_labels=num_labels,
        args=model_args
    )
    
    # Train model
    model.train_model(
        train_df,
        eval_df=eval_df,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        args={
            "use_multiprocessing": False,
            "process_count": 1,
            "use_multiprocessing_for_evaluation": False
        }
    )
    
    return model, save_dir

# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def load_model(model_path, model_type="roberta-base"):
    """Load a trained model from the specified path."""
    print(f"Loading model from: {model_path}")
    model = ClassificationModel(model_type.split("-")[0], model_path)
    return model

def prepare_prediction_data(predict_df, text_cols):
    """Prepare data for prediction based on number of text columns."""
    if len(text_cols) == 1:
        predict_list = predict_df["text"].tolist()
    elif len(text_cols) == 2:
        predict_list = predict_df[["text_a", "text_b"]].values.tolist()
    else:
        raise ValueError("You can have up to 2 text columns to classify!")
    return predict_list

def save_predictions(predictions, index_list, output_path, filename, index_colname="index"):
    """Save predictions to a file."""
    output_file = os.path.join(output_path, f"{filename}_preds.txt")
    with open(output_file, 'w') as f:
        f.write(f"{index_colname}\tpred\n")
        for index, pred in zip(index_list, predictions):
            f.write(f"{index}\t{pred}\n")
    print(f"Predictions saved to: {output_file}")

def predict_with_model(model, predict_df, text_cols, output_path, filename, 
                      index_colname="index", model_path=None):
    """Make predictions using a trained model."""
    # Load model if not provided
    if model is None:
        if model_path is None:
            raise ValueError("Either model or model_path must be provided")
        model = load_model(model_path)
    
    # Prepare prediction data
    predict_list = prepare_prediction_data(predict_df, text_cols)
    index_list = predict_df[index_colname].tolist()
    
    # Make predictions
    print(f"Making predictions on {len(predict_list)} examples...")
    predictions, outputs = model.predict(predict_list)
    
    # Save predictions
    save_predictions(predictions, index_list, output_path, filename, index_colname)
    
    return predictions

# =============================================================================
# CROSS-VALIDATION FUNCTIONS
# =============================================================================

def run_cross_validation(train_data, text_cols, label_col, predict_index_col, 
                        output_dir, model_type, num_train_epochs, balance_labels,
                        n_splits=5):
    """Run k-fold cross-validation and save the best performing model."""
    print(f"Running {n_splits}-fold cross-validation...")
    
    kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    fold_results = []
    best_f1 = 0
    best_model_path = None
    best_fold = -1
    
    for fold, (train_index, val_index) in enumerate(kf.split(train_data)):
        print(f"\n=== Fold {fold} ===")
        output_dir_k = os.path.join(output_dir, f"{label_col}_k{fold}")
        
        # Split data
        train_df = train_data.iloc[train_index]
        eval_df = train_data.iloc[val_index]
        
        print(f"Train fold size: {len(train_df)}")
        print(f"Eval fold size: {len(eval_df)}")
        
        # Train model for this fold
        model, save_dir = train_model(
            label_col, train_df, eval_df, text_cols, 
            output_dir=output_dir_k, model_type=model_type, 
            num_train_epochs=num_train_epochs, 
            balance_labels=balance_labels, cross_validate=True
        )
        
        # Make predictions on validation set
        predict_list = prepare_prediction_data(eval_df, text_cols)
        predictions, outputs = model.predict(predict_list)
        
        # Calculate metrics
        true_labels = eval_df["labels"].tolist()
        fold_f1 = f1(predictions, true_labels)
        fold_accuracy = accuracy(predictions, true_labels)
        fold_precision = precision(predictions, true_labels)
        fold_recall = recall(predictions, true_labels)
        
        fold_results.append({
            'fold': fold,
            'f1': fold_f1,
            'accuracy': fold_accuracy,
            'precision': fold_precision,
            'recall': fold_recall,
            'model_path': os.path.join(save_dir, "best_model")
        })
        
        print(f"Fold {fold} - F1: {fold_f1:.4f}, Accuracy: {fold_accuracy:.4f}")
        
        # Track best model
        if fold_f1 > best_f1:
            best_f1 = fold_f1
            best_model_path = os.path.join(save_dir, "best_model")
            best_fold = fold
        
        # Save predictions for this fold
        index_list = eval_df[predict_index_col].tolist()
        filename = f"{label_col}_{train_data.name if hasattr(train_data, 'name') else 'data'}_split_{fold}"
        save_predictions(predictions, index_list, output_dir_k, filename, predict_index_col)
    
    # Print overall results
    print(f"\n=== Cross-Validation Results ===")
    avg_f1 = sum(result['f1'] for result in fold_results) / len(fold_results)
    avg_accuracy = sum(result['accuracy'] for result in fold_results) / len(fold_results)
    print(f"Average F1: {avg_f1:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Best fold: {best_fold} (F1: {best_f1:.4f})")
    print(f"Best model saved at: {best_model_path}")
    
    # Copy best model to main output directory
    best_model_final_path = os.path.join(output_dir, f"{label_col}_best_cv_model")
    if best_model_path and os.path.exists(best_model_path):
        import shutil
        if os.path.exists(best_model_final_path):
            shutil.rmtree(best_model_final_path)
        shutil.copytree(best_model_path, best_model_final_path)
        print(f"Best model copied to: {best_model_final_path}")
    
    return fold_results, best_model_final_path


# =============================================================================
# ARGUMENT PARSING AND MAIN FUNCTION
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Classroom Transcript Analysis - Text Classification Tool")
    
    # Mode selection
    parser.add_argument("--train", action='store_true', help="Train model")
    parser.add_argument("--cv", action='store_true', help="Run cross validation")
    parser.add_argument("--predict", action='store_true', help="Make predictions")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, default="data/paired_annotations.csv",
                        help="Training data CSV file")
    parser.add_argument("--predict_data", type=str, default="data/paired_utterances.csv",
                        help="Prediction data CSV file")
    parser.add_argument("--text_cols", type=str, required=True,
                        help="Text columns, comma separated")
    parser.add_argument("--label_col", type=str, help="Label column name (required for training/cv)")
    parser.add_argument("--predict_index_col", type=str, required=True,
                        help="Index column for mapping predictions to input")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="roberta-base", help="Model type")
    parser.add_argument("--model_path", type=str, help="Path to pre-trained model (for prediction only)")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, default="outputs/roberta", help="Output directory")
    
    # Training arguments
    parser.add_argument("--dev_split_size", type=float, default=0.2,
                        help="Validation split size (0 means use training data as validation)")
    parser.add_argument("--balance_labels", action='store_true',
                        help="Balance label distributions via upsampling")
    
    return parser.parse_args()

def main():
    """Main function to coordinate training, cross-validation, and prediction."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    
    # Validate arguments
    if (args.train or args.cv) and not args.label_col:
        raise ValueError("--label_col is required for training or cross-validation")
    
    if args.predict and not (args.model_path or args.train or args.cv):
        raise ValueError("For prediction, either --model_path must be provided or training must be run first")
    
    # Parse text columns
    text_cols = args.text_cols.split(",")
    print(f"Using text columns: {text_cols}")
    
    model = None
    best_model_path = None
    
    # Load training data if needed
    if args.cv or args.train:
        train_data = load_and_preprocess_data(
            args.train_data, text_cols, args.label_col, 
            args.predict_index_col, is_prediction_data=False
        )
    
    # Cross-validation mode
    if args.cv:
        fold_results, best_model_path = run_cross_validation(
            train_data, text_cols, args.label_col, args.predict_index_col,
            args.output_dir, args.model_type, args.num_train_epochs, 
            args.balance_labels
        )
        
        # Load best model for potential prediction
        if best_model_path:
            model = load_model(best_model_path, args.model_type)
    
    # Training mode (single train/test split)
    elif args.train:
        if args.dev_split_size > 0:
            train_df, eval_df = train_test_split(train_data, test_size=args.dev_split_size, 
                                               random_state=42)
        else:
            train_df = eval_df = train_data
        
        model, save_dir = train_model(
            args.label_col, train_df, eval_df, text_cols, args.output_dir,
            args.model_type, num_train_epochs=args.num_train_epochs,
            balance_labels=args.balance_labels
        )
        best_model_path = os.path.join(save_dir, "best_model")
    
    # Prediction mode
    if args.predict:
        # Determine model path
        if args.model_path:
            model_path_for_prediction = args.model_path
        elif best_model_path:
            model_path_for_prediction = best_model_path
        else:
            raise ValueError("No model available for prediction")
        
        # Load prediction data
        predict_df = load_and_preprocess_data(
            args.predict_data, text_cols, None, args.predict_index_col, 
            is_prediction_data=True
        )
        
        # Generate filename
        data_name = os.path.splitext(os.path.basename(args.predict_data))[0]
        filename = f"{args.label_col or 'predictions'}_{data_name}"
        
        # Make predictions
        predictions = predict_with_model(
            model, predict_df, text_cols, 
            model_path_for_prediction, filename, 
            args.predict_index_col, model_path_for_prediction
        )
        
        print(f"Prediction completed. Results saved in {model_path_for_prediction}")

if __name__ == "__main__":
    main()