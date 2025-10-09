#!/bin/bash
source venv/bin/activate

# for col in student teacher
# do
#    python run_classifier.py \
#    --train_data=data/paired_annotations.csv \
#    --dev_split_size=0.1 \
#    --num_train_epochs=5 \
#    --text_cols="${col}_text" \
#    --label_col="${col}_on_task" \
#    --balance_labels \
#    --cv \
#    --predict_index_col=exchange_idx
# done



# for col in high_uptake focusing_question
# do
#     python run_classifier.py \
#     --cv \
#     --train_data=data/paired_annotations.csv \
#     --dev_split_size=0.1 \
#     --num_train_epochs=5 \
#     --text_cols=student_text,teacher_text \
#     --label_col="${col}" \
#     --balance_labels \
#     --predict_index_col=exchange_idx
# done

# echo "Running student reasoning classifier>>>>>>>>"
# python run_classifier.py \
#    --cv \
#    --train_data=data/student_reasoning.csv \
#    --dev_split_size=0.1 \
#    --num_train_epochs=5 \
#    --text_cols=text \
#    --label_col=student_reasoning \
#    --balance_labels \
#    --predict_index_col=comb_idx


echo "Running student reasoning classifier (prediction)>>>>>>>>"
python run_classifier.py \
--predict \
--predict_data=data/ncte_single_utterances_cands.csv \
--dev_split_size=0.1 \
--num_train_epochs=5 \
--text_cols=text \
--label_col=student_reasoning \
--balance_labels \
--predict_index_col=comb_idx \
--model_path=outputs/roberta/student_reasoning_best_cv_model
