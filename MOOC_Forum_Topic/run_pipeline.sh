#!/bin/bash
# Run complete pipeline for MOOC Forum Topic Analysis

echo "=========================================="
echo "MOOC Forum Topic Analysis Pipeline"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: venv not found. Please run setup first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check if data exists
if [ ! -f "dataset/stanfordMOOCForumPostsSet.xlsx" ]; then
    echo "Error: Dataset not found!"
    echo "Please download to: dataset/stanfordMOOCForumPostsSet.xlsx"
    exit 1
fi

# Run pipeline
echo ""
echo "[1/7] Data preprocessing..."
python step1_preprocess_data.py || exit 1

echo ""
echo "[2/7] Text cleaning..."
python step2_text_preprocessing.py || exit 1

echo ""
echo "[3/7] Generate embeddings..."
python step3_generate_embeddings.py || exit 1

echo ""
echo "[4/7] BERTopic training + Grid search..."
python step4_train_bertopic.py || exit 1

echo ""
echo "[4b/7] Grid search visualization..."
python step4_visualize_grid_search.py || exit 1

echo ""
echo "[6/7] Train traditional models..."
python step6_traditional_models.py "LDA,LSI" || exit 1

echo ""
echo "[7/7] Model comparison..."
python step7_paper_comparison.py || exit 1

echo ""
echo "=========================================="
echo "✓ Pipeline completed successfully!"
echo "=========================================="
