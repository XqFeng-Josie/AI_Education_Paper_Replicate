"""
使用合成数据增强的训练实验
将合成数据添加到训练集，测试集始终使用原始数据
"""
import pandas as pd
import numpy as np
import json
import sys
from datetime import datetime
from pathlib import Path

# 添加baseline模块到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'baseline'))

from data_preprocessing import DataPreprocessor
from models import (
    NaiveBaselineClassifier, NaiveBaselineRegressor,
    get_decision_tree_classifier, get_decision_tree_regressor,
    get_random_forest_classifier, get_random_forest_regressor
)
from evaluation import cross_validate


def cross_validate_with_augmentation(model, X_original, y_original, X_synthetic, y_synthetic,
                                     setup='A', task='classification', 
                                     n_runs=20, n_folds=10, random_state=42, g2_data=None,
                                     g2_synthetic=None):
    """
    执行带数据增强的交叉验证
    合成数据只添加到训练集，测试集始终使用原始数据
    
    Args:
        model: 模型对象
        X_original: 原始特征矩阵
        y_original: 原始目标变量
        X_synthetic: 合成特征矩阵
        y_synthetic: 合成目标变量
        setup: 'A' 或 'C'
        task: 'classification' 或 'regression'
        n_runs: 重复次数
        n_folds: 折数
        random_state: 随机种子
        g2_data: 原始G2数据
        g2_synthetic: 合成G2数据
        
    Returns:
        results: 所有折的结果列表
        mean_score: 平均分数
        std_score: 标准差
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, mean_squared_error
    from tqdm import tqdm
    
    results = []
    
    print(f"\n开始增强数据交叉验证: {n_runs}次 × {n_folds}折")
    print(f"任务: {task}, 设置: Setup {setup}")
    print(f"原始数据: {len(X_original)} 条, 合成数据: {len(X_synthetic)} 条")
    
    for run in tqdm(range(n_runs), desc="运行次数"):
        # 每次运行使用不同的随机种子
        run_seed = random_state + run
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=run_seed)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_original)):
            # 划分原始数据的训练集和测试集
            if hasattr(X_original, 'iloc'):
                X_train_orig = X_original.iloc[train_idx]
                X_test = X_original.iloc[test_idx]
            else:
                X_train_orig = X_original[train_idx]
                X_test = X_original[test_idx]
            
            if hasattr(y_original, 'iloc'):
                y_train_orig = y_original.iloc[train_idx]
                y_test = y_original.iloc[test_idx]
            else:
                y_train_orig = y_original[train_idx]
                y_test = y_original[test_idx]
            
            # 将合成数据添加到训练集
            if hasattr(X_synthetic, 'iloc'):
                X_train = pd.concat([X_train_orig, X_synthetic], ignore_index=True)
            else:
                X_train = np.vstack([X_train_orig, X_synthetic])
            
            if hasattr(y_synthetic, 'iloc'):
                y_train = pd.concat([y_train_orig, y_synthetic], ignore_index=True)
            else:
                y_train = np.concatenate([y_train_orig, y_synthetic])
            
            # 准备G2数据（如果需要）
            g2_train = None
            g2_test = None
            if g2_data is not None:
                if hasattr(g2_data, 'iloc'):
                    g2_train_orig = g2_data.iloc[train_idx]
                    g2_test = g2_data.iloc[test_idx]
                else:
                    g2_train_orig = g2_data[train_idx]
                    g2_test = g2_data[test_idx]
                
                # 添加合成G2数据到训练集
                if g2_synthetic is not None:
                    if hasattr(g2_synthetic, 'iloc'):
                        g2_train = pd.concat([g2_train_orig, g2_synthetic], ignore_index=True)
                    else:
                        g2_train = np.concatenate([g2_train_orig, g2_synthetic])
                else:
                    g2_train = g2_train_orig
            
            # 训练模型（仅支持 g2_data 的模型才传入该参数）
            if hasattr(model, 'fit'):
                if g2_train is not None:
                    try:
                        model.fit(X_train, y_train, g2_data=g2_train)
                    except TypeError:
                        model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
            
            # 预测（只在原始测试集上）
            if g2_test is not None:
                try:
                    y_pred = model.predict(X_test, g2_data=g2_test)
                except TypeError:
                    y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
            
            # 评估
            if task == 'classification':
                score = accuracy_score(y_test, y_pred)
            else:  # regression
                score = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE
            
            results.append(score)
    
    results = np.array(results)
    mean_score = np.mean(results)
    std_score = np.std(results)
    
    return results, mean_score, std_score


def evaluate_model_with_augmentation(model, X_original, y_original, X_synthetic, y_synthetic,
                                     setup='A', task='classification',
                                     n_runs=20, n_folds=10, random_state=42, g2_data=None,
                                     g2_synthetic=None, model_name='Model'):
    """
    评估带数据增强的模型
    """
    results, mean_score, std_score = cross_validate_with_augmentation(
        model, X_original, y_original, X_synthetic, y_synthetic,
        setup=setup, task=task,
        n_runs=n_runs, n_folds=n_folds, random_state=random_state,
        g2_data=g2_data, g2_synthetic=g2_synthetic
    )
    
    result_dict = {
        'model_name': model_name,
        'setup': setup,
        'task': task,
        'mean_score': mean_score,
        'std_score': std_score,
        'all_results': results.tolist(),
        'n_runs': n_runs,
        'n_folds': n_folds,
        'augmented': True
    }
    
    # 打印结果
    if task == 'classification':
        print(f"\n{model_name} (Setup {setup}, 数据增强) - 分类准确率:")
    else:
        print(f"\n{model_name} (Setup {setup}, 数据增强) - 回归RMSE:")
    
    print(f"  均值: {mean_score:.4f} ± {std_score:.4f}")
    
    return result_dict


def run_augmentation_experiments(synthetic_data_path='../data/student-por-synthetic.csv',
                                  n_synthetic=None):
    """
    运行数据增强实验
    
    Args:
        synthetic_data_path: 合成数据文件路径
        n_synthetic: 使用的合成数据数量（None表示使用全部）
    """
    
    print("=" * 80)
    print("数据增强实验")
    print("=" * 80)
    
    # 检查合成数据文件
    if not Path(synthetic_data_path).exists():
        print(f"错误: 合成数据文件不存在: {synthetic_data_path}")
        print("请先运行 generate_synthetic_data.py 生成合成数据")
        return
    
    # 加载原始数据
    print("\n加载原始数据...")
    preprocessor_orig = DataPreprocessor('../data/student-por.csv')
    preprocessor_orig.load_data()
    
    # 加载合成数据
    print(f"\n加载合成数据: {synthetic_data_path}")
    synthetic_data = pd.read_csv(synthetic_data_path, sep=';')
    if n_synthetic:
        synthetic_data = synthetic_data.head(n_synthetic)
    print(f"合成数据形状: {synthetic_data.shape}")
    
    # 存储所有结果
    all_results = []
    
    # 对每个设置运行实验
    for setup in ['A', 'C']:
        print(f"\n{'=' * 80}")
        print(f"Setup {setup} 数据增强实验")
        print(f"{'=' * 80}")
        
        # 准备原始数据
        X_orig, y_binary_orig, y_regression_orig, feature_names = preprocessor_orig.get_setup_data(setup)
        g2_orig = None
        if setup == 'A':
            g2_orig = preprocessor_orig.data['G2']
        
        # 准备合成数据
        preprocessor_synth = DataPreprocessor()
        preprocessor_synth.data = synthetic_data
        X_synth, y_binary_synth, y_regression_synth, _ = preprocessor_synth.get_setup_data(setup)
        g2_synth = None
        if setup == 'A':
            g2_synth = synthetic_data['G2']
        
        print(f"\n原始数据: {len(X_orig)} 条")
        print(f"合成数据: {len(X_synth)} 条")
        print(f"增强后训练集: {len(X_orig)} + {len(X_synth)} = {len(X_orig) + len(X_synth)} 条")
        
        # 1. 决策树模型（分类）
        print(f"\n{'-' * 80}")
        print("1. 决策树模型 (Decision Tree) - 分类")
        print(f"{'-' * 80}")
        dt_clf = get_decision_tree_classifier()
        result_dt_clf = evaluate_model_with_augmentation(
            dt_clf, X_orig, y_binary_orig, X_synth, y_binary_synth,
            setup=setup, task='classification',
            model_name='DT (分类, 增强)'
        )
        all_results.append(result_dt_clf)
        
        # 2. 决策树模型（回归）
        print(f"\n{'-' * 80}")
        print("2. 决策树模型 (Decision Tree) - 回归")
        print(f"{'-' * 80}")
        dt_reg = get_decision_tree_regressor()
        result_dt_reg = evaluate_model_with_augmentation(
            dt_reg, X_orig, y_regression_orig, X_synth, y_regression_synth,
            setup=setup, task='regression',
            model_name='DT (回归, 增强)'
        )
        all_results.append(result_dt_reg)
        
        # 3. 随机森林模型（分类）
        print(f"\n{'-' * 80}")
        print("3. 随机森林模型 (Random Forest) - 分类")
        print(f"{'-' * 80}")
        rf_clf = get_random_forest_classifier()
        result_rf_clf = evaluate_model_with_augmentation(
            rf_clf, X_orig, y_binary_orig, X_synth, y_binary_synth,
            setup=setup, task='classification',
            g2_data=g2_orig, g2_synthetic=g2_synth,
            model_name='RF (分类, 增强)'
        )
        all_results.append(result_rf_clf)
        
        # 4. 随机森林模型（回归）
        print(f"\n{'-' * 80}")
        print("4. 随机森林模型 (Random Forest) - 回归")
        print(f"{'-' * 80}")
        rf_reg = get_random_forest_regressor()
        result_rf_reg = evaluate_model_with_augmentation(
            rf_reg, X_orig, y_regression_orig, X_synth, y_regression_synth,
            setup=setup, task='regression',
            g2_data=g2_orig, g2_synthetic=g2_synth,
            model_name='RF (回归, 增强)'
        )
        all_results.append(result_rf_reg)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"../results/results_augmentation_{timestamp}.json"
    summary_file = f"../results/results_augmentation_summary_{timestamp}.csv"
    
    print(f"\n{'=' * 80}")
    print("保存结果")
    print(f"{'=' * 80}")
    
    # 保存详细结果
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"详细结果已保存: {results_file}")
    
    # 保存汇总表格
    summary_data = []
    for result in all_results:
        summary_data.append({
            '模型': result['model_name'],
            '设置': f"Setup {result['setup']}",
            '任务': result['task'],
            '均值': f"{result['mean_score']:.4f}",
            '标准差': f"{result['std_score']:.4f}",
            '数据增强': '是'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"汇总表格已保存: {summary_file}")
    
    # 打印汇总
    print(f"\n{'=' * 80}")
    print("实验结果汇总")
    print(f"{'=' * 80}")
    print(summary_df.to_string(index=False))
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='运行数据增强实验')
    parser.add_argument(
        '--synthetic_data',
        type=str,
        default='../data/student-por-synthetic.csv',
        help='合成数据文件路径'
    )
    parser.add_argument(
        '--n_synthetic',
        type=int,
        default=None,
        help='使用的合成数据数量（默认: 全部）'
    )
    
    args = parser.parse_args()
    
    run_augmentation_experiments(
        synthetic_data_path=args.synthetic_data,
        n_synthetic=args.n_synthetic
    )

