"""
Step 5: 模型训练和评估

严格遵循instruction.txt:
- 使用相同的超参数和pipeline作为baseline
- 评估指标: PR-AUC, F1, Precision, Recall
- 模型: Logistic Regression, Random Forest, Naive Bayes, XGBoost
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    precision_recall_curve, 
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: XGBoost未安装，将跳过XGBoost模型")


class ModelTrainer:
    """模型训练器 - 训练和评估多个模型"""
    
    def __init__(self, random_seed=42):
        """
        初始化训练器
        
        Args:
            random_seed: 随机种子
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        # 保存特征信息，用于特征对齐
        self._feature_columns = None  # 训练时的特征列名
        self._categorical_encoders = {}  # 类别特征的编码映射 {col: {value: code}}
        self._list_col_expansions = {}  # 列表特征展开后的列数 {col: num_cols}
    
    def _prepare_features(self, df, fit_mode=False):
        """
        准备特征（编码、处理列表特征等）
        
        Args:
            df: 输入DataFrame
            fit_mode: 如果为True，保存特征信息用于后续对齐；如果为False，使用保存的特征信息对齐
            
        Returns:
            X: 特征矩阵
            y: 标签向量
        """
        df = df.copy()
        
        # 分离标签
        if 'submitted' not in df.columns:
            raise ValueError("DataFrame必须包含'submitted'列")
        
        y = df['submitted'].values
        X = df.drop(columns=['submitted'])
        
        # 移除ID列
        for col in ['id_student', 'student_type']:
            if col in X.columns:
                X = X.drop(columns=[col])
        
        # 处理列表特征（weekly_vle_clicks等）
        list_cols = ['weekly_vle_clicks', 'active_days_per_week', 'recency_gaps']
        for col in list_cols:
            if col in X.columns:
                # 如果是字符串，解析为列表
                if X[col].dtype == 'object':
                    import ast
                    X[col] = X[col].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                    )
                
                # 转换为多列（week_0, week_1, ...）
                expanded = pd.DataFrame(
                    X[col].tolist(),
                    index=X.index
                )
                
                if fit_mode:
                    # 训练模式：保存展开后的列数
                    self._list_col_expansions[col] = len(expanded.columns)
                    expanded.columns = [f'{col}_{i}' for i in range(len(expanded.columns))]
                else:
                    # 测试模式：使用训练时保存的列数
                    if col not in self._list_col_expansions:
                        raise ValueError(f"列表特征 {col} 未在训练时见过，无法对齐")
                    expected_cols = self._list_col_expansions[col]
                    # 如果列数不足，用0填充
                    if len(expanded.columns) < expected_cols:
                        for i in range(len(expanded.columns), expected_cols):
                            expanded[f'{col}_{i}'] = 0
                    # 如果列数过多，截断
                    elif len(expanded.columns) > expected_cols:
                        expanded = expanded.iloc[:, :expected_cols]
                    expanded.columns = [f'{col}_{i}' for i in range(expected_cols)]
                
                X = pd.concat([X, expanded], axis=1)
                X = X.drop(columns=[col])
        
        # 编码分类特征
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                if fit_mode:
                    # 训练模式：保存编码映射
                    cat = pd.Categorical(X[col])
                    self._categorical_encoders[col] = {
                        val: code for val, code in zip(cat.categories, range(len(cat.categories)))
                    }
                    X[col] = cat.codes
                else:
                    # 测试模式：使用训练时的编码映射
                    if col in self._categorical_encoders:
                        # 使用保存的编码映射，未知类别映射为-1（会被后续fillna处理为0）
                        X[col] = X[col].map(self._categorical_encoders[col]).fillna(-1).astype(int)
                    else:
                        # 如果训练时没有这个列，编码为0（会在后续特征对齐时被移除）
                        X[col] = 0
        
        # 填充缺失值
        X = X.fillna(0)
        
        # 转换为数值类型（确保所有列都是数值类型）
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                # 如果还有未编码的类别特征，强制编码为0
                X[col] = 0
        X = X.astype(float)
        
        # 特征对齐：确保测试数据的特征列与训练数据一致
        if fit_mode:
            # 训练模式：保存特征列名
            self._feature_columns = X.columns.tolist()
        else:
            # 测试模式：对齐特征列
            if self._feature_columns is None:
                raise ValueError("未找到训练时的特征列，请先使用fit_mode=True训练")
            
            # 添加缺失的特征列（填充为0）
            missing_cols = set(self._feature_columns) - set(X.columns)
            for col in missing_cols:
                X[col] = 0
            
            # 移除多余的特征列
            extra_cols = set(X.columns) - set(self._feature_columns)
            if extra_cols:
                X = X.drop(columns=list(extra_cols))
            
            # 确保列顺序一致
            X = X[self._feature_columns]
        
        return X, y
    
    def _create_models(self):
        """
        创建模型列表（与baseline相同的超参数）
        
        Returns:
            List: 模型列表 [(model, name), ...]
        """
        models = []
        
        # Logistic Regression (与baseline一致: C=0.01)
        models.append((
            LogisticRegression(C=0.01, max_iter=1000, random_state=self.random_seed),
            'LR'
        ))
        
        # Random Forest (默认参数，与baseline一致)
        models.append((
            RandomForestClassifier(n_estimators=100, random_state=self.random_seed, n_jobs=-1),
            'RF'
        ))
        
        # Naive Bayes
        models.append((
            GaussianNB(),
            'NB'
        ))
        
        # XGBoost (如果可用)
        if XGBOOST_AVAILABLE:
            models.append((
                xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=self.random_seed,
                    eval_metric='logloss'
                ),
                'XGB'
            ))
        
        return models
    
    def _evaluate_model(self, model, X_test, y_test):
        """
        评估模型
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            Dict: 评估指标
        """
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 计算指标
        pr_auc = average_precision_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        # ROC-AUC (如果可能)
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = np.nan
        
        return {
            'pr_auc': pr_auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        }
    
    def train_and_evaluate(self, train_df, test_df, model_name='LR'):
        """
        训练和评估单个模型
        
        Args:
            train_df: 训练DataFrame
            test_df: 测试DataFrame
            model_name: 模型名称
            
        Returns:
            Dict: 评估结果
        """
        # 准备数据（训练时保存特征信息）
        X_train, y_train = self._prepare_features(train_df, fit_mode=True)
        # 测试时使用保存的特征信息对齐
        X_test, y_test = self._prepare_features(test_df, fit_mode=False)
        
        # 创建模型
        models = self._create_models()
        model_dict = {name: model for model, name in models}
        
        if model_name not in model_dict:
            raise ValueError(f"未知模型: {model_name}")
        
        model = model_dict[model_name]
        
        # 训练
        print(f"  训练 {model_name}...", end=' ')
        model.fit(X_train, y_train)
        
        # 评估
        metrics = self._evaluate_model(model, X_test, y_test)
        print(f"PR-AUC={metrics['pr_auc']:.4f}")
        
        return metrics
    
    def run_full_experiment(self, datasets, test_df, models=None):
        """
        运行完整实验（所有数据集 × 所有模型）
        
        Args:
            datasets: 数据集字典 {condition: {'df': DataFrame, ...}, ...}
            test_df: 测试集DataFrame
            models: 模型列表（默认: ['LR', 'RF', 'NB', 'XGB']）
            
        Returns:
            List: 结果列表
        """
        if models is None:
            models = ['LR', 'RF', 'NB']
            if XGBOOST_AVAILABLE:
                models.append('XGB')
        
        print(f"\n{'='*80}")
        print("训练和评估模型")
        print(f"{'='*80}")
        print(f"模型: {', '.join(models)}")
        print(f"数据集: {', '.join(datasets.keys())}")
        print(f"{'='*80}\n")
        
        results = []
        test_size = len(test_df)
        
        # 为每个数据集和模型组合训练和评估
        for condition, data_info in datasets.items():
            train_df = data_info['df']
            train_size = len(train_df)
            
            print(f"\n条件: {condition} (训练集大小: {train_size})")
            print("-" * 60)
            
            # 重置特征信息（每个数据集独立）
            self._feature_columns = None
            self._categorical_encoders = {}
            self._list_col_expansions = {}
            
            # 准备训练数据（保存特征信息）
            X_train, y_train = self._prepare_features(train_df, fit_mode=True)
            
            # 准备测试数据（使用训练时的特征信息对齐）
            X_test, y_test = self._prepare_features(test_df, fit_mode=False)
            
            for model_name in models:
                try:
                    
                    # 创建模型
                    model_list = self._create_models()
                    model_dict = {name: model for model, name in model_list}
                    
                    if model_name not in model_dict:
                        print(f"  跳过 {model_name} (不可用)")
                        continue
                    
                    model = model_dict[model_name]
                    
                    # 训练
                    print(f"  {model_name}...", end=' ')
                    model.fit(X_train, y_train)
                    
                    # 评估
                    metrics = self._evaluate_model(model, X_test, y_test)
                    print(f"PR-AUC={metrics['pr_auc']:.4f}, F1={metrics['f1']:.4f}")
                    
                    # 保存结果
                    result = {
                        'condition': condition,
                        'model': model_name,
                        'seed': self.random_seed,
                        'pr_auc': metrics['pr_auc'],
                        'f1': metrics['f1'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'train_size': train_size,
                        'test_size': test_size
                    }
                    
                    if not np.isnan(metrics['roc_auc']):
                        result['roc_auc'] = metrics['roc_auc']
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"  ✗ {model_name} 失败: {e}")
                    continue
        
        print(f"\n{'='*80}")
        print(f"✓ 完成！共 {len(results)} 个结果")
        print(f"{'='*80}\n")
        
        return results


if __name__ == "__main__":
    # 测试代码
    print("="*80)
    print("测试模型训练器")
    print("="*80)
    
    # 创建mock数据
    np.random.seed(42)
    n_train = 100
    n_test = 50
    n_features = 20
    
    train_df = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    train_df['submitted'] = np.random.randint(0, 2, n_train)
    
    test_df = pd.DataFrame(
        np.random.randn(n_test, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    test_df['submitted'] = np.random.randint(0, 2, n_test)
    
    # 创建训练器
    trainer = ModelTrainer(random_seed=42)
    
    # 创建数据集
    datasets = {
        'baseline': {'df': train_df, 'path': None, 'size': n_train}
    }
    
    # 运行实验
    results = trainer.run_full_experiment(datasets, test_df, models=['LR', 'RF', 'NB'])
    
    # 打印结果
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    print("\n测试完成！")


