"""
Módulo para salvar resultados do projeto Adversarial Debiasing
Autor: Trabalho Final - IA
Data: Janeiro/2025
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class ResultsSaver:
    """Classe para organizar e salvar todos os resultados do projeto"""
    
    def __init__(self, results_base_path='../results'):
        self.base_path = results_base_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Criar estrutura de pastas
        self.figures_path = os.path.join(self.base_path, 'figures')
        self.metrics_path = os.path.join(self.base_path, 'metrics')
        self.models_path = os.path.join(self.base_path, 'models')
        
        os.makedirs(self.figures_path, exist_ok=True)
        os.makedirs(self.metrics_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
    
    def save_all(self, data_dict):
        """Salva todos os resultados"""
        print("\n💾 Salvando resultados...")
        
        self.save_metrics(data_dict)
        self.save_figures(data_dict)
        self.save_models(data_dict)
        self.create_readme(data_dict)
        
        print(f"\n✅ Todos os resultados salvos!")
        print(f"   - Figuras: {self.figures_path}")
        print(f"   - Métricas: {self.metrics_path}")
        print(f"   - Modelos: {self.models_path}")
    
    def save_metrics(self, data):
        """Salva métricas em JSON e CSV"""
        # JSON completo
        metrics_summary = {
            'timestamp': self.timestamp,
            'dataset': {
                'name': 'IBM HR Analytics',
                'total_samples': int(data['total_samples']),
                'features': int(data['num_features']),
                'train_size': int(data['train_size']),
                'test_size': int(data['test_size'])
            },
            'baseline': {
                'name': data['baseline_name'],
                'accuracy': float(data['accuracy_baseline']),
                'f1_score': float(data['f1_baseline']),
                'auc_roc': float(data['auc_baseline'])
            },
            'adversarial': {
                'accuracy': float(data['accuracy_adversarial']),
                'f1_score': float(data['f1_adversarial'])
            },
            'fairness_baseline': {
                'demographic_parity': float(data['fairness_baseline']['demographic_parity_diff']),
                'disparate_impact': float(data['fairness_baseline']['disparate_impact']),
                'equal_opportunity': float(data['fairness_baseline']['equal_opportunity_diff'])
            },
            'fairness_adversarial': {
                'demographic_parity': float(data['fairness_adversarial']['demographic_parity_diff']),
                'disparate_impact': float(data['fairness_adversarial']['disparate_impact']),
                'equal_opportunity': float(data['fairness_adversarial']['equal_opportunity_diff'])
            }
        }
        
        json_path = os.path.join(self.metrics_path, f'metrics_{self.timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(metrics_summary, f, indent=4)
        
        # CSVs
        data['comparison_perf'].to_csv(
            os.path.join(self.metrics_path, f'performance_{self.timestamp}.csv'), 
            index=False
        )
        data['comparison_fair'].to_csv(
            os.path.join(self.metrics_path, f'fairness_{self.timestamp}.csv'), 
            index=False
        )
        data['feature_importance_df'].to_csv(
            os.path.join(self.metrics_path, f'shap_{self.timestamp}.csv'), 
            index=False
        )
        
        print(f"   ✓ Métricas salvas")
    
    def save_figures(self, data):
        """Salva figuras"""
        # 1. Attrition Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        data['attrition_dist'].plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
        axes[0].set_title('Distribuição de Attrition', fontsize=14, fontweight='bold')
        axes[0].set_xticklabels(['No', 'Yes'], rotation=0)
        data['attrition_pct'].plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'])
        axes[1].set_title('Distribuição de Attrition (%)', fontsize=14, fontweight='bold')
        axes[1].set_xticklabels(['No', 'Yes'], rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, f'01_attrition_{self.timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Gender Analysis
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        data['gender_dist'].plot(kind='bar', ax=axes[0], color=['#3498db', '#e91e63'])
        axes[0].set_title('Distribuição de Gender', fontsize=14, fontweight='bold')
        data['gender_attrition'].plot(kind='bar', ax=axes[1], stacked=False)
        axes[1].set_title('Taxa de Attrition por Gender', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, f'02_gender_{self.timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. SMOTE Comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        pd.Series(data['y_train']).value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
        axes[0].set_title('Antes do SMOTE', fontsize=14, fontweight='bold')
        axes[0].set_xticklabels(['No', 'Yes'], rotation=0)
        pd.Series(data['y_train_balanced']).value_counts().plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'])
        axes[1].set_title('Depois do SMOTE', fontsize=14, fontweight='bold')
        axes[1].set_xticklabels(['No', 'Yes'], rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, f'03_smote_{self.timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Confusion Matrix - Baseline
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(data['cm_baseline'], annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.title(f'Confusion Matrix - {data["baseline_name"]}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, f'04_cm_baseline_{self.timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Confusion Matrix - Adversarial
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(data['cm_adversarial'], annot=True, fmt='d', cmap='Greens', 
                   xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.title('Confusion Matrix - Adversarial', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, f'05_cm_adversarial_{self.timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ 5 figuras salvas (300 DPI)")
    
    def save_models(self, data):
        """Salva modelos"""
        joblib.dump(data['baseline_model'], 
                   os.path.join(self.models_path, f'baseline_{self.timestamp}.pkl'))
        joblib.dump(data['scaler'], 
                   os.path.join(self.models_path, f'scaler_{self.timestamp}.pkl'))
        joblib.dump(data['label_encoders'], 
                   os.path.join(self.models_path, f'encoders_{self.timestamp}.pkl'))
        
        print(f"   ✓ 3 modelos salvos")
    
    def create_readme(self, data):
        """Cria README"""
        readme = f"""# Resultados - Adversarial Debiasing

**Data:** {datetime.now().strftime("%d/%m/%Y %H:%M")}
**Timestamp:** {self.timestamp}

## Performance

| Modelo | Accuracy | F1-Score |
|--------|----------|----------|
| Baseline | {data['accuracy_baseline']:.4f} | {data['f1_baseline']:.4f} |
| Adversarial | {data['accuracy_adversarial']:.4f} | {data['f1_adversarial']:.4f} |

## Fairness (Female vs Male)

| Métrica | Baseline | Adversarial |
|---------|----------|-------------|
| Demographic Parity | {data['fairness_baseline']['demographic_parity_diff']:.4f} | {data['fairness_adversarial']['demographic_parity_diff']:.4f} |
| Disparate Impact | {data['fairness_baseline']['disparate_impact']:.4f} | {data['fairness_adversarial']['disparate_impact']:.4f} |

## Arquivos

- `metrics/` - Métricas em JSON e CSV
- `figures/` - Visualizações (300 DPI)
- `models/` - Modelos treinados (.pkl)
"""
        
        with open(os.path.join(self.base_path, f'README_{self.timestamp}.md'), 'w') as f:
            f.write(readme)
        
        print(f"   ✓ README criado")