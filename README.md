# Adversarial Debiasing para Mitigação de Vieses em Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Trabalho Final - Disciplina de Inteligência Artificial  
Mestrado Profissional em Engenharia de Produção e Sistemas Computacionais  
Universidade Federal Fluminense

## 📋 Sobre o Projeto

Este projeto implementa técnicas de **Adversarial Debiasing** para mitigação de vieses em modelos de Machine Learning, utilizando o dataset **IBM HR Analytics Employee Attrition**. O objetivo é criar modelos de predição de rotatividade de funcionários que sejam justos em relação a atributos sensíveis como gênero e idade.

### 🎯 Objetivos

- Analisar vieses presentes em modelos de ML para predição de attrition
- Implementar Adversarial Debiasing usando AIF360
- Realizar Grid Search de hiperparâmetros (λ) para otimização
- Comparar métricas de fairness antes e depois da mitigação
- Avaliar trade-offs entre performance e equidade
- Utilizar SHAP para análise de explicabilidade

## 📊 Dataset

**IBM HR Analytics Employee Attrition Dataset**

- **Fonte:** [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Amostras:** 1.470 funcionários
- **Features:** 35 variáveis (demográficas, profissionais, satisfação)
- **Target:** Attrition (Yes/No)
- **Atributos sensíveis:** Gender, Age

### Características do Dataset

- **Desbalanceamento:** 84% No Attrition, 16% Yes Attrition
- **Distribuição de gênero:** ~60% Male, ~40% Female
- **Proxies identificados:** JobRole, Department (correlacionados com Gender)

## 🏗️ Estrutura do Projeto

```
adversarial-debiasing-hr/
├── data/
│   ├── raw/                           # Dataset original
│   │   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│   └── processed/                     # Dados processados
├── notebooks/
│   └── adversarial_debiasing_complete.ipynb  # Notebook principal
├── src/
│   ├── __init__.py
│   └── save_results.py                # Salvamento de resultados
├── results/
│   ├── figures/                       # Gráficos e visualizações (300 DPI)
│   ├── metrics/                       # Métricas salvas (CSV/JSON)
│   └── models/                        # Modelos treinados
├── requirements.txt                   # Dependências
├── .gitignore                         # Arquivos ignorados pelo Git
├── README.md                          # Este arquivo
└── LICENSE                            # Licença do projeto
```

## 🔧 Instalação e Configuração

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Git

### Instalação

1. **Clone o repositório:**

```bash
git clone https://github.com/seu-usuario/adversarial-debiasing-hr.git
cd adversarial-debiasing-hr
```

2. **Crie um ambiente virtual (recomendado):**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Instale as dependências:**

```bash
pip install -r requirements.txt
```

4. **Baixe o dataset:**

Opção 1 - Manual:

- Acesse [Kaggle Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Baixe o arquivo CSV
- Coloque em `data/raw/`

## 🚀 Como Executar

### Opção 1: Jupyter Notebook Local

```bash
# Ativar ambiente virtual
venv\Scripts\activate     # Windows
source venv/bin/activate  # Linux/Mac

# Iniciar Jupyter
jupyter notebook

# Abrir notebooks/adversarial_debiasing_complete.ipynb
```

### Opção 2: VSCode

1. Abra a pasta do projeto no VSCode
2. Instale a extensão "Jupyter"
3. Abra `notebooks/adversarial_debiasing_complete.ipynb`
4. Selecione o kernel do ambiente virtual
5. Execute as células sequencialmente

### Opção 3: Python Específico (Windows)

```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

## 📈 Metodologia

### 1. Análise Exploratória

- Análise de distribuições (target, variáveis sensíveis)
- Identificação de correlações e proxies
- Testes estatísticos (Chi-quadrado)
- Visualizações de viés por grupo protegido

### 2. Preprocessamento

- Encoding de variáveis categóricas (LabelEncoder)
- Normalização/padronização (StandardScaler)
- Split treino/teste (70/30) estratificado
- Tratamento de desbalanceamento com **SMOTE** (k_neighbors=5)

### 3. Modelos Baseline

Foram testados **dois modelos baseline** para comparação:

- **Baseline v1:** Random Forest com `class_weight='balanced'`
- **Baseline v2:** Random Forest treinado com dados balanceados por SMOTE

**Configuração:**

- n_estimators: 100
- max_depth: 10
- random_state: 42

**Critério de seleção:** Modelo com maior F1-Score

### 4. Adversarial Debiasing

**Grid Search de Hiperparâmetros:**

Implementação usando AIF360 com otimização do parâmetro λ (adversary_loss_weight):

- **Lambda testados:** [0.01, 0.05, 0.1, 0.2, 0.5]
- **Configuração:**
  - Épocas: 50
  - Batch size: 128
  - Atributo sensível: Gender (Female = 1, Male = 0)

**Critério de seleção:**

1. Filtrar modelos com Demographic Parity < 0.1
2. Entre os válidos, selecionar o com maior F1-Score

**Resultado:** λ ótimo varia entre 0.2 e 0.5 dependendo da execução (devido à natureza estocástica do treinamento)

### 5. Análise de Explicabilidade

- **SHAP values** para interpretação de features
- Identificação das top 10 features mais importantes
- Análise de proxies e suas relações com atributos sensíveis
- Summary plots e feature importance

## 📊 Resultados

### Performance dos Modelos (Exemplo - Resultados variam por execução)

| Modelo                     | Accuracy | F1-Score | AUC-ROC |
| -------------------------- | -------- | -------- | ------- |
| Baseline v1 (class_weight) | 0.8614   | 0.8500   | 0.8300  |
| Baseline v2 (SMOTE)        | 0.8367   | 0.8402   | 0.8100  |
| Adversarial (λ=0.2)        | 0.8617   | 0.8621   | -       |
| Adversarial (λ=0.5)        | 0.8503   | 0.8520   | -       |

### Métricas de Fairness (Female vs Male)

| Métrica                 | Baseline v2 | Adversarial (λ=0.5) | Melhoria       |
| ----------------------- | ----------- | ------------------- | -------------- |
| Demographic Parity Diff | 0.1139      | 0.0121              | ✅ 89% melhor  |
| Disparate Impact        | 0.5444      | 1.0730              | ✅ Quase ideal |
| Equal Opportunity Diff  | 0.0214      | 0.0821              | -              |

### Grid Search de Lambda

O experimento demonstrou que:

- **λ baixo (0.01-0.05):** Prioriza accuracy, fairness moderada
- **λ médio (0.1-0.2):** Bom equilíbrio, pode ter melhor accuracy
- **λ alto (0.5):** Excelente fairness, mantém alta performance

**Observação importante:** Devido à natureza estocástica das redes neurais adversariais, diferentes execuções podem selecionar λ=0.2 ou λ=0.5 como ótimo. Ambos apresentam excelentes resultados.

### Principais Insights

1. ✅ **Adversarial Debiasing reduziu significativamente o viés**

   - Demographic Parity: ~0.11 → ~0.01-0.05 (redução de até 90%)
   - Disparate Impact próximo de 1.0 (ideal)

2. ⚖️ **Trade-off aceitável ou inexistente:**

   - Em alguns casos, λ=0.5 manteve accuracy similar ao baseline
   - Em outros, λ=0.2 até SUPEROU o baseline em performance
   - Perda máxima de accuracy < 3% quando ocorre

3. 🔍 **Proxies identificados:**

   - JobRole e Department correlacionam fortemente com Gender
   - Modelo adversarial aprende a ignorar esses atalhos

4. 📊 **SMOTE vs class_weight:**

   - Ambas as abordagens são válidas
   - Baseline v2 (SMOTE) geralmente selecionado por melhor F1-Score

5. 🎲 **Variabilidade estocástica:**
   - Resultados variam ligeiramente entre execuções
   - Zona ótima de λ entre 0.2-0.5 consistentemente identificada
   - Demonstra robustez da metodologia

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **Bibliotecas principais:**
  - `pandas 2.0+`, `numpy 1.24+` - Manipulação de dados
  - `scikit-learn 1.3+` - Modelos de ML e preprocessamento
  - `aif360 0.5+` - Fairness e Adversarial Debiasing
  - `fairlearn 0.9+` - Métricas de fairness adicionais
  - `shap 0.42+` - Explicabilidade (SHAP values)
  - `imbalanced-learn 0.11+` - SMOTE para balanceamento
  - `tensorflow 2.x` - Backend para Adversarial Debiasing
  - `matplotlib 3.7+`, `seaborn 0.12+` - Visualizações
  - `jupyter` - Ambiente interativo

## 📝 Configurações do Experimento

Todas as configurações podem ser ajustadas no dicionário `CONFIG`:

```python
CONFIG = {
    'test_size': 0.3,               # 70/30 split
    'smote_k_neighbors': 5,         # Vizinhos para SMOTE
    'rf_n_estimators': 100,         # Árvores no Random Forest
    'rf_max_depth': 10,             # Profundidade máxima
    'adversarial_epochs': 50,       # Épocas de treinamento
    'adversarial_batch_size': 128,  # Tamanho do batch
    'adversarial_lambda': 0.1,      # Peso inicial (ajustado por grid search)
    'shap_sample_size': 500         # Amostras para SHAP
}
```

## 📚 Referências

1. **Zhang, B. H., Lemoine, B., & Mitchell, M. (2018).** _Mitigating Unwanted Biases with Adversarial Learning._ AIES 2018. [arXiv:1801.07593](https://arxiv.org/abs/1801.07593)

2. **Bellamy, R. K. E., et al. (2019).** _AI Fairness 360: An extensible toolkit for detecting and mitigating algorithmic bias._ IBM Journal of Research and Development, 63(4/5).

3. **Lundberg, S. M., & Lee, S. I. (2017).** _A Unified Approach to Interpreting Model Predictions._ NeurIPS 2017.

4. **Hardt, M., Price, E., & Srebro, N. (2016).** _Equality of Opportunity in Supervised Learning._ NeurIPS 2016.

5. **Chawla, N. V., et al. (2002).** _SMOTE: Synthetic Minority Over-sampling Technique._ Journal of Artificial Intelligence Research, 16, 321-357.

### Links Úteis

- [AIF360 Documentation](https://aif360.readthedocs.io/)
- [Fairlearn](https://fairlearn.org/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [IBM HR Analytics Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- [Imbalanced-Learn](https://imbalanced-learn.org/)

## 🔬 Reprodutibilidade

Para garantir reprodutibilidade dos resultados:

1. **Seeds fixadas:** `RANDOM_STATE = 42` em todos os componentes aleatórios
2. **Eager execution desabilitada:** Necessário para AIF360
3. **Versões fixas:** Use o `requirements.txt` fornecido

**Nota:** Devido à natureza estocástica do Adversarial Debiasing (redes neurais), pequenas variações nos resultados são esperadas entre execuções. O grid search foi implementado justamente para identificar a zona robusta de hiperparâmetros.

## 💾 Resultados Salvos

Ao executar o notebook, os seguintes arquivos são gerados automaticamente em `results/`:

### `figures/` (300 DPI)

- Distribuição de Attrition
- Análise de variáveis sensíveis (Gender, Age)
- Comparação SMOTE (antes/depois)
- Confusion matrices (Baseline e Adversarial)
- SHAP summary plots
- Grid search visualizations (trade-off plots)

### `metrics/`

- `comparison_performance.csv` - Comparação de performance
- `comparison_fairness.csv` - Comparação de fairness
- `feature_importance.csv` - Importância das features (SHAP)
- `grid_search_lambda.csv` - Resultados de todos os lambdas testados

### `models/`

- `baseline_model.pkl` - Modelo baseline selecionado
- `scaler.pkl` - StandardScaler treinado
- `label_encoders.pkl` - Encoders das variáveis categóricas

## 👥 Autores

**Christian Ferreira**
**Wanderley Rangel**

**Orientador:** Prof. Dr. Leonard Barreto Moreira  
Universidade Federal Fluminense - UFF

## 📄 Licença

Este projeto está sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

---
