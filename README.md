# BloomLens

BloomLens: A Few-Shot Learning Framework for Fine-Grained Flower Classification

## Important Resources
1. https://arxiv.org/pdf/2110.07097 : this has the table comparism we want
2. https://www.researchgate.net/profile/Huthaifa-Almogdady/publication/329191674_A_Flower_Recognition_System_Based_On_Image_Processing_And_Neural_Networks/links/5bfc1036a6fdcc76e721c657/A-Flower-Recognition-System-Based-On-Image-Processing-And-Neural-Networks.pdf : This is the benchmark we want to beat

## Project Overview

This project aims to develop and evaluate a few-shot learning framework for fine-grained flower classification using the Oxford 102 Flower dataset. The project encompasses several key components, including baseline model evaluation, reduced data experiments, few-shot learning implementation, and explainability analysis.

## Project Components

### A. Baseline Model Evaluation

**Objective**: Test and verify the performance of different pre-trained models on the Oxford 102 Flower dataset to establish baseline accuracies.

**Steps**:
1. Set up environment
2. Load the Oxford Flower Dataset
3. Implement pre-trained models
4. Create training pipeline
5. Monitor performance

**Deliverable**: Performance table of pre-trained models on the full dataset.

### B. Reduced Data Experiments

**Objective**: Test pre-trained backbone performance with systematically reduced dataset sizes.

**Steps**:
1. Define data reduction strategy
2. Implement stratified sampling
3. Train models on reduced datasets
4. Generate performance comparison table

**Deliverable**: Table showing model performance degradation as dataset size is reduced.

### C. Few-Shot Learning Implementation

**Objective**: Develop a few-shot learning model that outperforms pre-trained models with reduced data.

**Steps**:
1. Select few-shot learning framework
2. Prepare dataset for few-shot tasks
3. Implement and train few-shot model
4. Evaluate performance against baselines

**Deliverable**: Performance comparison of few-shot model vs. pre-trained backbones on reduced data.

### D. Explainability Analysis

**Objective**: Visualize and understand model decision-making processes for flower classification.

**Steps**:
1. Choose explainability method (e.g., Grad-CAM, SHAP)
2. Generate attention maps for all models
3. Compare explanations between few-shot and pre-trained models

**Deliverable**: Visualizations and comparison of model interpretability.

## Final Report Structure

1. **Introduction**
   - Problem definition
   - Importance of reduced data handling and explainability

2. **Related Work**
   - Pre-trained models in image classification
   - Few-shot learning techniques
   - Explainability methods

3. **Methodology**
   - Experimental setup for baseline models
   - Data reduction strategy
   - Few-shot learning implementation
   - Explainability methods

4. **Experiments**
   - Baseline results (full dataset)
   - Reduced data experiment results
   - Few-shot learning model performance

5. **Explainability**
   - Attention maps presentation
   - Interpretation comparisons

6. **Discussion**
   - Impact of reduced data on model performance
   - Advantages and limitations of few-shot learning
   - Interpretation of explainability results

7. **Conclusion**
   - Key findings summary
   - Future research directions

## Getting Started

(Add instructions for setting up the project, installing dependencies, and running experiments)

## Contributors

(List project contributors)

## License

(Specify the project license)
