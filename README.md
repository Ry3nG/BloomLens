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
4. Create training pipeline with both cross-entropy and triplet loss
5. Implement MixUp augmentation
6. Monitor performance

**Deliverable**:
- Performance table of pre-trained models on the full dataset
- Ablation study showing impact of MixUp and triplet loss

### B. Reduced Data Experiments

**Objective**: Test pre-trained backbone performance with systematically reduced dataset sizes.

**Steps**:
1. Define data reduction strategy
2. Implement stratified sampling
3. Train models on reduced datasets
4. Generate performance comparison table
5. Analyze computational efficiency at different data scales

**Deliverable**:
- Table showing model performance degradation as dataset size is reduced
- Computational efficiency metrics (training time, memory usage)

### C. Few-Shot Learning Implementation

**Objective**: Develop a few-shot learning model that outperforms pre-trained models with reduced data.

**Steps**:
1. Select few-shot learning framework
2. Prepare dataset for few-shot tasks
3. Implement and train few-shot model
4. Evaluate performance against baselines
5. Conduct ablation studies on different components

**Deliverable**:
- Performance comparison of few-shot model vs. pre-trained backbones on reduced data
- Component-wise contribution analysis

### D. Explainability Analysis

**Objective**: Visualize and understand model decision-making processes for flower classification.

**Steps**:
1. Choose explainability method (e.g., Grad-CAM, SHAP)
2. Generate attention maps for all models
3. Compare explanations between few-shot and pre-trained models
4. Quantitative evaluation of attention maps
5. Analysis of failure cases

**Deliverable**:
- Visualizations and comparison of model interpretability
- Quantitative metrics for attention map evaluation
- Failure case analysis with explanations

## Final Report Structure

1. **Introduction**
   - Problem definition
   - Importance of reduced data handling and explainability
   - Technical challenges in fine-grained classification

2. **Related Work**
   - Pre-trained models in image classification
   - Few-shot learning techniques
   - Explainability methods
   - MixUp and triplet loss applications

3. **Methodology**
   - Experimental setup for baseline models
   - Data reduction strategy
   - Few-shot learning implementation
   - Advanced training techniques (MixUp, triplet loss)
   - Explainability methods

4. **Experiments**
   - Baseline results (full dataset)
   - Reduced data experiment results
   - Few-shot learning model performance
   - Ablation studies
   - Computational efficiency analysis

5. **Explainability**
   - Attention maps presentation
   - Quantitative evaluation metrics
   - Interpretation comparisons
   - Failure case analysis

6. **Discussion**
   - Impact of reduced data on model performance
   - Advantages and limitations of few-shot learning
   - Interpretation of explainability results
   - Trade-offs between performance and computational efficiency

7. **Conclusion**
   - Key findings summary
   - Future research directions
   - Practical implications

## Project Timeline

- Week 1-2: Environment setup, baseline implementation
- Week 3: Reduced data experiments, MixUp implementation
- Week 4: Few-shot learning implementation
- Week 5: Explainability analysis
- Final Week: Report writing and result compilation

## Getting Started

(Add instructions for setting up the project, installing dependencies, and running experiments)

## Contributors

(List project contributors)

## License

(Specify the project license)
