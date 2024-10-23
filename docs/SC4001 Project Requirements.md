# SC4001 CE/CZ4042: Neural Networks and Deep Learning Group Project

**Last Update:** 16 Oct 2024
**Start Date:** 11 October 2024
**Deadline:** 15 November 2024 (11:59 PM)

## Project Overview

Students will propose and execute a final project on an application or research issue related to neural networks and deep learning. Projects can be carried out in groups of up to three members.

### Project Objectives
1. Develop a potential technique for the application or to mitigate an issue
2. Implement associated code
3. Compare with existing methods

Students may choose, focus, and expand on project ideas A – F (only F is detailed in this document).

## Submission Requirements

Submit the following to NTULearn by the deadline:

1. **Project Report**
   - Format: PDF
   - Length: 10 A4 pages (Arial 10 font)
   - Exclusions: References, content page, and cover page

2. **Associated Code**
   - Format: ZIP file
   - Requirements: Well-commented and easily testable

### Report Structure
- Front page (with team members' names)
- Introduction to the project idea
- Review of existing techniques
- Description of methods used
- Experiments and results
- Discussion

## Assessment Criteria

| Criterion | Weight |
|-----------|--------|
| Project execution | 30% |
| Experiments and results | 30% |
| Report presentation | 15% |
| Novelty | 15% |
| Peer review | 10% |

**Late Submission Penalty:** 5% per day (up to 3 days)

## Project Option F: Flowers Recognition

### Dataset: Oxford Flowers 102

- 102 flower categories common in the UK
- 40 to 258 images per class
- Variations: scale, pose, light, intra-class, and inter-class similarities

#### Dataset Split
- Training set: 1,020 images (10 per class)
- Validation set: 1,020 images (10 per class)
- Test set: 6,149 images (minimum 20 per class)

### Main Task
Implement a classification model for flower image categorization.

### Dataset Resources
- [TorchVision Flowers102 Dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.Flowers102.html)
- [Original Oxford Flowers 102 Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

### Additional Tasks

1. **Architecture Modification**
   - Increase network depth
   - Reduce parameters
   - Explore advanced techniques (e.g., deformable convolution, visual prompt tuning for Transformers)

2. **Few-shot Learning Analysis**
   - Evaluate model performance with reduced training data

3. **Advanced Transformation Techniques**
   - Implement MixUp (refer to original paper and PyTorch implementation)

4. **Advanced Loss Functions**
   - Experiment with triplet loss
