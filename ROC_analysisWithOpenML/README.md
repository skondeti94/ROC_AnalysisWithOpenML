Dorothea Dataset(4137):
Task: Designed for drug discovery to identify thrombin-binding compounds. Involves classifying compounds as active (+1) or inactive (-1).
Features:
  - Comprises real and probe features (100,000 attributes).
  - Numerical nature.
  - Distractor features called 'probes' added (no predictive power).
Target:
  - Binary target: +1 (active compounds), -1 (inactive compounds).
Dataset:
  - Split into training, validation, and test sets.
  - Varying quantities of positive and negative examples.

Bio response Dataset(4134):
Task: Predicts biological response based on chemical properties. Binary classification to determine a molecule's ability to elicit a response (1) or not (0).
Features:
  - Each row represents a molecule, with 1776 molecular descriptors.
  - Descriptors capture size, shape, or elemental constitution.
  - Descriptor matrix normalized.
Target:
  - Binary target: 1 (response), 0 (no response).
Dataset:
  - Original training and test sets merged into a single dataset.
