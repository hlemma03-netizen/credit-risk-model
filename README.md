# Credit Risk Probability Model

This project develops an end-to-end credit risk scoring system for Bati Bank using transaction data from an eCommerce platform.

## Project Structure

* data/
* notebooks/
* src/
* tests/
* Dockerfile
* docker-compose.yml

## Objective

Build a machine learning system that predicts customer credit risk and generates credit scores for buy-now-pay-later services.
## Credit Scoring Business Understanding

### 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The Basel II Accord requires financial institutions to maintain transparent and reliable risk management processes. Credit risk models must be interpretable so that lending decisions can be explained to regulators, auditors, and business stakeholders. Well-documented models improve accountability, support compliance requirements, and enable ongoing monitoring and validation. As a result, model transparency is often as important as predictive performance in regulated financial environments.

### 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

The available dataset does not contain information about whether customers actually defaulted on credit obligations. Therefore, a proxy variable must be created to approximate credit risk. In this project, customer behavior patterns derived from Recency, Frequency, and Monetary (RFM) analysis are used to identify potentially high-risk and low-risk customer groups.

The use of proxy variables introduces several risks. The proxy may not perfectly represent true default behavior, leading to incorrect classifications. This can result in creditworthy customers being denied access to credit or risky customers being approved. Consequently, model results should be interpreted carefully and validated continuously.

### 3. What are the key trade-offs between a simple, interpretable model and a high-performance model in a regulated financial context?

Simple models such as Logistic Regression combined with Weight of Evidence (WoE) transformations offer strong interpretability. Stakeholders can understand how individual features influence predictions, making these models easier to explain and validate.

More advanced models such as Gradient Boosting often achieve higher predictive accuracy by capturing complex patterns in the data. However, these models are generally less transparent and harder to explain. In regulated financial environments, organizations must balance predictive performance against interpretability, compliance requirements, and stakeholder trust.

