# PRD — ML-Based Money Muling Detection

## Project Title

**ML-Based Detection and Visualization of Money-Muling Activity in Financial Transaction Networks**

## 1. Objective

Build an ML-based system that detects **potential money-mule accounts and suspicious money flows** by analyzing financial transactions as a network.

The system should demonstrate how:

> **Transaction data → Network representation → ML model → Risk score → Suspicious account/transaction visualization**

The primary goal is **demonstration and explainability**, not production-grade financial fraud detection.

---

## 2. Dataset

### Primary Dataset

**IBM AML HI-Small Dataset**

Synthetic financial transaction dataset developed for Anti-Money Laundering research.

It contains:

- Sender account
- Receiver account
- Transaction amount
- Timestamp
- Currency
- Payment type
- Money-laundering label

It also contains different transaction/network patterns representing suspicious activity.

### Why this dataset?

- Designed specifically for AML research
- Has labelled suspicious transactions
- Naturally represents a graph
- Large enough to demonstrate realistic network behavior
- Works with the selected GNN model

---

## 3. ML Model

### Primary Model

**FraudSentinel — GINE Graph Neural Network**

Architecture:

```text
Financial Transactions
        ↓
Transaction Graph
        ↓
GINE Layers
        ↓
Node/Edge Representations
        ↓
Transaction Classification
        ↓
Suspicion Probability
```

The model will provide a **suspicion score for transactions**.

We can then aggregate suspicious transactions to identify potentially suspicious accounts.

### Optional Baseline

Implement a simple **XGBoost/Random Forest** model using manually engineered transaction features.

This gives us a useful comparison:

> Traditional ML vs Graph-based ML

---

## 4. System Components

### A. Data Processing Module

Responsibilities:

- Load IBM AML dataset
- Clean and preprocess transactions
- Convert accounts into graph nodes
- Convert transactions into directed edges
- Prepare data in the format expected by the GNN

### B. Transaction Graph Module

Represent the financial system as:

```text
Account = Node

Transaction = Directed Edge

Amount = Edge Weight

Timestamp = Transaction Attribute
```

Example:

```text
A ──₹20,000──→ B
C ──₹15,000──→ B
B ──₹32,000──→ D
```

### C. ML Detection Module

Run the pretrained GINE model on transactions.

Output:

```text
Transaction ID
Sender
Receiver
Risk Score
Predicted Label
```

Example:

```text
T1842
A102 → A592

Risk Score: 0.87
Prediction: Suspicious
```

### D. Account Risk Module

Aggregate suspicious transactions to calculate an account-level risk score.

Example:

```text
Account: A592

Incoming transactions: 42
Outgoing transactions: 35
Suspicious transactions: 18

Risk Score: 91%

Status: HIGH RISK
```

---

## 5. Visualization Dashboard

This should be the **main part of the demo**.

### 5.1 Transaction Network

Interactive graph showing:

```text
        A
        │
        ↓
B →  MULE  → D
        ↑
        │
        C
```

Users should be able to:

- zoom
- pan
- click accounts
- inspect transactions
- highlight suspicious edges
- filter by risk score

### 5.2 Account Details

When an account is selected:

```text
Account: A592

Total Transactions     77
Incoming               42
Outgoing               35
Unique Senders         31
Unique Receivers       28

Suspicious Transactions 18

ML Risk Score          91%
Risk Level             HIGH
```

### 5.3 Transaction Details

Clicking an edge should show:

```text
Transaction ID: T1842

Sender: A102
Receiver: A592
Amount: ₹20,000
Timestamp: 14:32

ML Risk Score: 87%

Prediction:
⚠ Suspicious
```

### 5.4 Money-Flow Visualization

Show the movement of money through the network:

```text
A ──₹50K──→ M
B ──₹30K──→ M
C ──₹20K──→ M
             │
             ├──₹45K──→ X
             └──₹40K──→ Y
```

This is the **money-mule behavior you want the professor to visually understand**.

---

## 6. Main User Flow

The demo should work like this:

```text
1. Load Dataset
       ↓
2. Build Transaction Graph
       ↓
3. Run ML Model
       ↓
4. Generate Risk Scores
       ↓
5. Display Transaction Network
       ↓
6. Highlight Suspicious Transactions
       ↓
7. Identify High-Risk Accounts
       ↓
8. Show Account + Transaction Details
```

---

## 7. Evaluation

We should report:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix

Because AML datasets are highly imbalanced, **accuracy should not be the primary metric**.
