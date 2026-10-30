# Deep Learning Midterm Project

**Customer Churn Prediction (DNN) and Pneumonia Detection from Chest X-Rays (CNN)**

This report covers two deep learning models built for the midterm project. The first is a deep neural network (DNN) that predicts whether a telecom customer will leave the company, using tabular customer data. The second is a convolutional neural network (CNN) that classifies chest X-ray images as normal or pneumonia. Both were built in Python with TensorFlow/Keras and trained on a laptop CPU. All graphs mentioned below (EDA plots, training curves, confusion matrices and ROC curves) are shown inside the notebooks.

| Model | Dataset | Test accuracy | ROC-AUC | Notebook |
|-------|---------|---------------|---------|----------|
| DNN | Telco Customer Churn (7,043 customers) | 79.7% | 0.837 | `churn_dnn.ipynb` |
| CNN | Chest X-Ray Images (Pneumonia) (5,856 images) | 71.6% | 0.915 | `pneumonia_cnn.ipynb` |

---

## Part 1: DNN Model - Customer Churn Prediction

### 1.1 Problem Definition

Telecom companies lose a large share of their revenue when customers cancel their service (churn), and getting a new customer costs much more than keeping an existing one. The problem is to predict, from a customer's account details, services and billing information, whether they are likely to churn, so the company can contact them with an offer before they leave. This is a binary classification problem: Churn = Yes (1) or No (0).

### 1.2 Dataset Description and Source Link

**Dataset:** Telco Customer Churn (IBM sample dataset)
**Source:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn

| Property | Value |
|----------|-------|
| Rows (customers) | 7,043 |
| Columns | 21 (19 features, customerID and the target Churn) |
| Class balance | 5,174 No (73.5%), 1,869 Yes (26.5%) |
| Missing values | 11 blank values in TotalCharges |

- **Demographics:** gender, SeniorCitizen, Partner, Dependents
- **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Account and billing:** tenure (0-72 months), Contract, PaperlessBilling, PaymentMethod, MonthlyCharges (18.25-118.75), TotalCharges

Exploratory analysis showed that customers on month-to-month contracts, new customers with low tenure, and customers with high monthly charges churn much more often than the rest.

### 1.3 Proposed Solution

A fully connected deep neural network is trained on the encoded customer features. For each customer it outputs a probability of churn between 0 and 1, and customers with a probability of 0.5 or more are flagged as likely to churn. The company can then target these customers with retention offers. Because the classes are imbalanced, the model is judged on precision, recall, F1 and ROC-AUC as well as accuracy.

### 1.4 Data Preprocessing

1. **Fixed TotalCharges:** it was stored as text. It was converted to numbers with `pd.to_numeric(errors='coerce')`; the 11 blank values belong to customers with tenure 0 who have not been billed yet, so they were filled with 0.
2. **Dropped customerID**, since it is only an identifier and carries no information about churn.
3. **Encoded the target:** Yes = 1, No = 0.
4. **Encoded binary columns** (e.g. gender, Partner, PaperlessBilling) as 0/1.
5. **One-hot encoded multi-value columns** (e.g. Contract, PaymentMethod, InternetService) with `drop_first=True`, so no false order is imposed on the categories and no redundant column is created. This gave **30 input features**.
6. **Split the data** 70/15/15, stratified on Churn: 4,930 training, 1,056 validation and 1,057 test customers.
7. **Scaled** tenure, MonthlyCharges and TotalCharges with `StandardScaler`, fitted on the training set only to avoid data leakage.

### 1.5 Model Architecture

| Layer | Output shape | Parameters |
|-------|--------------|------------|
| Input | 30 | 0 |
| Dense, ReLU | 64 | 1,984 |
| Batch Normalization | 64 | 256 |
| Dropout (0.3) | 64 | 0 |
| Dense, ReLU | 32 | 2,080 |
| Batch Normalization | 32 | 128 |
| Dropout (0.3) | 32 | 0 |
| Dense, ReLU | 16 | 528 |
| Dense, Sigmoid (output) | 1 | 17 |
| **Total** | | **4,993 (4,801 trainable)** |

| Setting | Value |
|---------|-------|
| Loss | Binary cross-entropy |
| Optimizer | Adam, learning rate 0.001 |
| Batch size / max epochs | 32 / 100 |
| EarlyStopping | monitor val_loss, patience 10, restore best weights |
| ReduceLROnPlateau | monitor val_loss, factor 0.5, patience 5, min 0.00001 |
| Regularisation | Dropout and Batch Normalization |

### 1.6 Training Results and Evaluation

Training stopped early at **epoch 18**. The lowest validation loss (0.426) was reached at epoch 8, and after 10 epochs with no improvement EarlyStopping stopped training and restored the epoch 8 weights. ReduceLROnPlateau halved the learning rate to 0.0005 after epoch 13. Training and validation accuracy both ended around 80-81% and the validation loss stayed flat, so the model did not overfit.

| Metric (test set, 1,057 customers) | Value |
|-----------------------------------|-------|
| Accuracy | 79.66% |
| Precision (Churn) | 0.662 |
| Recall (Churn) | 0.480 |
| F1 score (Churn) | 0.557 |
| ROC-AUC | 0.837 |
| Recall (No Churn) | 0.91 |

**Confusion matrix (test set)**

| | Predicted No Churn | Predicted Churn |
|---|---|---|
| **Actual No Churn** | 707 | 69 |
| **Actual Churn** | 146 | 135 |

Out of 1,057 test customers, 842 were classified correctly. The model correctly recognised 707 of the 776 customers who stayed, but it caught only 135 of the 281 customers who churned; 146 churners were missed and 69 loyal customers were wrongly flagged. A model that always predicts "No Churn" would already reach 73.5% accuracy, so the ROC-AUC of 0.837 and the per-class recall are the more meaningful measures here.

### 1.7 Conclusion

The DNN reached **79.7% test accuracy** and a ROC-AUC of **0.837**, improving on the 73.5% baseline by about 6 percentage points. Published results on this dataset generally top out around 80-82% accuracy, because customers with identical details often make different choices, so this is close to the realistic limit for the data.

The model's weak point is recall for the churn class: it finds about half of the customers who leave. An earlier experiment with class weights raised churn recall but lowered accuracy to about 76%, so the final model was trained without them. If catching churners matters most to the company, class weights or a lower decision threshold (for example 0.3) would be a reasonable choice. Further improvements could come from tuning the threshold on the validation set, adding features such as average charge per month of tenure, and comparing against logistic regression or gradient boosting.

---

## Part 2: CNN Model - Pneumonia Detection from Chest X-Rays

### 2.1 Problem Definition

Pneumonia is a lung infection that is a leading cause of death in young children worldwide. It is diagnosed mainly from chest X-rays, but reading them takes a trained radiologist, and many hospitals do not have enough of them. The problem is to automatically classify a chest X-ray as **Normal** or **Pneumonia**, so that likely pneumonia cases can be flagged and reviewed first.

### 2.2 Dataset Description and Source Link

**Dataset:** Chest X-Ray Images (Pneumonia)
**Source:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset contains pediatric chest X-rays (patients aged one to five) from Guangzhou Women and Children's Medical Center, graded by expert physicians. The images are grayscale JPEGs of very different sizes (heights from 156 to 2,543 pixels in a sample of 200).

| Folder | Normal | Pneumonia | Total |
|--------|--------|-----------|-------|
| train | 1,341 | 3,875 | 5,216 |
| val | 8 | 8 | 16 |
| test | 234 | 390 | 624 |

### 2.3 Proposed Solution

A convolutional neural network is trained from scratch to learn visual patterns in the lungs, such as the cloudy areas (opacities) caused by infection. The network outputs the probability that an X-ray shows pneumonia. Data augmentation and class weights are used to deal with the limited number of images and the class imbalance (about 3 pneumonia images for every normal one).

### 2.4 Data Preprocessing

1. **Collected file paths and labels** into a table (Normal = 0, Pneumonia = 1).
2. **Re-split train and validation:** the given validation folder has only 16 images, too few to judge the model. The train and val folders were merged and split 80/20 with stratification, giving **4,185 training** and **1,047 validation** images. The 624 test images were kept untouched for the final evaluation.
3. **Loaded each image as one grayscale channel**, resized it to **180 x 180** pixels and scaled pixel values from 0-255 to 0-1.
4. **Built `tf.data` pipelines** with shuffling (training only), batching (32) and prefetching.
5. **Computed balanced class weights** from the training labels: 1.94 for Normal and 0.67 for Pneumonia.
6. **Data augmentation** during training only: random rotation (5%), zoom (10%) and shifts (5%). Horizontal flipping was not used, because in a real chest X-ray the heart is always on the same side.

### 2.5 Model Architecture

The model has four convolutional blocks. Each block has two 3x3 Conv2D layers (each followed by Batch Normalization and ReLU) and a 2x2 MaxPooling layer, so the number of filters doubles while the image size halves.

| Layer / block | Output shape | Parameters |
|---------------|--------------|------------|
| Input (grayscale) | 180 x 180 x 1 | 0 |
| Data augmentation | 180 x 180 x 1 | 0 |
| Conv block 1 (32 filters) | 90 x 90 x 32 | 9,824 |
| Conv block 2 (64 filters) | 45 x 45 x 64 | 55,936 |
| Conv block 3 (128 filters) | 22 x 22 x 128 | 222,464 |
| Conv block 4 (256 filters) | 11 x 11 x 256 | 887,296 |
| Global Average Pooling | 256 | 0 |
| Dense, ReLU | 128 | 32,896 |
| Dropout (0.5) | 128 | 0 |
| Dense, Sigmoid (output) | 1 | 129 |
| **Total** | | **1,208,545 (1,206,625 trainable)** |

| Setting | Value |
|---------|-------|
| Loss | Binary cross-entropy |
| Optimizer | Adam, learning rate 0.001 |
| Batch size / epochs | 32 / 8 (limited by CPU training time, about 15 minutes per epoch) |
| Callbacks | EarlyStopping (patience 6, restore best weights), ReduceLROnPlateau (factor 0.3, patience 3), ModelCheckpoint (saves best model as `best_cnn.keras`) |
| Metrics tracked | Accuracy, AUC, precision, recall |

Global Average Pooling was used instead of Flatten because it needs far fewer parameters, which reduces overfitting on a small dataset.

### 2.6 Training Results and Evaluation

Training took about 2 hours for 8 epochs. Training accuracy rose steadily from 86.3% to 93.7%, but the validation results swung strongly from epoch to epoch. The best epoch was **epoch 7**, with validation loss 0.105 and **validation accuracy 96.1%**. Epoch 8 dropped back to 74.2%. The best model from epoch 7 was saved by ModelCheckpoint and used for testing.

| Metric (test set, 624 images) | Value |
|-------------------------------|-------|
| Accuracy | 71.63% |
| Precision (Pneumonia) | 0.689 |
| Recall (Pneumonia) | 0.997 |
| F1 score (Pneumonia) | 0.815 |
| ROC-AUC | 0.915 |
| Recall (Normal) | 0.25 |

**Confusion matrix (test set)**

| | Predicted Normal | Predicted Pneumonia |
|---|---|---|
| **Actual Normal** | 58 | 176 |
| **Actual Pneumonia** | 1 | 389 |

The model found 389 of the 390 pneumonia cases, missing only one. However, it also labelled 176 of the 234 normal X-rays as pneumonia, so only 58 normal images were classified correctly.

### 2.7 Conclusion

The CNN reached **71.6% test accuracy**, a ROC-AUC of **0.915** and **99.7% recall** for pneumonia. For a screening tool, missing only 1 sick patient is a good sign, since a missed case is far more dangerous than a false alarm. But labelling three quarters of healthy patients as sick would put too much extra work on doctors for the model to be used in practice.

There is a large gap between validation accuracy (96.1%) and test accuracy (71.6%). The validation images were split randomly from the training folder, and the dataset contains several X-rays from the same patients, so very similar images can appear in both training and validation, making the validation score too optimistic. The test folder contains different patients and is the more honest measure. The high ROC-AUC shows that the model still ranks pneumonia images above normal ones most of the time; the main problem is that on new patients its probabilities are pushed towards pneumonia, so at the 0.5 threshold most normal images are flagged.

Training was also limited by the hardware: on a laptop CPU only 8 epochs were possible, and the unstable validation curves show that training had not settled, likely because the Batch Normalization statistics had not stabilised and the learning rate of 0.001 was fairly high. Results could be improved by training for longer on a GPU with a lower learning rate, splitting training and validation by patient instead of by image, and using transfer learning from a pretrained network such as ResNet50 or EfficientNet.

---

## Overall Conclusion

This project applied two types of deep learning model to two real-world problems: a DNN on tabular customer data and a CNN on medical images. Both datasets were imbalanced, which made accuracy alone misleading, so precision, recall, F1, ROC-AUC and confusion matrices were used to judge the models properly. The churn DNN performs close to the known limit for its dataset. The pneumonia CNN catches almost every pneumonia case but produces too many false alarms on new patients, mainly because of the short CPU training and the validation split. Dropout, Batch Normalization, early stopping and checkpointing were used in both models to control overfitting and keep the best weights.

## Source Code

| File | Contents |
|------|----------|
| `churn_dnn.ipynb` | DNN for customer churn prediction (Part 1) |
| `pneumonia_cnn.ipynb` | CNN for pneumonia detection (Part 2) |
| `best_cnn.keras` | Trained CNN model (best epoch) |
| `requirements.txt` | Python libraries needed to run the notebooks |
