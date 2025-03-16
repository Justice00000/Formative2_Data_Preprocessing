# Formative2_Data_Preprocessing


## Video Explanation
[Expanation Video](https://drive.google.com/file/d/1wEO2u2RqSNT-EVMJWfWZ1izd3eH7HNul/view?usp=sharing)


## Part 1: Data Augmentation on CSV Files

### Steps Taken
1. **Transaction Data Preparation**
   - Converted date fields (purchase_date) to numerical components for processing
   - Normalized numerical features to [-1, 1] range using MinMaxScaler
   - Filled in missing values by prediction using KNNImputer
   - Added noise to numerical features to improve model generalization
   - Balanced product categories by oversampling minority classes using SMOTE

2. **Data Augmentation via GAN**
   - Trained **user-specific** GAN models to capture individual transaction patterns by itterating over each id
   - Generated one synthetic transaction per customer that preserves spending behavior
   - Assigned unique transaction IDs while maintaining user-product relationships
   - Applied proper date validation to ensure realistic timestamps of predicted transactions
   - Saved expanded dataset with original format and column structure

### Key Insights & Challenges
- **Challenges Solved**: 
  * Fixed GAN-generated negative values for purchase amounts and even date values using scaling
  * Resolved date validation issues (e.g., invalid dates like February 31st)
  * Implemented approach for users with limited transaction history
  * Preserved customer-specific purchasing patterns in synthetic data

- **Insights**:
  * User-specific GANs effectively captured individual spending patterns
  * The augmented dataset maintains the statistical properties of the original
  * Transaction patterns showed strong user-specific characteristics
  * The expanded dataset provides additional examples for improved model training

 ## Part 2: Merging Datasets with Transitive Properties

### Steps Taken
1. **Mapping Customer IDs**
   - Loaded the three datasets: `customer_transactions_augmented.csv`, `customer_social_profiles.csv`, and `id_mapping.csv`.
   - Used `id_mapping.csv` to link `customer_id_legacy` from transactions to `customer_id_new` from social profiles.
   - Performed a left merge to ensure all transactions were mapped properly.

2. **Handling Duplicate Mappings**
   - Since multiple `customer_id_legacy` values could map to the same `customer_id_new`, aggregated transactions per `customer_id_new`.
   - Applied `groupby('customer_id_new')` to sum total purchases and average engagement-related scores.

3. **Feature Engineering & Transformation**
   - **Moving Average of Transactions**
     - Computed a rolling mean of purchase amounts over the last three transactions per customer.
     - Used `.rolling(window=3, min_periods=1).mean()` to ensure at least one transaction was considered.
   
   - **Time-based Aggregation of Purchases**
     - Converted `purchase_date` to datetime format.
     - Extracted the monthly period (`.dt.to_period('M')`) and aggregated total spending per month.
     - Averaged monthly spending per customer and merged into the dataset.

   - **TF-IDF Application**
     - Although social media comments were not present in the dataset, simulated text-based feature extraction.
     - Applied TF-IDF on engagement-related text features such as `engagement_score`.
     - Used `TfidfVectorizer` from `sklearn.feature_extraction.text` to generate numerical representations.

   - **Customer Engagement Score Calculation**
     - Combined total spending, purchase interest score, and engagement score into a weighted formula:
       ```
       customer_engagement_score = (0.5 * purchase_amount) + (0.3 * purchase_interest_score) + (0.2 * engagement_score)
       ```

### Key Insights & Challenges
- **Challenges Solved**:
  * Ensured correct mappings between datasets using transitive relationships.
  * Managed customers with multiple transaction entries by applying aggregation.
  * Addressed missing values by averaging where necessary.
  * Implemented TF-IDF without explicit social media comment data by leveraging engagement-related features.

- **Insights**:
  * The linked dataset provided a more comprehensive customer profile.
  * Moving averages and time-based aggregations improved behavioral understanding.
  * Even without explicit text data, TF-IDF enhanced feature representation.
  * The final dataset allows for deeper analysis into customer spending and engagement patterns.



## Part 3: Data Consistency and Quality Checks

## Overview

In this part of the assignment, the goal was to ensure that the final preprocessed dataset is clean, structured, and ready for machine learning applications. This involved performing integrity checks, generating statistical summaries, visualizing distributions, and selecting important features to ensure the dataset's quality.

## Tasks Completed

### 1. Data Integrity Checks
- **Duplicate Entries**: Checked for any duplicate entries in the dataset and removed them to ensure consistency.
- **Categorical Values Mapping**: Validated that all categorical values are correctly mapped and that there are no incorrect or missing values.
- **Matching Transactions to Social Profiles**: Ensured that all customer transactions matched a valid social profile, addressing any missing or invalid relationships.

### 2. Statistical Summarization
- **Describe() Reports**: Generated summary statistics for all numerical columns in the dataset to understand the data distribution and detect any potential outliers or anomalies.
- **Transaction Amount Distribution Visualization**: Visualized the distribution of transaction amounts before and after data augmentation to assess how the augmentation techniques impacted the dataset.

### 3. Feature Selection for Machine Learning
- **Correlation Heatmap**: Created a correlation heatmap to identify highly correlated features and remove redundant data.
- **Top Features Selection**: Used a feature selection algorithm to select the top 10 most important features for the machine learning model.

### 4. Final Data Export
- After completing all the necessary preprocessing, the final dataset was saved as `final_dataset_ready_12.csv`.

## Key Insights

- **Data Quality**: Ensured that the dataset was free of duplicates and inconsistencies, allowing for reliable machine learning modeling.
- **Feature Engineering**: Successfully engineered new features such as the Customer Engagement Score and transactional features to enhance the predictive power of the dataset.
- **Data Augmentation Impact**: The augmentation techniques applied earlier improved the dataset's quality, balancing customer behavior data and providing better representation for modeling.

## Challenges Faced and Solutions

- **Handling Duplicates**: The dataset contained several duplicate entries, which were resolved by checking for duplicates and removing them programmatically.
- **Merging Data Across Multiple Sources**: Merging the customer transactions dataset with social media profiles required careful handling of missing and mismatched IDs. This was managed by ensuring that each customer transaction matched an entry in the social profiles dataset after resolving key discrepancies.
- **Feature Selection Complexity**: Identifying the most relevant features for the machine learning model was challenging due to the variety of features and the complex relationships between them. This was addressed by applying feature selection algorithms and focusing on the most influential features.

## Final Processed Files

- `final_customer_data_12.csv`: The final merged and feature-engineered dataset after completing all steps in Part 2.
- `final_dataset_ready_12.csv`: The cleaned, transformed, and ready-to-use dataset for machine learning.

## Conclusion

The data preprocessing steps in Part 3 ensured the final dataset was ready for machine learning modeling. By addressing issues such as duplicates, inconsistencies, and missing values, and by applying feature engineering techniques, we ensured that the dataset was robust for predictive modeling.



