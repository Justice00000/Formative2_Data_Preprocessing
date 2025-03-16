# Formative2_Data_Preprocessing
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