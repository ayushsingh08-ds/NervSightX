So first step 
load data set 
I am using Dreaddit as my training data set 
now i will be performing Eda to understand the data

Exploratory Data Analysis (EDA):
Class balance (stress vs non-stress)
Text length analysis (tokens, characters)
Distribution of LIWC categories (e.g., negemo, anx, posemo)
Distribution of DAL features (pleasantness, activation, imagery)
Sentiment distribution
Syntax complexity distribution (syntax_ari, syntax_fk_grade)
Social metadata overview (social_karma, etc.)
Correlation heatmap of lexical/psycholinguistic features
Inspect sample posts from each class
Identify potential noise, outliers, low-confidence labels
Save key plots and statistics
refer EDA.ipynb , dreaddit_eda_outputs,dreaddit_eda_outputs/eda_report_summary.txt

After eda i created auxiliary text-based numeric features
(token_len)
(char_len)
checking missing value and calculated LIWC variance ranking then prepared full numeric feature matrix and finally performed 80/20 Stratified Train/Test Split
-ensured correct class balance in both sets.
-test set is frozen (untouched, unscaled)
Saved as
train_raw.csv
test_frozen_raw.csv
included orig_index to trace rows back to the original dataset.

