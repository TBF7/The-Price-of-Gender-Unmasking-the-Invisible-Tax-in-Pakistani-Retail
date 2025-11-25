

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
   
    print("=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)
    df = pd.read_excel(filepath)
    print(f"✓ Loaded dataset with {df.shape[0]} observations and {df.shape[1]} columns")
    return df

def clean_gender_target(df):
    
    print("\n" + "=" * 80)
    print("STEP 2: CLEANING GENDER TARGET VARIABLE")
    print("=" * 80)
    
    print("\nOriginal gender_target distribution:")
    print(df['gender_target'].value_counts())
    
    # Create mapping for standardization
    gender_mapping = {
        'female': 'female',
        'FEMALE': 'female',
        'women': 'female',
        'femlae': 'female',  # typo correction
        'male': 'male',
        'men': 'male',
        'unisex': 'unisex',
        'UNISEX': 'unisex'
    }
    
    df['gender_clean'] = df['gender_target'].map(gender_mapping)
    
    print("\nCleaned gender distribution:")
    print(df['gender_clean'].value_counts())
    print(f"\nMissing gender values: {df['gender_clean'].isnull().sum()}")
    
    return df

def handle_missing_values(df):
    
    print("\n" + "=" * 80)
    print("STEP 3: HANDLING MISSING VALUES")
    print("=" * 80)
    
    print("\nMissing values before cleaning:")
    missing_summary = df.isnull().sum()
    print(missing_summary[missing_summary > 0])
    
 
    original_count = df.shape[0]
    
   
    df_clean = df.dropna(subset=['standard_price', 'gender_clean'])
    
    print(f"\n✓ Dropped rows with missing price or gender (critical variables only)")
    print(f"  Observations remaining: {df_clean.shape[0]} (removed {original_count - df_clean.shape[0]})")
    
 
    if df_clean['category_std'].isnull().any():
        mode_category = df_clean['category_std'].mode()[0]
        df_clean['category_std'].fillna(mode_category, inplace=True)
        print(f"✓ Filled missing categories with: {mode_category}")
    
   
    df_clean['brand'].fillna('unknown_brand', inplace=True)
    print(f"✓ Filled missing brands with: 'unknown_brand'")
    
   
    if df_clean['retailer'].isnull().any():
        mode_retailer = df_clean['retailer'].mode()[0]
        df_clean['retailer'].fillna(mode_retailer, inplace=True)
        print(f"✓ Filled missing retailer with: {mode_retailer}")
    
   
    df_clean['size_missing'] = df_clean['size_value_std'].isnull().astype(int)
    
   
    df_clean['size_value_filled'] = df_clean.groupby(['category_std', 'gender_clean'])['size_value_std'].transform(
        lambda x: x.fillna(x.median())
    )
    
    
    df_clean['size_value_filled'] = df_clean.groupby('category_std')['size_value_filled'].transform(
        lambda x: x.fillna(x.median())
    )
    
    
    overall_median_size = df_clean['size_value_std'].median()
    df_clean['size_value_filled'].fillna(overall_median_size, inplace=True)
    
    print(f"✓ Filled missing size values using category-gender medians")
    print(f"  Created 'size_missing' indicator for control")
    
    
    if 'observed_at' in df_clean.columns and df_clean['observed_at'].isnull().any():
        mode_date = df_clean['observed_at'].mode()[0] if not df_clean['observed_at'].mode().empty else df_clean['observed_at'].dropna().iloc[0]
        df_clean['observed_at'].fillna(mode_date, inplace=True)
        print(f"✓ Filled missing dates")
    
    print(f"\nFinal missing values check:")
    final_missing = df_clean.isnull().sum()
    print(final_missing[final_missing > 0] if final_missing.sum() > 0 else "✓ No missing values in critical variables!")
    
    return df_clean

def create_target_variable(df):
    
    print("\n" + "=" * 80)
    print("STEP 4: CREATING TARGET VARIABLE")
    print("=" * 80)
    
   
    df['female_product'] = (df['gender_clean'] == 'female').astype(int)
    df['male_product'] = (df['gender_clean'] == 'male').astype(int)
    df['unisex_product'] = (df['gender_clean'] == 'unisex').astype(int)
    
    print("\nTreatment variable distribution:")
    print(f"Female products (treatment): {df['female_product'].sum()}")
    print(f"Male products (control): {df['male_product'].sum()}")
    print(f"Unisex products: {df['unisex_product'].sum()}")
    
    return df

def create_price_variables(df):
    
    print("\n" + "=" * 80)
    print("STEP 5: CREATING PRICE VARIABLES")
    print("=" * 80)
    
    
    df['log_price'] = np.log(df['standard_price'])
    
    
    df['price_per_unit'] = df['standard_price'] / df['size_value_filled']
    df['log_price_per_unit'] = np.log(df['price_per_unit'].replace([np.inf, -np.inf], np.nan))
    
    
    df['price_per_unit'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['log_price_per_unit'].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    
    df['price_per_unit'].fillna(df['standard_price'], inplace=True)
    df['log_price_per_unit'].fillna(df['log_price'], inplace=True)
    
    print("✓ Created log_price, price_per_unit, and log_price_per_unit")
    print(f"\nPrice variable statistics:")
    print(df[['standard_price', 'log_price', 'price_per_unit']].describe())
    
    return df

def create_control_variables(df):
    
    print("\n" + "=" * 80)
    print("STEP 6: CREATING CONTROL VARIABLES")
    print("=" * 80)
    
   
    brand_counts = df['brand'].value_counts()
    print(f"\nTotal unique brands: {len(brand_counts)}")
    
   
    min_brand_obs = 5
    frequent_brands = brand_counts[brand_counts >= min_brand_obs].index
    df['brand_fe'] = df['brand'].where(df['brand'].isin(frequent_brands), 'other_brand')
    print(f"Brands with ≥{min_brand_obs} observations: {len(frequent_brands)}")
    
    
    print(f"\nRetailers: {df['retailer'].unique()}")
    
   
    df['category_size'] = df['category_std'].astype(str) + '_' + df['size_value_filled'].astype(str)
    
    
    brand_avg_price = df.groupby('brand')['standard_price'].mean()
    premium_threshold = brand_avg_price.quantile(0.80)
    premium_brands = brand_avg_price[brand_avg_price >= premium_threshold].index
    df['premium_brand'] = df['brand'].isin(premium_brands).astype(int)
    print(f"✓ Created premium_brand indicator (top 20%)")
    
   
    if 'observed_at' in df.columns:
        df['observed_date'] = pd.to_datetime(df['observed_at'], errors='coerce')
        df['year'] = df['observed_date'].dt.year
        df['month'] = df['observed_date'].dt.month
        df['quarter'] = df['observed_date'].dt.quarter
        print(f"✓ Created time variables: year, month, quarter")
    
    return df

def create_categorical_encodings(df):
    """Create dummy variables for categorical controls."""
    print("\n" + "=" * 80)
    print("STEP 7: CREATING CATEGORICAL ENCODINGS")
    print("=" * 80)
    
   
    le_category = LabelEncoder()
    le_brand = LabelEncoder()
    le_retailer = LabelEncoder()
    
   
    df['category_encoded'] = le_category.fit_transform(df['category_std'].astype(str))
    df['brand_encoded'] = le_brand.fit_transform(df['brand_fe'].astype(str))
    df['retailer_encoded'] = le_retailer.fit_transform(df['retailer'].astype(str))
    
    print("✓ Created label-encoded variables for sklearn models")
    
    
    df.attrs['category_encoder'] = le_category
    df.attrs['brand_encoder'] = le_brand
    df.attrs['retailer_encoder'] = le_retailer
    
    print(f"\nCategories encoded: {le_category.classes_}")
    print(f"Retailers encoded: {le_retailer.classes_}")
    
    return df

def create_hedonic_features(df):
   
    print("\n" + "=" * 80)
    print("STEP 8: CREATING HEDONIC FEATURES")
    print("=" * 80)
    
    
    if 'title' in df.columns:
       
        premium_keywords = ['anti-aging', 'whitening', 'luxury', 'professional', 
                           'dermatologist', 'organic', 'natural', 'premium', 
                           'advanced', 'clinical', 'spa']
        
        df['title_lower'] = df['title'].str.lower()
        df['has_premium_keyword'] = df['title_lower'].apply(
            lambda x: any(keyword in str(x) for keyword in premium_keywords)
        ).astype(int)
        
        
        df['title_length'] = df['title'].str.len()
        
        print(f"✓ Created premium keyword indicator")
        print(f"  Products with premium keywords: {df['has_premium_keyword'].sum()}")
    
   
    df['size_standardized'] = (df['size_value_filled'] - df['size_value_filled'].mean()) / df['size_value_filled'].std()
    
   
    df['size_category'] = pd.qcut(df['size_value_filled'], q=4, labels=['small', 'medium', 'large', 'xlarge'], duplicates='drop')
    
    print(f"✓ Created size_standardized and size_category")
    
    return df

def create_matching_variables(df):
   
    print("\n" + "=" * 80)
    print("STEP 9: PREPARING PROPENSITY SCORE MATCHING VARIABLES")
    print("=" * 80)
    
   
    df_psm = df[df['gender_clean'].isin(['female', 'male'])].copy()
    
    print(f"✓ PSM sample: {df_psm.shape[0]} observations (female + male only)")
    print(f"  Female: {(df_psm['gender_clean'] == 'female').sum()}")
    print(f"  Male: {(df_psm['gender_clean'] == 'male').sum()}")
    
   
    matching_vars = ['size_value_filled', 'category_encoded', 'brand_encoded', 
                     'retailer_encoded', 'premium_brand']
    
    if 'has_premium_keyword' in df_psm.columns:
        matching_vars.append('has_premium_keyword')
    
    print(f"\n✓ Matching covariates: {matching_vars}")
    
    return df_psm, matching_vars

def create_quantile_variables(df):
    
    print("\n" + "=" * 80)
    print("STEP 10: PREPARING QUANTILE REGRESSION VARIABLES")
    print("=" * 80)
    
 
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    
   
    for q in quantiles:
        threshold = df['standard_price'].quantile(q)
        df[f'price_q{int(q*100)}'] = (df['standard_price'] <= threshold).astype(int)
    
    print(f"✓ Created price quantile indicators for: {[int(q*100) for q in quantiles]}th percentiles")
    
    return df, quantiles

def create_decomposition_samples(df):
   
    print("\n" + "=" * 80)
    print("STEP 11: PREPARING OAXACA-BLINDER DECOMPOSITION SAMPLES")
    print("=" * 80)
    
    
    df_female = df[df['gender_clean'] == 'female'].copy()
    df_male = df[df['gender_clean'] == 'male'].copy()
    
    print(f"✓ Female sample: {df_female.shape[0]} observations")
    print(f"✓ Male sample: {df_male.shape[0]} observations")
    
  
    common_categories = set(df_female['category_std'].unique()) & set(df_male['category_std'].unique())
    
    df_female_matched = df_female[df_female['category_std'].isin(common_categories)]
    df_male_matched = df_male[df_male['category_std'].isin(common_categories)]
    
    print(f"\n✓ Matched samples (common categories only):")
    print(f"  Female: {df_female_matched.shape[0]} observations")
    print(f"  Male: {df_male_matched.shape[0]} observations")
    print(f"  Common categories: {len(common_categories)}")
    print(f"  Categories: {sorted(common_categories)}")
    
    return df_female, df_male, df_female_matched, df_male_matched

def generate_summary_statistics(df):
   
    print("\n" + "=" * 80)
    print("STEP 12: GENERATING SUMMARY STATISTICS")
    print("=" * 80)
    
 
    summary_vars = ['standard_price', 'log_price', 'size_value_filled', 'price_per_unit']
    
    print("\nOverall Summary Statistics:")
    print(df[summary_vars].describe())
    
   
    print("\n\nSummary by Gender:")
    summary_by_gender = df.groupby('gender_clean')[summary_vars].agg(['mean', 'median', 'std', 'count'])
    print(summary_by_gender)
    
   
    print("\n\nPrice Comparison (Female vs Male):")
    female_mean = df[df['gender_clean'] == 'female']['standard_price'].mean()
    male_mean = df[df['gender_clean'] == 'male']['standard_price'].mean()
    raw_gap = female_mean - male_mean
    pct_gap = (raw_gap / male_mean) * 100
    
    print(f"Female average price: PKR {female_mean:.2f}")
    print(f"Male average price: PKR {male_mean:.2f}")
    print(f"Raw price gap: PKR {raw_gap:.2f}")
    print(f"Percentage gap: {pct_gap:.2f}%")
    
  
    print("\n\nPrice by Category and Gender:")
    category_summary = df.groupby(['category_std', 'gender_clean'])['standard_price'].agg(['mean', 'count']).round(2)
    print(category_summary)
    
    return summary_by_gender, category_summary

def save_prepared_data(df, df_psm, df_female, df_male, df_female_matched, df_male_matched):
   
    print("\n" + "=" * 80)
    print("STEP 13: SAVING PREPARED DATASETS")
    print("=" * 80)
    
    output_files = {
        'pink_tax_main_analysis.csv': df,
        'pink_tax_psm_sample.csv': df_psm,
        'pink_tax_female_sample.csv': df_female,
        'pink_tax_male_sample.csv': df_male,
        'pink_tax_female_matched.csv': df_female_matched,
        'pink_tax_male_matched.csv': df_male_matched
    }
    
    for filename, data in output_files.items():
        filepath = filename  
        data.to_csv(filepath, index=False)
        print(f"✓ Saved: {filename} ({data.shape[0]} obs, {data.shape[1]} vars)")
    
 
    with open('variable_list.txt', 'w') as f:
        f.write("PINK TAX ANALYSIS - VARIABLE LIST\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("DEPENDENT VARIABLES:\n")
        f.write("-" * 40 + "\n")
        f.write("- standard_price: Standardized price in PKR\n")
        f.write("- log_price: Natural log of price\n")
        f.write("- price_per_unit: Price per unit of size\n")
        f.write("- log_price_per_unit: Natural log of price per unit\n\n")
        
        f.write("TREATMENT VARIABLES:\n")
        f.write("-" * 40 + "\n")
        f.write("- female_product: Binary (1 = female product, 0 = otherwise)\n")
        f.write("- male_product: Binary (1 = male product, 0 = otherwise)\n")
        f.write("- gender_clean: Categorical (female, male, unisex)\n\n")
        
        f.write("CONTROL VARIABLES:\n")
        f.write("-" * 40 + "\n")
        f.write("- category_std: Product category (standardized)\n")
        f.write("- brand_fe: Brand (frequent brands only, others grouped)\n")
        f.write("- retailer: Retail store\n")
        f.write("- size_value_filled: Product size (filled for missing)\n")
        f.write("- size_missing: Binary indicator for originally missing size\n")
        f.write("- premium_brand: Binary (1 = top 20% price brands)\n")
        f.write("- size_standardized: Z-score of size\n")
        f.write("- size_category: Categorical size (small/medium/large/xlarge)\n\n")
        
        if 'has_premium_keyword' in df.columns:
            f.write("- has_premium_keyword: Binary (premium keywords in title)\n")
        
        f.write("\nENCODED VARIABLES (for sklearn models):\n")
        f.write("-" * 40 + "\n")
        f.write("- category_encoded: Numeric encoding of categories\n")
        f.write("- brand_encoded: Numeric encoding of brands\n")
        f.write("- retailer_encoded: Numeric encoding of retailers\n\n")
        
        f.write("DATASETS:\n")
        f.write("-" * 40 + "\n")
        f.write("1. pink_tax_main_analysis.csv - Full dataset with all variables\n")
        f.write("2. pink_tax_psm_sample.csv - Female + Male only (for PSM)\n")
        f.write("3. pink_tax_female_sample.csv - Female products only (for Oaxaca-Blinder)\n")
        f.write("4. pink_tax_male_sample.csv - Male products only (for Oaxaca-Blinder)\n")
        f.write("5. pink_tax_female_matched.csv - Female in common categories\n")
        f.write("6. pink_tax_male_matched.csv - Male in common categories\n")
    
    print("\n✓ Saved variable documentation: variable_list.txt")
    
    return output_files

def main():
   
    print("\n" + "=" * 80)
    print(" " * 20 + "PINK TAX DATA PREPARATION")
    print(" " * 25 + "Maria's Research Project")
    print("=" * 80)
    
 
    filepath = 'master_dataset_final_models.xlsx'
    
    
    df = load_data(filepath)
    df = clean_gender_target(df)
    df = handle_missing_values(df)
    df = create_target_variable(df)
    df = create_price_variables(df)
    df = create_control_variables(df)
    df = create_categorical_encodings(df)
    df = create_hedonic_features(df)
    df_psm, matching_vars = create_matching_variables(df)
    df, quantiles = create_quantile_variables(df)
    df_female, df_male, df_female_matched, df_male_matched = create_decomposition_samples(df)
    summary_by_gender, category_summary = generate_summary_statistics(df)
    
   
    output_files = save_prepared_data(df, df_psm, df_female, df_male, 
                                      df_female_matched, df_male_matched)
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print("\nNext Steps:")
    print("1. Review the variable_list.txt for all created variables")
    print("2. Check summary statistics above")
    print("3. Proceed with model estimation using prepared datasets")
    print("\nReady for:")
    print("  ✓ Two-Way Fixed Effects")
    print("  ✓ Hedonic Regression")
    print("  ✓ Propensity Score Matching")
    print("  ✓ Quantile Regression")
    print("  ✓ Oaxaca-Blinder Decomposition")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()