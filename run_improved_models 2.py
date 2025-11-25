

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')



def run_improved_twfe(df):
   
    print("\n" + "="*80)
    print("IMPROVED PHASE 1: TWO-WAY FIXED EFFECTS")
    print("="*80)
    
    results = {}
    
    print("\n[Model 1] Basic TWFE (baseline)")
    formula1 = 'log_price ~ female_product + C(category_std) + C(brand_fe) + C(retailer)'
    model1 = smf.ols(formula=formula1, data=df).fit(cov_type='HC1')
    results['basic'] = model1
    print(f"Female coefficient: {model1.params['female_product']:.4f} (p={model1.pvalues['female_product']:.4f})")
    print(f"R²: {model1.rsquared:.4f}")
  
    print("\n[Model 2] TWFE + Size controls")
    formula2 = 'log_price ~ female_product + size_value_filled + size_missing + C(category_std) + C(brand_fe) + C(retailer)'
    model2 = smf.ols(formula=formula2, data=df).fit(cov_type='HC1')
    results['with_size'] = model2
    print(f"Female coefficient: {model2.params['female_product']:.4f} (p={model2.pvalues['female_product']:.4f})")
    print(f"R²: {model2.rsquared:.4f}")
    
    print("\n[Model 3] Brand × Category Fixed Effects (IMPROVED)")

    df['brand_category'] = df['brand_fe'].astype(str) + '_' + df['category_std'].astype(str)
   
    brand_cat_counts = df['brand_category'].value_counts()
    df['brand_category_fe'] = df['brand_category'].where(
        df['brand_category'].isin(brand_cat_counts[brand_cat_counts >= 5].index), 
        'other_brand_cat'
    )
    
    formula3 = 'log_price ~ female_product + size_value_filled + size_missing + premium_brand + C(brand_category_fe) + C(retailer)'
    model3 = smf.ols(formula=formula3, data=df).fit(cov_type='HC1')
    results['brand_category_fe'] = model3
    print(f"Female coefficient: {model3.params['female_product']:.4f} (p={model3.pvalues['female_product']:.4f})")
    print(f"R²: {model3.rsquared:.4f}")
    print(f"Unique brand-category combinations: {df['brand_category_fe'].nunique()}")
    
    print("\n[Model 4] Using log(price_per_unit) as DV")
    df_price_unit = df.dropna(subset=['log_price_per_unit'])
    formula4 = 'log_price_per_unit ~ female_product + premium_brand + C(category_std) + C(brand_fe) + C(retailer)'
    model4 = smf.ols(formula=formula4, data=df_price_unit).fit(cov_type='HC1')
    results['price_per_unit'] = model4
    print(f"Female coefficient: {model4.params['female_product']:.4f} (p={model4.pvalues['female_product']:.4f})")
    print(f"R²: {model4.rsquared:.4f}")
    print(f"N: {int(model4.nobs)}")
    
    print("\n✓ Improved TWFE models completed")
    return results



def run_inverse_probability_weighting(df):
    """IPW - Better than PSM for handling selection"""
    print("\n" + "="*80)
    print("IMPROVED PHASE 2: INVERSE PROBABILITY WEIGHTING (IPW)")
    print("="*80)
  
    df_ipw = df[df['gender_clean'].isin(['female', 'male'])].copy()
    df_ipw = df_ipw.dropna(subset=['log_price', 'size_value_filled', 'premium_brand', 
                                    'category_encoded', 'retailer_encoded'])
    
    print(f"\nSample size: {df_ipw.shape[0]}")
    print(f"  Female: {(df_ipw['gender_clean']=='female').sum()}")
    print(f"  Male: {(df_ipw['gender_clean']=='male').sum()}")
    

    print("\n[Step 1] Estimating propensity scores...")
    X_cols = ['size_value_filled', 'premium_brand', 'category_encoded', 'retailer_encoded']
    if 'has_premium_keyword' in df_ipw.columns:
        X_cols.append('has_premium_keyword')
    
    X = df_ipw[X_cols].values
    y = df_ipw['female_product'].values
    
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
 
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X_scaled, y)
    df_ipw['propensity_score'] = ps_model.predict_proba(X_scaled)[:, 1]
    
    print(f"✓ Propensity scores estimated")
    print(f"  Mean PS (female): {df_ipw[df_ipw['female_product']==1]['propensity_score'].mean():.4f}")
    print(f"  Mean PS (male): {df_ipw[df_ipw['female_product']==0]['propensity_score'].mean():.4f}")
    

    print("\n[Step 2] Calculating IPW weights...")
   
    df_ipw['ps_trimmed'] = df_ipw['propensity_score'].clip(0.1, 0.9)
    
    
    df_ipw['ipw'] = np.where(
        df_ipw['female_product'] == 1,
        1 / df_ipw['ps_trimmed'],
        1 / (1 - df_ipw['ps_trimmed'])
    )
    
   
    df_ipw['ipw_normalized'] = df_ipw['ipw'] * len(df_ipw) / df_ipw['ipw'].sum()
    
    print(f"✓ IPW weights calculated")
    print(f"  Mean weight (female): {df_ipw[df_ipw['female_product']==1]['ipw_normalized'].mean():.2f}")
    print(f"  Mean weight (male): {df_ipw[df_ipw['female_product']==0]['ipw_normalized'].mean():.2f}")
    
   
    print("\n[Step 3] Running weighted regression...")
    formula = 'log_price ~ female_product + size_value_filled + premium_brand + C(category_encoded) + C(retailer_encoded)'
    model_ipw = smf.wls(formula=formula, data=df_ipw, weights=df_ipw['ipw_normalized']).fit(cov_type='HC1')
    
    coef = model_ipw.params['female_product']
    se = model_ipw.bse['female_product']
    pval = model_ipw.pvalues['female_product']
    effect_pct = (np.exp(coef) - 1) * 100
    
    print(f"\nIPW Results:")
    print(f"  ATE: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
    print(f"  Effect: {effect_pct:.2f}%")
    print(f"  95% CI: [{model_ipw.conf_int().loc['female_product', 0]:.4f}, {model_ipw.conf_int().loc['female_product', 1]:.4f}]")
    
   
    print("\n[Balance Check] Weighted vs Unweighted means:")
    for var in X_cols:
        if var in df_ipw.columns:
           
            mean_female_unw = df_ipw[df_ipw['female_product']==1][var].mean()
            mean_male_unw = df_ipw[df_ipw['female_product']==0][var].mean()
            std_diff_unw = (mean_female_unw - mean_male_unw) / df_ipw[var].std() * 100
            
            
            mean_female_w = np.average(df_ipw[df_ipw['female_product']==1][var], 
                                      weights=df_ipw[df_ipw['female_product']==1]['ipw_normalized'])
            mean_male_w = np.average(df_ipw[df_ipw['female_product']==0][var], 
                                    weights=df_ipw[df_ipw['female_product']==0]['ipw_normalized'])
            std_diff_w = (mean_female_w - mean_male_w) / df_ipw[var].std() * 100
            
            print(f"  {var:20s}: {std_diff_unw:>6.1f}% → {std_diff_w:>6.1f}%")
    
    print("\n✓ IPW analysis completed")
    return {
        'model': model_ipw,
        'ate': coef,
        'df_ipw': df_ipw
    }



def run_within_brand_analysis(df):
    """Compare prices within brands that have both female and male products"""
    print("\n" + "="*80)
    print("NEW: WITHIN-BRAND ANALYSIS")
    print("="*80)
    
    
    brand_gender_counts = df.groupby('brand')['gender_clean'].apply(
        lambda x: set(x) & {'female', 'male'}
    )
    brands_with_both = brand_gender_counts[brand_gender_counts.apply(len) >= 2].index
    
    print(f"\nBrands with both female AND male products: {len(brands_with_both)}")
    
    if len(brands_with_both) < 10:
        print("⚠ Warning: Very few brands have both genders. Results may be unreliable.")
    
   
    df_within = df[df['brand'].isin(brands_with_both) & 
                   df['gender_clean'].isin(['female', 'male'])].copy()
    
    print(f"Within-brand sample size: {df_within.shape[0]}")
    print(f"  Female: {(df_within['gender_clean']=='female').sum()}")
    print(f"  Male: {(df_within['gender_clean']=='male').sum()}")
    
   
    print("\n[Model 1] Within-brand with brand FE")
    formula1 = 'log_price ~ female_product + size_value_filled + C(brand) + C(category_std) + C(retailer)'
    model1 = smf.ols(formula=formula1, data=df_within).fit(cov_type='HC1')
    
    print(f"Female coefficient: {model1.params['female_product']:.4f} (p={model1.pvalues['female_product']:.4f})")
    print(f"Effect: {(np.exp(model1.params['female_product'])-1)*100:.2f}%")
    print(f"R²: {model1.rsquared:.4f}")
    
    
    print("\n[Model 2] Within-brand × category")
    df_within['brand_cat'] = df_within['brand'].astype(str) + '_' + df_within['category_std'].astype(str)
   
    brand_cat_gender = df_within.groupby('brand_cat')['gender_clean'].nunique()
    brand_cat_both = brand_cat_gender[brand_cat_gender >= 2].index
    df_within_strict = df_within[df_within['brand_cat'].isin(brand_cat_both)]
    
    print(f"Strict within-brand-category sample: {df_within_strict.shape[0]}")
    
    if df_within_strict.shape[0] > 100:
        formula2 = 'log_price ~ female_product + size_value_filled + C(brand_cat) + C(retailer)'
        model2 = smf.ols(formula=formula2, data=df_within_strict).fit(cov_type='HC1')
        
        print(f"Female coefficient: {model2.params['female_product']:.4f} (p={model2.pvalues['female_product']:.4f})")
        print(f"Effect: {(np.exp(model2.params['female_product'])-1)*100:.2f}%")
        print(f"R²: {model2.rsquared:.4f}")
    else:
        print("⚠ Sample too small for brand × category FE")
        model2 = None
    
    print("\n✓ Within-brand analysis completed")
    return {
        'model_brand_fe': model1,
        'model_brand_cat_fe': model2,
        'brands_with_both': brands_with_both,
        'df_within': df_within
    }


def run_placebo_tests(df):
   
    print("\n" + "="*80)
    print("NEW: PLACEBO TESTS")
    print("="*80)
    
   
    print("\n[Placebo Test 1] Unisex vs Male products")
    df_placebo1 = df[df['gender_clean'].isin(['unisex', 'male'])].copy()
    df_placebo1['unisex_product'] = (df_placebo1['gender_clean'] == 'unisex').astype(int)
    
    formula_placebo1 = 'log_price ~ unisex_product + size_value_filled + premium_brand + C(category_std) + C(brand_fe) + C(retailer)'
    model_placebo1 = smf.ols(formula=formula_placebo1, data=df_placebo1).fit(cov_type='HC1')
    
    coef1 = model_placebo1.params['unisex_product']
    pval1 = model_placebo1.pvalues['unisex_product']
    print(f"Unisex coefficient: {coef1:.4f} (p={pval1:.4f})")
    print(f"Effect: {(np.exp(coef1)-1)*100:.2f}%")
    
    if pval1 < 0.05:
        print("⚠ WARNING: Significant effect on unisex-male comparison suggests issue with specification!")
    else:
        print("✓ Good: No significant difference (as expected)")
    
    print("\n[Placebo Test 2] Random 'female' assignment")
    df_placebo2 = df[df['gender_clean'].isin(['female', 'male'])].copy()
    np.random.seed(42)
    df_placebo2['random_female'] = np.random.binomial(1, 0.5, size=len(df_placebo2))
    
    formula_placebo2 = 'log_price ~ random_female + size_value_filled + premium_brand + C(category_std) + C(brand_fe) + C(retailer)'
    model_placebo2 = smf.ols(formula=formula_placebo2, data=df_placebo2).fit(cov_type='HC1')
    
    coef2 = model_placebo2.params['random_female']
    pval2 = model_placebo2.pvalues['random_female']
    print(f"Random 'female' coefficient: {coef2:.4f} (p={pval2:.4f})")
    print(f"Effect: {(np.exp(coef2)-1)*100:.2f}%")
    
    if pval2 < 0.10:
        print("⚠ WARNING: Random assignment shows effect - suggests confounding!")
    else:
        print("✓ Good: Random assignment shows no effect")
    
    print("\n✓ Placebo tests completed")
    return {
        'placebo_unisex_male': model_placebo1,
        'placebo_random': model_placebo2
    }



def run_subgroup_analysis(df):
    """Test pink tax separately for different subgroups"""
    print("\n" + "="*80)
    print("NEW: SUBGROUP ANALYSIS")
    print("="*80)
    
    results = {}
    
    
    print("\n[Subgroup 1] By Category")
    print("-" * 60)
    formula = 'log_price ~ female_product + size_value_filled + premium_brand + C(brand_fe) + C(retailer)'
    
    category_results = []
    for cat in df['category_std'].unique():
        if pd.isna(cat):
            continue
        df_cat = df[df['category_std'] == cat]
      
        n_female = (df_cat['gender_clean'] == 'female').sum()
        n_male = (df_cat['gender_clean'] == 'male').sum()
        
        if n_female >= 20 and n_male >= 20:
            try:
                model = smf.ols(formula, data=df_cat).fit(cov_type='HC1')
                coef = model.params['female_product']
                pval = model.pvalues['female_product']
                effect = (np.exp(coef) - 1) * 100
                sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
                
                print(f"{cat:20s}: {effect:>6.2f}% {sig:3s} (N={int(model.nobs):4d}, p={pval:.3f})")
                
                category_results.append({
                    'category': cat,
                    'coefficient': coef,
                    'effect_pct': effect,
                    'pvalue': pval,
                    'n_obs': int(model.nobs),
                    'n_female': n_female,
                    'n_male': n_male
                })
            except:
                print(f"{cat:20s}: Model failed to converge")
    
    results['by_category'] = pd.DataFrame(category_results)
 
    print("\n[Subgroup 2] By Retailer")
    print("-" * 60)
    formula_retailer = 'log_price ~ female_product + size_value_filled + premium_brand + C(category_std) + C(brand_fe)'
    
    for retailer in df['retailer'].unique():
        if pd.isna(retailer):
            continue
        df_ret = df[df['retailer'] == retailer]
        
        n_female = (df_ret['gender_clean'] == 'female').sum()
        n_male = (df_ret['gender_clean'] == 'male').sum()
        
        if n_female >= 50 and n_male >= 50:
            try:
                model = smf.ols(formula_retailer, data=df_ret).fit(cov_type='HC1')
                coef = model.params['female_product']
                pval = model.pvalues['female_product']
                effect = (np.exp(coef) - 1) * 100
                sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
                
                print(f"{retailer:20s}: {effect:>6.2f}% {sig:3s} (N={int(model.nobs):4d}, p={pval:.3f})")
            except:
                print(f"{retailer:20s}: Model failed to converge")
   
    print("\n[Subgroup 3] By Premium Brand Status")
    print("-" * 60)
    formula_simple = 'log_price ~ female_product + size_value_filled + C(category_std) + C(retailer)'
    
    for premium in [0, 1]:
        label = "Premium Brands" if premium == 1 else "Non-Premium Brands"
        df_prem = df[df['premium_brand'] == premium]
        
        n_female = (df_prem['gender_clean'] == 'female').sum()
        n_male = (df_prem['gender_clean'] == 'male').sum()
        
        if n_female >= 50 and n_male >= 50:
            try:
                model = smf.ols(formula_simple, data=df_prem).fit(cov_type='HC1')
                coef = model.params['female_product']
                pval = model.pvalues['female_product']
                effect = (np.exp(coef) - 1) * 100
                sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
                
                print(f"{label:20s}: {effect:>6.2f}% {sig:3s} (N={int(model.nobs):4d}, p={pval:.3f})")
            except:
                print(f"{label:20s}: Model failed to converge")
    
    print("\n✓ Subgroup analysis completed")
    return results



def run_robustness_checks(df):
    """Additional robustness checks"""
    print("\n" + "="*80)
    print("NEW: ROBUSTNESS CHECKS")
    print("="*80)
    
    results = {}

    print("\n[Robustness 1] Excluding price outliers (top/bottom 1%)")
    p1, p99 = df['standard_price'].quantile([0.01, 0.99])
    df_no_outliers = df[(df['standard_price'] >= p1) & (df['standard_price'] <= p99)]
    
    formula = 'log_price ~ female_product + size_value_filled + size_missing + premium_brand + C(category_std) + C(brand_fe) + C(retailer)'
    model_no_outliers = smf.ols(formula, data=df_no_outliers).fit(cov_type='HC1')
    
    coef = model_no_outliers.params['female_product']
    pval = model_no_outliers.pvalues['female_product']
    print(f"Without outliers: {coef:.4f} (p={pval:.4f}), Effect: {(np.exp(coef)-1)*100:.2f}%")
    print(f"N: {int(model_no_outliers.nobs)} (removed {len(df) - len(df_no_outliers)} obs)")
    results['no_outliers'] = model_no_outliers
 
    print("\n[Robustness 2] Only categories with both female AND male products")
    cat_gender_counts = df.groupby(['category_std', 'gender_clean']).size().unstack(fill_value=0)
    balanced_cats = cat_gender_counts[(cat_gender_counts['female'] >= 20) & 
                                      (cat_gender_counts['male'] >= 20)].index
    
    df_balanced = df[df['category_std'].isin(balanced_cats) & 
                     df['gender_clean'].isin(['female', 'male'])]
    
    model_balanced = smf.ols(formula, data=df_balanced).fit(cov_type='HC1')
    
    coef = model_balanced.params['female_product']
    pval = model_balanced.pvalues['female_product']
    print(f"Balanced categories: {coef:.4f} (p={pval:.4f}), Effect: {(np.exp(coef)-1)*100:.2f}%")
    print(f"Categories included: {len(balanced_cats)}")
    print(f"N: {int(model_balanced.nobs)}")
    results['balanced_categories'] = model_balanced

    print("\n[Robustness 3] Clustered standard errors (by brand)")
    model_clustered = smf.ols(formula, data=df).fit(
        cov_type='cluster', 
        cov_kwds={'groups': df['brand_fe']}
    )
    
    coef = model_clustered.params['female_product']
    se_clustered = model_clustered.bse['female_product']
    pval = model_clustered.pvalues['female_product']
    print(f"Clustered SE: {coef:.4f} (SE: {se_clustered:.4f}, p={pval:.4f})")
    results['clustered_se'] = model_clustered
    
    print("\n✓ Robustness checks completed")
    return results



def create_improved_summary(twfe_results, ipw_results, within_brand_results, 
                           placebo_results, robustness_results):
    """Comprehensive summary of all improved results"""
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY - IMPROVED ANALYSIS")
    print("="*80)
    
    print("\n" + "-"*80)
    print("MAIN RESULTS")
    print("-"*80)
    
    print("\nImproved TWFE Models:")
    for name, model in twfe_results.items():
        coef = model.params['female_product']
        pval = model.pvalues['female_product']
        effect = (np.exp(coef) - 1) * 100
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
        print(f"  {name:25s}: {effect:>6.2f}% {sig:3s} (coef={coef:.4f}, p={pval:.4f})")
    
    print("\nInverse Probability Weighting:")
    coef = ipw_results['ate']
    effect = (np.exp(coef) - 1) * 100
    pval = ipw_results['model'].pvalues['female_product']
    sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
    print(f"  IPW ATE: {effect:>6.2f}% {sig:3s} (coef={coef:.4f}, p={pval:.4f})")
    
    print("\nWithin-Brand Analysis:")
    coef = within_brand_results['model_brand_fe'].params['female_product']
    pval = within_brand_results['model_brand_fe'].pvalues['female_product']
    effect = (np.exp(coef) - 1) * 100
    sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
    print(f"  Within-brand FE: {effect:>6.2f}% {sig:3s} (coef={coef:.4f}, p={pval:.4f})")
    
    print("\n" + "-"*80)
    print("ROBUSTNESS & VALIDITY CHECKS")
    print("-"*80)
    
    print("\nPlacebo Tests:")
    coef1 = placebo_results['placebo_unisex_male'].params['unisex_product']
    pval1 = placebo_results['placebo_unisex_male'].pvalues['unisex_product']
    print(f"  Unisex vs Male: {(np.exp(coef1)-1)*100:>6.2f}% (p={pval1:.4f}) - Should be ≈0")
    
    coef2 = placebo_results['placebo_random'].params['random_female']
    pval2 = placebo_results['placebo_random'].pvalues['random_female']
    print(f"  Random assignment: {(np.exp(coef2)-1)*100:>6.2f}% (p={pval2:.4f}) - Should be ≈0")
    
    print("\nRobustness Checks:")
    for name, model in robustness_results.items():
        coef = model.params['female_product']
        pval = model.pvalues['female_product']
        effect = (np.exp(coef) - 1) * 100
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
        print(f"  {name:25s}: {effect:>6.2f}% {sig:3s} (p={pval:.4f})")
    
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    
    estimates = []
    estimates.append(twfe_results['with_size'].params['female_product'])
    estimates.append(twfe_results['brand_category_fe'].params['female_product'])
    estimates.append(ipw_results['ate'])
    estimates.append(within_brand_results['model_brand_fe'].params['female_product'])
    
    mean_effect = np.mean([(np.exp(e)-1)*100 for e in estimates])
    std_effect = np.std([(np.exp(e)-1)*100 for e in estimates])
    
    print(f"\nAverage pink tax across main methods: {mean_effect:.2f}%")
    print(f"Standard deviation: {std_effect:.2f}%")
    
    if std_effect > 2:
        print("\n⚠ HIGH VARIABILITY: Results are sensitive to specification")
        print("   → Evidence for pink tax is WEAK")
    else:
        print("\n✓ LOW VARIABILITY: Results are robust across methods")
    
 
    if placebo_results['placebo_unisex_male'].pvalues['unisex_product'] < 0.10:
        print("\n⚠ PLACEBO TEST FAILED: Finding effects where none should exist")
        print("   → Suggests specification issues or confounding")
    else:
        print("\n✓ PLACEBO TEST PASSED: No spurious effects detected")
    
    print("\n" + "="*80 + "\n")



def main():
   
    print("\n" + "="*80)
    print(" "*20 + "IMPROVED PINK TAX MODEL ESTIMATION")
    print(" "*25 + "Maria's Research Project")
    print("="*80)
    
    # Load data
    print("\nLoading prepared datasets...")
    df_main = pd.read_csv('pink_tax_main_analysis.csv')
    print(f"✓ Main dataset: {df_main.shape[0]} observations")
    
    try:
      
        print("\n" + "="*80)
        print("RUNNING IMPROVED ECONOMETRIC ANALYSIS")
        print("="*80)
        
        
        twfe_results = run_improved_twfe(df_main)
        
        
        ipw_results = run_inverse_probability_weighting(df_main)
        
       
        within_brand_results = run_within_brand_analysis(df_main)
        
        
        placebo_results = run_placebo_tests(df_main)
        
        
        subgroup_results = run_subgroup_analysis(df_main)
        
        
        robustness_results = run_robustness_checks(df_main)
        
        
        create_improved_summary(twfe_results, ipw_results, within_brand_results,
                               placebo_results, robustness_results)
        
        print("="*80)
        print("IMPROVED ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nAll models have been estimated with improved specifications.")
        print("Review the results above for your paper.")
        print("="*80 + "\n")
        
        return {
            'twfe': twfe_results,
            'ipw': ipw_results,
            'within_brand': within_brand_results,
            'placebo': placebo_results,
            'subgroup': subgroup_results,
            'robustness': robustness_results
        }
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
