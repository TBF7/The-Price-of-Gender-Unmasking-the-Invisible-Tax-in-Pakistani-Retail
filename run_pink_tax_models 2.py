

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')



def run_twfe_models(df):
  
    print("\n" + "="*80)
    print("PHASE 1A: TWO-WAY FIXED EFFECTS (TWFE)")
    print("="*80)
    
    results = {}
    
   
    print("\n[Model 1] Basic TWFE")
    formula1 = 'log_price ~ female_product + C(category_std) + C(brand_fe)'
    print(f"Equation: {formula1}")
    model1 = smf.ols(formula=formula1, data=df).fit(cov_type='HC1')
    results['twfe_basic'] = model1
    print("\n" + str(model1.summary()))
    print(f"\n>>> KEY RESULT: Female coefficient = {model1.params['female_product']:.4f} ({(np.exp(model1.params['female_product'])-1)*100:.2f}% effect)")
    
   
    print("\n[Model 2] TWFE with retailer")
    formula2 = 'log_price ~ female_product + C(category_std) + C(brand_fe) + C(retailer)'
    print(f"Equation: {formula2}")
    model2 = smf.ols(formula=formula2, data=df).fit(cov_type='HC1')
    results['twfe_retailer'] = model2
    print("\n" + str(model2.summary()))
    print(f"\n>>> KEY RESULT: Female coefficient = {model2.params['female_product']:.4f} ({(np.exp(model2.params['female_product'])-1)*100:.2f}% effect)")
    
   
    print("\n[Model 3] TWFE with size controls")
    formula3 = 'log_price ~ female_product + size_value_filled + size_missing + C(category_std) + C(brand_fe) + C(retailer)'
    print(f"Equation: {formula3}")
    model3 = smf.ols(formula=formula3, data=df).fit(cov_type='HC1')
    results['twfe_size'] = model3
    print("\n" + str(model3.summary()))
    print(f"\n>>> KEY RESULT: Female coefficient = {model3.params['female_product']:.4f} ({(np.exp(model3.params['female_product'])-1)*100:.2f}% effect)")
    
   
    print("\n[Model 4] Full TWFE")
    formula4 = 'log_price ~ female_product + size_value_filled + size_missing + premium_brand + C(category_std) + C(brand_fe) + C(retailer)'
    print(f"Equation: {formula4}")
    model4 = smf.ols(formula=formula4, data=df).fit(cov_type='HC1')
    results['twfe_full'] = model4
    print("\n" + str(model4.summary()))
    print(f"\n>>> KEY RESULT: Female coefficient = {model4.params['female_product']:.4f} ({(np.exp(model4.params['female_product'])-1)*100:.2f}% effect)")
    
    print("\n✓ Two-Way Fixed Effects models completed")
    return results

def run_hedonic_regression(df):
  
    print("\n" + "="*80)
    print("PHASE 1B: HEDONIC REGRESSION")
    print("="*80)
    
    results = {}
    
   
    print("\n[Model 1] Basic hedonic")
    formula1 = 'log_price ~ female_product + size_value_filled + premium_brand'
    print(f"Equation: {formula1}")
    model1 = smf.ols(formula=formula1, data=df).fit(cov_type='HC1')
    results['hedonic_basic'] = model1
    print("\n" + str(model1.summary()))
    print(f"\n>>> KEY RESULT: Female coefficient = {model1.params['female_product']:.4f} ({(np.exp(model1.params['female_product'])-1)*100:.2f}% effect)")
    
   
    print("\n[Model 2] With category controls")
    formula2 = 'log_price ~ female_product + size_value_filled + premium_brand + C(category_std)'
    print(f"Equation: {formula2}")
    model2 = smf.ols(formula=formula2, data=df).fit(cov_type='HC1')
    results['hedonic_category'] = model2
    print("\n" + str(model2.summary()))
    print(f"\n>>> KEY RESULT: Female coefficient = {model2.params['female_product']:.4f} ({(np.exp(model2.params['female_product'])-1)*100:.2f}% effect)")
    

    print("\n[Model 3] With retailer controls")
    formula3 = 'log_price ~ female_product + size_value_filled + premium_brand + C(category_std) + C(retailer)'
    print(f"Equation: {formula3}")
    model3 = smf.ols(formula=formula3, data=df).fit(cov_type='HC1')
    results['hedonic_retailer'] = model3
    print("\n" + str(model3.summary()))
    print(f"\n>>> KEY RESULT: Female coefficient = {model3.params['female_product']:.4f} ({(np.exp(model3.params['female_product'])-1)*100:.2f}% effect)")
    
    if 'has_premium_keyword' in df.columns:
        print("\n[Model 4] Full hedonic")
        formula4 = 'log_price ~ female_product + size_value_filled + size_missing + premium_brand + has_premium_keyword + C(category_std) + C(retailer)'
        print(f"Equation: {formula4}")
        model4 = smf.ols(formula=formula4, data=df).fit(cov_type='HC1')
        results['hedonic_full'] = model4
        print("\n" + str(model4.summary()))
        print(f"\n>>> KEY RESULT: Female coefficient = {model4.params['female_product']:.4f} ({(np.exp(model4.params['female_product'])-1)*100:.2f}% effect)")
    
    print("\n✓ Hedonic regression models completed")
    return results



def run_propensity_score_matching(df_psm):
   
    print("\n" + "="*80)
    print("PHASE 2A: PROPENSITY SCORE MATCHING")
    print("="*80)
    
    
    df_psm = df_psm.dropna(subset=['log_price', 'female_product', 'size_value_filled', 
                                    'category_encoded', 'retailer_encoded', 'premium_brand'])
    
    print(f"\nSample size: {df_psm.shape[0]}")
    print(f"  Treatment (female): {df_psm['female_product'].sum()}")
    print(f"  Control (male): {(1-df_psm['female_product']).sum()}")
    
   
    print("\n[Step 1] Estimating propensity scores...")
    covariates = ['size_value_filled', 'category_encoded', 'retailer_encoded', 'premium_brand']
    if 'has_premium_keyword' in df_psm.columns:
        covariates.append('has_premium_keyword')
    
    X_ps = df_psm[covariates]
    y_treatment = df_psm['female_product']
    
   
    X_ps_const = sm.add_constant(X_ps)
    ps_model = sm.Logit(y_treatment, X_ps_const).fit(disp=0)
    df_psm['propensity_score'] = ps_model.predict(X_ps_const)
    
    print(f"✓ Propensity scores estimated")
    print(f"  Mean PS (treated): {df_psm[df_psm['female_product']==1]['propensity_score'].mean():.4f}")
    print(f"  Mean PS (control): {df_psm[df_psm['female_product']==0]['propensity_score'].mean():.4f}")
    
    
    print("\n[Step 2] Performing nearest neighbor matching (1:1)...")
    treated = df_psm[df_psm['female_product'] == 1].copy()
    control = df_psm[df_psm['female_product'] == 0].copy()
    
    
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(control[['propensity_score']].values)
    
    distances, indices = nn.kneighbors(treated[['propensity_score']].values)
    
   
    matched_control_indices = control.iloc[indices.flatten()].index
    matched_treated_indices = treated.index
    
    df_matched = pd.concat([
        df_psm.loc[matched_treated_indices],
        df_psm.loc[matched_control_indices]
    ])
    
    print(f"✓ Matching completed")
    print(f"  Matched sample size: {df_matched.shape[0]}")
    print(f"  Treated: {df_matched['female_product'].sum()}")
    print(f"  Control: {(1-df_matched['female_product']).sum()}")
    
    print("\n[Step 3] Estimating Average Treatment Effect on Treated (ATT)...")
    
    
    att_simple = (df_matched[df_matched['female_product']==1]['log_price'].mean() - 
                  df_matched[df_matched['female_product']==0]['log_price'].mean())
    
    print(f"\nSimple ATT (log price difference): {att_simple:.4f}")
    print(f"Percentage effect: {(np.exp(att_simple)-1)*100:.2f}%")
    
    
    formula_matched = 'log_price ~ female_product + size_value_filled + premium_brand + C(category_encoded) + C(retailer_encoded)'
    model_matched = smf.ols(formula=formula_matched, data=df_matched).fit(cov_type='HC1')
    
    print(f"\nRegression ATT: {model_matched.params['female_product']:.4f}")
    print(f"Standard error: {model_matched.bse['female_product']:.4f}")
    print(f"P-value: {model_matched.pvalues['female_product']:.4f}")
    print(f"95% CI: [{model_matched.conf_int().loc['female_product', 0]:.4f}, {model_matched.conf_int().loc['female_product', 1]:.4f}]")
    
   
    print("\n[Balance Check] Comparing covariates after matching:")
    for var in covariates:
        if var in df_matched.columns:
            mean_treated = df_matched[df_matched['female_product']==1][var].mean()
            mean_control = df_matched[df_matched['female_product']==0][var].mean()
            std_diff = (mean_treated - mean_control) / df_psm[var].std() * 100
            print(f"  {var}: Std diff = {std_diff:.2f}%")
    
    print("\n✓ Propensity Score Matching completed")
    
    return {
        'att_simple': att_simple,
        'model_matched': model_matched,
        'df_matched': df_matched,
        'ps_model': ps_model
    }

def run_quantile_regression(df, quantiles=[0.10, 0.25, 0.50, 0.75, 0.90]):
   
    print("\n" + "="*80)
    print("PHASE 2B: QUANTILE REGRESSION")
    print("="*80)
    
    results = {}
   
    df_qr = df.dropna(subset=['log_price', 'female_product', 'size_value_filled', 
                               'category_encoded', 'retailer_encoded'])
    
    print(f"\nSample size: {df_qr.shape[0]}")
    print(f"Estimating at quantiles: {[int(q*100) for q in quantiles]}")
    
   
    covariates = ['female_product', 'size_value_filled', 'premium_brand', 
                  'category_encoded', 'retailer_encoded']
    X = sm.add_constant(df_qr[covariates])
    y = df_qr['log_price']
    
    print("\n" + "-"*60)
    print("Quantile | Coef (female) | Std Err | P-value | Effect %")
    print("-"*60)
    
    for q in quantiles:
        qr_model = QuantReg(y, X).fit(q=q)
        results[f'q{int(q*100)}'] = qr_model
        
        coef = qr_model.params['female_product']
        se = qr_model.bse['female_product']
        pval = qr_model.pvalues['female_product']
        effect_pct = (np.exp(coef) - 1) * 100
        
        print(f"{int(q*100):>8} | {coef:>13.4f} | {se:>7.4f} | {pval:>7.4f} | {effect_pct:>7.2f}%")
    
    print("-"*60)
    print("\n✓ Quantile regression completed")
    print("Interpretation: Pink tax varies across price distribution")
    
    return results



def run_oaxaca_blinder(df_female, df_male):
    
    print("\n" + "="*80)
    print("PHASE 3: OAXACA-BLINDER DECOMPOSITION")
    print("="*80)
    
   
    df_female_clean = df_female.dropna(subset=['log_price', 'size_value_filled', 
                                                'premium_brand', 'category_encoded'])
    df_male_clean = df_male.dropna(subset=['log_price', 'size_value_filled', 
                                            'premium_brand', 'category_encoded'])
    
    print(f"\nSample sizes:")
    print(f"  Female: {df_female_clean.shape[0]}")
    print(f"  Male: {df_male_clean.shape[0]}")
    
   
    covariates = ['size_value_filled', 'premium_brand', 'category_encoded', 'retailer_encoded']
    
    
    print("\n[Step 1] Running separate regressions by gender...")
    
    X_female = sm.add_constant(df_female_clean[covariates])
    y_female = df_female_clean['log_price']
    model_female = sm.OLS(y_female, X_female).fit(cov_type='HC1')
    
    X_male = sm.add_constant(df_male_clean[covariates])
    y_male = df_male_clean['log_price']
    model_male = sm.OLS(y_male, X_male).fit(cov_type='HC1')
    
    print(f"✓ Female model R²: {model_female.rsquared:.4f}")
    print(f"✓ Male model R²: {model_male.rsquared:.4f}")
    
    
    print("\n[Step 2] Computing mean characteristics...")
    X_female_mean = df_female_clean[covariates].mean()
    X_male_mean = df_male_clean[covariates].mean()
    
    
    print("\n[Step 3] Performing Oaxaca-Blinder decomposition...")
    
    
    y_female_mean = y_female.mean()
    y_male_mean = y_male.mean()
    total_gap = y_female_mean - y_male_mean
    
    print(f"\nMean log prices:")
    print(f"  Female: {y_female_mean:.4f}")
    print(f"  Male: {y_male_mean:.4f}")
    print(f"  Total gap: {total_gap:.4f}")
    print(f"  Total gap (%): {(np.exp(total_gap)-1)*100:.2f}%")
    
    
    explained = ((X_female_mean - X_male_mean) * model_male.params[1:]).sum()
    
  
    intercept_diff = model_female.params['const'] - model_male.params['const']
    coef_diff = ((model_female.params[1:] - model_male.params[1:]) * X_female_mean).sum()
    unexplained = intercept_diff + coef_diff
    
    print(f"\n{'='*60}")
    print("DECOMPOSITION RESULTS")
    print(f"{'='*60}")
    print(f"Total gap:        {total_gap:.4f} ({(np.exp(total_gap)-1)*100:.2f}%)")
    print(f"Explained:        {explained:.4f} ({(np.exp(explained)-1)*100:.2f}%) [{explained/total_gap*100:.1f}% of gap]")
    print(f"Unexplained:      {unexplained:.4f} ({(np.exp(unexplained)-1)*100:.2f}%) [{unexplained/total_gap*100:.1f}% of gap]")
    print(f"{'='*60}")
    
   
    print(f"\nDetailed breakdown of EXPLAINED component (characteristics):")
    for i, var in enumerate(covariates):
        contrib = (X_female_mean[var] - X_male_mean[var]) * model_male.params.iloc[i+1]
        print(f"  {var:25s}: {contrib:>8.4f} ({contrib/total_gap*100:>6.1f}% of total gap)")
    
    print("\nInterpretation:")
    print(f"  - {explained/total_gap*100:.1f}% of price gap due to product characteristics")
    print(f"  - {unexplained/total_gap*100:.1f}% of price gap unexplained (potential pink tax)")
    
    print("\n✓ Oaxaca-Blinder decomposition completed")
    
    return {
        'total_gap': total_gap,
        'explained': explained,
        'unexplained': unexplained,
        'model_female': model_female,
        'model_male': model_male,
        'explained_pct': explained/total_gap*100,
        'unexplained_pct': unexplained/total_gap*100
    }


def create_results_summary(twfe_results, hedonic_results, psm_results, 
                          quantile_results, oaxaca_results):
    """Create comprehensive results summary"""
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)
    
    print("\n" + "-"*80)
    print("PHASE 1: BASELINE RESULTS")
    print("-"*80)
    
    print("\nTwo-Way Fixed Effects:")
    for name, model in twfe_results.items():
        coef = model.params['female_product']
        pval = model.pvalues['female_product']
        effect = (np.exp(coef) - 1) * 100
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        print(f"  {name:20s}: {effect:>6.2f}% {sig:3s} (coef={coef:.4f}, p={pval:.4f})")
    
    print("\nHedonic Regression:")
    for name, model in hedonic_results.items():
        coef = model.params['female_product']
        pval = model.pvalues['female_product']
        effect = (np.exp(coef) - 1) * 100
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        print(f"  {name:20s}: {effect:>6.2f}% {sig:3s} (coef={coef:.4f}, p={pval:.4f})")
    
    print("\n" + "-"*80)
    print("PHASE 2: ROBUSTNESS CHECKS")
    print("-"*80)
    
    print("\nPropensity Score Matching:")
    att = psm_results['att_simple']
    att_effect = (np.exp(att) - 1) * 100
    print(f"  ATT (simple):     {att_effect:>6.2f}% (coef={att:.4f})")
    
    reg_att = psm_results['model_matched'].params['female_product']
    reg_att_pval = psm_results['model_matched'].pvalues['female_product']
    reg_att_effect = (np.exp(reg_att) - 1) * 100
    sig = "***" if reg_att_pval < 0.01 else "**" if reg_att_pval < 0.05 else "*" if reg_att_pval < 0.10 else ""
    print(f"  ATT (regression): {reg_att_effect:>6.2f}% {sig:3s} (coef={reg_att:.4f}, p={reg_att_pval:.4f})")
    
    print("\nQuantile Regression (female effect at different quantiles):")
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    for q in quantiles:
        model = quantile_results[f'q{int(q*100)}']
        coef = model.params['female_product']
        pval = model.pvalues['female_product']
        effect = (np.exp(coef) - 1) * 100
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        print(f"  Q{int(q*100):02d}:              {effect:>6.2f}% {sig:3s} (coef={coef:.4f})")
    
    print("\n" + "-"*80)
    print("PHASE 3: MECHANISM ANALYSIS")
    print("-"*80)
    
    print("\nOaxaca-Blinder Decomposition:")
    total_effect = (np.exp(oaxaca_results['total_gap']) - 1) * 100
    explained_effect = (np.exp(oaxaca_results['explained']) - 1) * 100
    unexplained_effect = (np.exp(oaxaca_results['unexplained']) - 1) * 100
    
    print(f"  Total price gap:       {total_effect:>6.2f}%")
    print(f"  Explained (char):      {explained_effect:>6.2f}% ({oaxaca_results['explained_pct']:.1f}% of gap)")
    print(f"  Unexplained (pink tax):{unexplained_effect:>6.2f}% ({oaxaca_results['unexplained_pct']:.1f}% of gap)")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    
    main_coef = twfe_results['twfe_full'].params['female_product']
    main_effect = (np.exp(main_coef) - 1) * 100
    main_pval = twfe_results['twfe_full'].pvalues['female_product']
    
    print(f"\n1. Main estimate (Full TWFE): Female products cost {main_effect:.2f}% more")
    print(f"   Statistical significance: {'YES' if main_pval < 0.05 else 'NO'} (p={main_pval:.4f})")
    
    print(f"\n2. Robustness: Results are {'consistent' if abs(reg_att_effect - main_effect) < 2 else 'somewhat variable'} across methods")
    
    print(f"\n3. Price distribution: Pink tax {'varies' if np.std([quantile_results[f'q{int(q*100)}'].params['female_product'] for q in quantiles]) > 0.02 else 'is consistent'} across price quantiles")
    
    print(f"\n4. Mechanism: {oaxaca_results['unexplained_pct']:.1f}% of price gap is unexplained by product characteristics")
    print(f"   This suggests evidence of {'gender-based pricing (pink tax)' if oaxaca_results['unexplained_pct'] > 30 else 'some pink tax, but mostly explained by product differences'}")
    
    print("\n" + "="*80)
    print("\nNote: Significance levels: *** p<0.01, ** p<0.05, * p<0.10")

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print(" "*25 + "PINK TAX MODEL ESTIMATION")
    print(" "*25 + "Maria's Research Project")
    print("="*80)
    

    print("\nLoading prepared datasets...")
    df_main = pd.read_csv('pink_tax_main_analysis.csv')
    df_psm = pd.read_csv('pink_tax_psm_sample.csv')
    df_female = pd.read_csv('pink_tax_female_matched.csv')
    df_male = pd.read_csv('pink_tax_male_matched.csv')
    
    print(f"✓ Main dataset: {df_main.shape[0]} observations")
    print(f"✓ PSM dataset: {df_psm.shape[0]} observations")
    print(f"✓ Female sample: {df_female.shape[0]} observations")
    print(f"✓ Male sample: {df_male.shape[0]} observations")
    
   
    try:
        
        twfe_results = run_twfe_models(df_main)
        hedonic_results = run_hedonic_regression(df_main)
     
        psm_results = run_propensity_score_matching(df_psm)
        quantile_results = run_quantile_regression(df_main)
   
        oaxaca_results = run_oaxaca_blinder(df_female, df_male)
 
        create_results_summary(twfe_results, hedonic_results, psm_results, 
                             quantile_results, oaxaca_results)
        
        print("\n" + "="*80)
        print("ALL MODELS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nNext steps:")
        print("1. Review results above")
        print("2. For detailed output, access individual model objects")
        print("3. Create tables and figures for your paper")
        print("="*80 + "\n")
        
        return {
            'twfe': twfe_results,
            'hedonic': hedonic_results,
            'psm': psm_results,
            'quantile': quantile_results,
            'oaxaca': oaxaca_results
        }
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
