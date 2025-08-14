#!/usr/bin/env python3
"""
AI-Driven Customer Success Program - Customer Scoring Algorithm
Identifies high-potential customer advocates based on multiple engagement metrics
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import json
from datetime import datetime, timedelta

class CustomerAdvocacyScorer:
    def __init__(self, config_file='config/scoring-parameters.json'):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def load_customer_data(self, data_path='data/sample-customer-data.csv'):
        """Load and preprocess customer data"""
        df = pd.read_csv(data_path)
        return self.preprocess_data(df)
    
    def preprocess_data(self, df):
        """Clean and engineer features for scoring"""
        # Calculate engagement metrics
        df['login_frequency_score'] = np.log1p(df['monthly_logins'])
        df['feature_adoption_rate'] = df['features_used'] / df['features_available']
        df['support_sentiment_score'] = df['support_tickets_resolved'] / (df['support_tickets_total'] + 1)
        df['contract_stability_score'] = df['months_as_customer'] / 12
        df['expansion_indicator'] = (df['current_arr'] / df['initial_arr']) - 1
        
        # Industry influence scoring
        df['title_influence_score'] = df['contact_title'].map(self.config['title_influence_mapping'])
        df['company_size_score'] = np.log1p(df['company_employee_count']) / 10
        
        # Competitive displacement value
        df['competitive_value'] = df['replaced_competitor'].map(self.config['competitor_value_mapping'])
        
        return df
    
    def calculate_advocacy_score(self, df):
        """Calculate comprehensive advocacy potential score"""
        features = [
            'login_frequency_score', 'feature_adoption_rate', 'support_sentiment_score',
            'contract_stability_score', 'expansion_indicator', 'title_influence_score',
            'company_size_score', 'competitive_value', 'nps_score', 'reference_willingness'
        ]
        
        # Handle missing values
        df[features] = df[features].fillna(df[features].median())
        
        # Calculate weighted advocacy score
        weights = self.config['feature_weights']
        advocacy_score = 0
        
        for feature in features:
            if feature in weights:
                advocacy_score += df[feature] * weights[feature]
        
        # Normalize to 0-100 scale
        df['advocacy_score'] = ((advocacy_score - advocacy_score.min()) / 
                               (advocacy_score.max() - advocacy_score.min())) * 100
        
        return df
    
    def identify_top_candidates(self, df, top_n=50):
        """Identify top advocacy candidates"""
        # Additional filtering criteria
        df_filtered = df[
            (df['nps_score'] >= 8) &
            (df['months_as_customer'] >= 6) &
            (df['expansion_indicator'] >= 0) &
            (df['support_sentiment_score'] >= 0.7)
        ]
        
        # Sort by advocacy score
        top_candidates = df_filtered.nlargest(top_n, 'advocacy_score')
        
        return top_candidates[['customer_id', 'company_name', 'contact_name', 
                              'contact_title', 'advocacy_score', 'nps_score', 
                              'current_arr', 'industry']]
    
    def generate_outreach_recommendations(self, candidates_df):
        """Generate personalized outreach recommendations"""
        recommendations = []
        
        for _, row in candidates_df.iterrows():
            rec = {
                'customer_id': row['customer_id'],
                'company_name': row['company_name'],
                'contact_name': row['contact_name'],
                'advocacy_score': round(row['advocacy_score'], 1),
                'outreach_priority': self.get_priority_level(row['advocacy_score']),
                'recommended_approach': self.get_outreach_approach(row),
                'expected_value': self.estimate_referral_value(row),
                'talking_points': self.generate_talking_points(row)
            }
            recommendations.append(rec)
        
        return recommendations
    
    def get_priority_level(self, score):
        """Determine outreach priority based on advocacy score"""
        if score >= 90:
            return "Immediate - Executive Outreach"
        elif score >= 80:
            return "High - Account Manager Outreach"  
        elif score >= 70:
            return "Medium - Customer Success Outreach"
        else:
            return "Low - Automated Outreach"
    
    def get_outreach_approach(self, row):
        """Recommend personalized outreach approach"""
        if row['current_arr'] > 100000:
            return "Executive dinner or strategic session"
        elif row['nps_score'] >= 9:
            return "Customer advisory board invitation"
        else:
            return "Structured success story interview"
    
    def estimate_referral_value(self, row):
        """Estimate potential value of referrals from this customer"""
        base_value = row['current_arr'] * 0.5  # Assume referrals are 50% of current ARR
        industry_multiplier = self.config.get('industry_referral_multipliers', {}).get(row['industry'], 1.0)
        return base_value * industry_multiplier
    
    def generate_talking_points(self, row):
        """Generate personalized talking points for outreach"""
        points = []
        
        if row['advocacy_score'] >= 85:
            points.append(f"Recognition as top 10% customer success story")
        
        if row['current_arr'] > 50000:
            points.append(f"Enterprise-level implementation insights valuable to peers")
            
        points.append(f"Opportunity to influence product roadmap through advisory participation")
        points.append(f"Exclusive access to beta features and industry research")
        
        return points

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Customer Advocacy Scoring System')
    parser.add_argument('--initialize', action='store_true', help='Initialize scoring system')
    parser.add_argument('--top-candidates', action='store_true', help='Generate top candidate list')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    scorer = CustomerAdvocacyScorer()
    
    if args.initialize:
        print("Initializing Customer Advocacy Scoring System...")
        df = scorer.load_customer_data()
        df = scorer.calculate_advocacy_score(df)
        print(f"Processed {len(df)} customer records")
        print(f"Average advocacy score: {df['advocacy_score'].mean():.1f}")
        
    if args.top_candidates:
        df = scorer.load_customer_data()
        df = scorer.calculate_advocacy_score(df)
        candidates = scorer.identify_top_candidates(df)
        recommendations = scorer.generate_outreach_recommendations(candidates)
        
        if args.output:
            pd.DataFrame(recommendations).to_csv(args.output, index=False)
            print(f"Top candidates saved to {args.output}")
        else:
            print("\nTop Advocacy Candidates:")
            for rec in recommendations[:10]:
                print(f"- {rec['company_name']}: Score {rec['advocacy_score']} ({rec['outreach_priority']})")
