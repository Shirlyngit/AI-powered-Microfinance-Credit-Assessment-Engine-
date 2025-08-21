import pandas as pd
import numpy as np

np.random.seed(42)
n_records = 16000

# Categorical mappings
education_mapping = {
    0: 'Primary',
    1: 'Secondary',
    2: 'Undergraduate',
    3: 'Postgraduate'
}

employment_type_mapping = {
    0: 'Unemployed',
    1: 'Salaried',
    2: 'Self-Employed',
    3: 'Contract-Part-time'
}

marital_status_mapping = {
    0: 'Single',
    1: 'Married',
    2: 'Divorced/Widowed'
}

loan_purpose_mapping = {
    0: 'Debt Consolidation',
    1: 'Home Improvement',
    2: 'Business',
    3: 'Education',
    4: 'Other-Miscellaneous'
}

# numerical features
age = np.random.randint(low=18, high=75, size=n_records)
income = np.random.lognormal(mean=11.5, sigma=0.8, size=n_records).astype(int)
months_employed = np.minimum(np.random.geometric(p=0.015, size=n_records) * 12, 120)
dtiratio = np.random.beta(a=2, b=5, size=n_records)

# categorical features
education = np.random.choice(list(education_mapping.keys()), size=n_records, p=[0.1, 0.35, 0.45, 0.1])
employment_type = np.random.choice(list(employment_type_mapping.keys()), size=n_records, p=[0.15, 0.45, 0.25, 0.15])
marital_status = np.random.choice(list(marital_status_mapping.keys()), size=n_records, p=[0.4, 0.45, 0.15])
loan_purpose = np.random.choice(list(loan_purpose_mapping.keys()), size=n_records, p=[0.35, 0.2, 0.15, 0.1, 0.2])

# Binary features
has_mortgage = np.random.randint(0, 2, size=n_records)
has_dependents = np.random.choice([0, 1], size=n_records, p=[0.3, 0.7])
has_cosigner = np.random.choice([0, 1], size=n_records, p=[0.7, 0.3])

# Credit Score
base_credit_score = 500 + (income / 10000) * 10 + (months_employed / 10) * 5 - (dtiratio * 100) * 2
credit_score = np.clip(base_credit_score + np.random.normal(0, 50, size=n_records), 300, 850).astype(int)


data_dict = {
    'Age': age,
    'Income': income,
    'MonthsEmployed': months_employed,
    'DTIRatio': dtiratio,
    'Education': education,
    'EmploymentType': employment_type,
    'MaritalStatus': marital_status,
    'HasMortgage': has_mortgage,
    'HasDependents': has_dependents,
    'LoanPurpose': loan_purpose,
    'HasCoSigner': has_cosigner,
    'CreditScore': credit_score
}

df = pd.DataFrame(data_dict)

# categorical columns
df['Education'] = df['Education'].map(education_mapping)
df['EmploymentType'] = df['EmploymentType'].map(employment_type_mapping)
df['MaritalStatus'] = df['MaritalStatus'].map(marital_status_mapping)
df['LoanPurpose'] = df['LoanPurpose'].map(loan_purpose_mapping)

# df to a CSV file
df.to_csv('loans.csv', index=False)

print("Successfully generated 16,000 realistic records with categorical labels and saved them to 'loans.csv'.")
print("\nDataFrame head:")
print(df.head())