"""
STEP 1 — DATA PREPARATION (Fixed for local Windows paths)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

print("=" * 55)
print("  BehavioralGraph — Step 1: Preparing Data")
print("=" * 55)

# ── YOUR FILE PATHS ───────────────────────────────────────────
BA_PATH      = r'C:\Users\Sejal\OneDrive\Desktop\AI\BehavioralGraph\behavioralgraph\data\customer_booking.csv'
CB_PATH      = r'C:\Users\Sejal\OneDrive\Desktop\AI\BehavioralGraph\behavioralgraph\data\mobile_customers.xlsx'
OUT_FEATURES = r'C:\Users\Sejal\OneDrive\Desktop\AI\BehavioralGraph\behavioralgraph\data\features.csv'
OUT_LABELS   = r'C:\Users\Sejal\OneDrive\Desktop\AI\BehavioralGraph\behavioralgraph\data\labels.csv'

# ── 1. Load BA data ───────────────────────────────────────────
print("\n[1/5] Loading British Airways booking data...")
ba = pd.read_csv(BA_PATH, encoding='latin1')
print(f"      Loaded {len(ba):,} booking records")

# ── 2. Load CommBank data ─────────────────────────────────────
print("\n[2/5] Loading CommBank customer data...")
cb = pd.read_excel(CB_PATH, engine='openpyxl')
print(f"      Loaded {len(cb):,} customer records")

# ── 3. Prepare BA features ────────────────────────────────────
print("\n[3/5] Engineering features from BA data...")

# Sample 10,000 rows to keep the graph manageable
ba_sample = ba.sample(n=10000, random_state=42).reset_index(drop=True)

# Convert ALL text columns to numbers first
le = LabelEncoder()
ba_sample['sales_channel_num']  = le.fit_transform(ba_sample['sales_channel'].astype(str))
ba_sample['trip_type_num']      = le.fit_transform(ba_sample['trip_type'].astype(str))
ba_sample['booking_origin_num'] = le.fit_transform(ba_sample['booking_origin'].astype(str))

# Total add-ons
ba_sample['total_addons'] = (
    ba_sample['wants_extra_baggage'] +
    ba_sample['wants_preferred_seat'] +
    ba_sample['wants_in_flight_meals']
)

ba_features = ba_sample[[
    'num_passengers', 'purchase_lead', 'length_of_stay',
    'flight_hour', 'flight_duration', 'total_addons',
    'sales_channel_num', 'trip_type_num', 'booking_origin_num',
    'booking_complete'
]].copy()

# ── 4. Prepare CommBank features ──────────────────────────────
print("\n[4/5] Engineering features from CommBank data...")

cb_sample = cb.sample(n=10000, random_state=42).reset_index(drop=True)
cb_sample['gender_num']  = le.fit_transform(cb_sample['gender'].astype(str))
cb_sample['salary_norm'] = (cb_sample['salary'] - cb_sample['salary'].min()) / \
                            (cb_sample['salary'].max() - cb_sample['salary'].min())
cb_sample['age_norm']    = (cb_sample['age'] - cb_sample['age'].min()) / \
                            (cb_sample['age'].max() - cb_sample['age'].min())

cb_features = cb_sample[['gender_num', 'age_norm', 'salary_norm']].copy()

# ── 5. Combine and scale ──────────────────────────────────────
print("\n[5/5] Combining datasets and scaling...")

combined = pd.concat([
    ba_features.reset_index(drop=True),
    cb_features.reset_index(drop=True)
], axis=1)

feature_cols = [
    'num_passengers', 'purchase_lead', 'length_of_stay',
    'flight_hour', 'flight_duration', 'total_addons',
    'sales_channel_num', 'trip_type_num', 'booking_origin_num',
    'gender_num', 'age_norm', 'salary_norm'
]

X = combined[feature_cols]
y = combined['booking_complete']

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

X_scaled.to_csv(OUT_FEATURES, index=False)
y.to_csv(OUT_LABELS, index=False)

print("\n✅ Done! Files saved:")
print(f"   features.csv — {X_scaled.shape[0]:,} customers, {X_scaled.shape[1]} features (all numeric)")
print(f"   labels.csv   — booking outcome (0 or 1)")
print(f"\n   Class balance: {int(y.sum()):,} completed ({y.mean()*100:.1f}%)")
print(f"                  {int((1-y).sum()):,} not completed ({(1-y.mean())*100:.1f}%)")