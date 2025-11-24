
# trader_sentiment_analysis.py
# Ready-to-run analysis script for the "Trader Behavior Insights" assignment.
# Usage:
# - Place 'historical_data.csv' and 'fear_greed_index.csv' in the same folder as this script,
#   OR update the TRADER_CSV and FG_CSV paths below.
# - Run with: python trader_sentiment_analysis.py
# - Recommended: run inside a virtualenv or Colab (the notebook version is better for interactive work).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import warnings
import datetime
warnings.filterwarnings('ignore')

# ----- CONFIG: update file paths if needed -----
TRADER_CSV = "historical_data.csv"   # replace with your trader CSV filename
FG_CSV     = "fear_greed_index.csv"  # replace with your fear/greed CSV filename

def safe_read_csv(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        raise

def main():
    print("Loading data...")
    tr = safe_read_csv(TRADER_CSV)
    fg = safe_read_csv(FG_CSV)

    print(f"Trader rows: {len(tr)}; Fear/Greed rows: {len(fg)}")
    print("Preview trader columns:", tr.columns.tolist())
    print("Preview FG columns:", fg.columns.tolist())

    # --- Parse dates in trader data ---
    # Find likely time column
    time_col = None
    for c in tr.columns:
        if 'time' in c.lower() or 'timestamp' in c.lower() or 'date' in c.lower():
            time_col = c
            break
    if time_col is None:
        raise ValueError("No time/timestamp column found in trader CSV. Please inspect column names.")
    print("Using time column:", time_col)
    tr['time'] = pd.to_datetime(tr[time_col], errors='coerce')

    # --- Parse date in fear/greed CSV ---
    date_col = None
    for c in fg.columns:
        if 'date' in c.lower():
            date_col = c
            break
    if date_col is None:
        raise ValueError("No date column found in fear/greed CSV. Please inspect column names.")
    fg['date'] = pd.to_datetime(fg[date_col], errors='coerce').dt.date

    # Identify sentiment column
    sent_col = None
    for c in fg.columns:
        if 'class' in c.lower() or 'sent' in c.lower() or 'fear' in c.lower() or 'greed' in c.lower() or 'value' in c.lower():
            sent_col = c
            break
    if sent_col is None:
        # fallback to second column
        sent_col = fg.columns[1]
    fg = fg.rename(columns={sent_col: 'sentiment_raw'})

    # Map sentiment to categories if numeric
    def map_sentiment(x):
        try:
            x = float(x)
            if x <= 25:
                return 'Fear'
            elif x <= 50:
                return 'Neutral'
            elif x <= 75:
                return 'Greed'
            else:
                return 'Extreme Greed'
        except:
            s = str(x).strip().title()
            if 'Fear' in s:
                return 'Fear'
            if 'Greed' in s:
                return 'Greed'
            return s

    fg['sentiment_cat'] = fg['sentiment_raw'].apply(map_sentiment)

    # --- Prepare trader aggregates ---
    tr['date'] = tr['time'].dt.date

    # Find closed PnL column
    pnl_col = None
    for c in tr.columns:
        if 'pnl' in c.lower() or 'closed' in c.lower():
            pnl_col = c
            break
    if pnl_col is None:
        raise ValueError("No pnl/closedPnL column found in trader CSV. Please inspect column names.")
    tr['closedPnL'] = pd.to_numeric(tr[pnl_col], errors='coerce')

    # Leverage column
    lev_col = None
    for c in tr.columns:
        if 'lev' in c.lower():
            lev_col = c
            break
    if lev_col is not None:
        tr['leverage'] = pd.to_numeric(tr[lev_col], errors='coerce')
    else:
        tr['leverage'] = np.nan

    # Aggregate
    daily = tr.groupby(['account','date']).agg(
        trades=('account','count'),
        pnl_sum=('closedPnL','sum'),
        pnl_mean=('closedPnL','mean'),
        avg_leverage=('leverage','mean')
    ).reset_index()
    daily['profitable'] = (daily['pnl_sum'] > 0).astype(int)

    day_overall = tr.groupby('date').agg(
        total_pnl=('closedPnL','sum'),
        n_trades=('closedPnL','count'),
        mean_leverage=('leverage','mean')
    ).reset_index()

    # Merge with sentiment
    day_overall = day_overall.merge(fg[['date','sentiment_raw','sentiment_cat']], on='date', how='left')
    daily = daily.merge(fg[['date','sentiment_raw','sentiment_cat']], on='date', how='left')

    # --- Save summaries ---
    day_overall.to_csv("day_overall_summary.csv", index=False)
    daily.to_csv("account_day_summary.csv", index=False)
    print("Saved day_overall_summary.csv and account_day_summary.csv")

    # --- Simple EDA plots saved as PNGs ---
    try:
        plt.figure(figsize=(9,5))
        sns.boxplot(x='sentiment_cat', y='total_pnl', data=day_overall)
        plt.title('Daily Total PnL by Sentiment Category')
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig("boxplot_total_pnl_by_sentiment.png")
        plt.close()

        plt.figure(figsize=(9,5))
        sns.violinplot(x='sentiment_cat', y='pnl_sum', data=daily)
        plt.title('Account-Day PnL by Sentiment Category')
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig("violin_pnl_by_sentiment.png")
        plt.close()
        print("Saved plots: boxplot_total_pnl_by_sentiment.png, violin_pnl_by_sentiment.png")
    except Exception as e:
        print("Plotting error:", e)

    # --- Basic statistical tests ---
    cats = day_overall['sentiment_cat'].dropna().unique().tolist()
    groups = {c: day_overall.loc[day_overall['sentiment_cat']==c, 'total_pnl'].dropna() for c in cats}
    from itertools import combinations
    print("Pairwise Mann-Whitney U tests:")
    for a,b in combinations(groups.keys(),2):
        try:
            u,p = stats.mannwhitneyu(groups[a], groups[b], alternative='two-sided')
            print(f"{a} vs {b}: p={p:.4f}")
        except Exception as e:
            print("Error testing", a, b, e)

    # ANOVA
    anova_df = day_overall.dropna(subset=['total_pnl','sentiment_cat'])
    if len(anova_df) > 0:
        model = ols('total_pnl ~ C(sentiment_cat)', data=anova_df).fit()
        print(sm.stats.anova_lm(model, typ=2))
    else:
        print("ANOVA skipped due to insufficient data.")

    # --- Lag analysis ---
    try:
        day_overall['sentiment_num'] = pd.to_numeric(day_overall['sentiment_raw'], errors='coerce')
    except:
        mapping = {'Fear':25,'Neutral':50,'Greed':75,'Extreme Greed':90}
        day_overall['sentiment_num'] = day_overall['sentiment_cat'].map(mapping)

    for lag in range(1,8):
        day_overall[f'sent_lag_{lag}'] = day_overall['sentiment_num'].shift(lag)

    for lag in range(1,8):
        df = day_overall.dropna(subset=[f'sent_lag_{lag}','total_pnl'])
        if len(df)>30:
            lr = LinearRegression().fit(df[[f'sent_lag_{lag}']], df['total_pnl'])
            print(f"Lag {lag}: coef={lr.coef_[0]:.4f}, R2={lr.score(df[[f'sent_lag_{lag}']], df['total_pnl']):.4f}")

    # Cross-correlation
    series_pnl = day_overall.set_index('date')['total_pnl'].dropna()
    series_sent = day_overall.set_index('date')['sentiment_num'].dropna()
    common_index = series_pnl.index.intersection(series_sent.index)
    if len(common_index) > 0:
        s1 = series_pnl.loc[common_index]
        s2 = series_sent.loc[common_index]
        maxlag = 14
        corrs = [s1.corr(s2.shift(lag)) for lag in range(-maxlag, maxlag+1)]
        # save results
        pd.DataFrame({'lag':range(-maxlag, maxlag+1),'corr':corrs}).to_csv("crosscorr_sentiment_pnl.csv", index=False)
        print("Saved crosscorr_sentiment_pnl.csv")
    else:
        print("Cross-correlation skipped due to mismatched indices.")

    # --- Simple predictive model ---
    features = ['sentiment_num','avg_leverage','trades']
    if all(f in daily.columns for f in features):
        model_df = daily.dropna(subset=features+['profitable'])
        X = model_df[features].fillna(0)
        y = model_df['profitable']
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        clf = LogisticRegression(max_iter=1000)
        tss = TimeSeriesSplit(n_splits=5)
        try:
            scores = cross_val_score(clf, Xs, y, cv=tss, scoring='roc_auc')
            print("Logistic ROC AUC CV:", scores, "mean:", scores.mean())
            clf.fit(Xs,y)
            coefs = pd.Series(clf.coef_[0], index=features).sort_values(key=abs, ascending=False)
            print("Coefficients:\n", coefs)
        except Exception as e:
            print("Modeling error:", e)
    else:
        print("Skipping predictive model - required features not present in aggregated data.")

    print("All done. Check the generated CSVs and PNGs for outputs.")

if __name__ == "__main__":
    main()
