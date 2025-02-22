# first line: 50
@memory.cache
def preprocess_data(df, target_col='attack_cat'):
    for col in ['id', 'label']:
        if col in df.columns:
            df = df.drop(columns=[col])
    target = None
    if target_col in df.columns:
        target = df[target_col]
        df = df.drop(columns=[target_col])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in numeric_cols:
        med = df[col].median()
        if med > 0 and df[col].max() > 10 * med and df[col].max() > 10:
            perc95 = df[col].quantile(0.95)
            df[col] = np.where(df[col] > perc95, perc95, df[col])
    for col in numeric_cols:
        if df[col].nunique() > 50:
            if df[col].min() == 0:
                df[col] = np.log(df[col] + 1)
            else:
                df[col] = np.log(df[col])
    for col in categorical_cols:
        if df[col].nunique() > 6:
            top_categories = df[col].value_counts().head(5).index
            df[col] = df[col].apply(lambda x: x if x in top_categories else '-')
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    if target is not None:
        df[target_col] = target
    return df
