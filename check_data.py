import pandas as pd
import os

assets = ['EURUSD', 'SP500', 'Oro', 'ECH']
data_dir = 'data/processed'

for activo in assets:
    path = os.path.join(data_dir, f'{activo}_processed.csv')
    if not os.path.exists(path):
        print(f'\n❌ {activo}: ARCHIVO NO ENCONTRADO')
        continue
    df = pd.read_csv(path)
    print(f'\n{"="*70}')
    print(f'📊 {activo} | Filas: {len(df)} | Columnas: {len(df.columns)}')
    print(f'{"="*70}')
    
    # Check nulls
    nulls = df.isnull().sum()
    cols_with_nulls = nulls[nulls > 0]
    if len(cols_with_nulls) > 0:
        print(f'⚠️  Columnas con NaN:')
        for col, n in cols_with_nulls.items():
            pct = n/len(df)*100
            print(f'   {col}: {n} NaN ({pct:.1f}%)')
    else:
        print(f'✅ Sin NaN en ninguna columna')
    
    # Check columns with all zeros
    zero_cols = [c for c in df.select_dtypes(include='number').columns if (df[c] == 0).all()]
    if zero_cols:
        print(f'❌ Columnas 100% CEROS: {zero_cols}')
    
    # Check columns with constant values  
    const_cols = [c for c in df.select_dtypes(include='number').columns if df[c].nunique() <= 1]
    if const_cols:
        print(f'⚠️  Columnas CONSTANTES (1 valor): {const_cols}')
    
    # Check date range
    if 'time' in df.columns:
        print(f'📅 Rango: {df["time"].iloc[0]} → {df["time"].iloc[-1]}')
    
    # Print all columns with basic stats
    print(f'\n   {"Col":<25} {"Min":>12} {"Max":>12} {"Mean":>12} {"Std":>12} {"NaN":>6}')
    print(f'   {"-"*25} {"-"*12} {"-"*12} {"-"*12} {"-"*12} {"-"*6}')
    for col in df.select_dtypes(include='number').columns:
        mn = df[col].min()
        mx = df[col].max()
        me = df[col].mean()
        st = df[col].std()
        na = df[col].isna().sum()
        print(f'   {col:<25} {mn:>12.4f} {mx:>12.4f} {me:>12.4f} {st:>12.4f} {na:>6}')
