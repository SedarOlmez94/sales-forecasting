# Promotion1 f2_score:  0.8514850585077608
# Promotion2 f2_score:  0.5711826844935245
# Promotion3 f2_score:  0.3159253215777095
# Promotion4 f2_score:  0.8756440486801857
# Promotion5 f2_score:  0.49345474586167937
# CPI f2_score:  0.33283797245311286

import datawig
import sklearn
import pandas as pd
from sklearn.metrics import r2_score as score
from sklearn.metrics import classification_report as cr

df = pd.read_csv('Promotion1_imputed.csv')
df = df.round(1)
df = df.fillna(0)

df_train, df_test = datawig.utils.random_split(df, split_ratios=[0.8, 0.2])

imputer = datawig.SimpleImputer(
    input_columns = ['Weekly_Sales', 'Temperature', 'Fuel_Price',
    'Promotion5', 'Promotion4', 'CPI', 'Promotion2', 'Promotion3'],
    output_column = 'Promotion1',
    output_path = 'imputer_model'
)

imputer.fit(train_df=df_train, num_epochs = 50)

imputed = imputer.predict(df_test)

f1 = score(imputed['Promotion1'], imputed['Promotion1_imputed'])
print('Promotion1 f2_score: ', f1)
