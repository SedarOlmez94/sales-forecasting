# Promotion1 f2_score:      0.8796632334818133
# Promotion2 f2_score:      0.6603054641256494
# Promotion3 f2_score:      0.4918742528293464
# Promotion4 f2_score:      0.8915634612104615
# Promotion5 f2_score:      0.5287041047093568
# CPI f2_score       :      0.35543474289083
# Weekly_Sales f2_score:  0.5619225329783258

import datawig
import pandas as pd
from sklearn.metrics import r2_score as score

df = pd.read_csv('impute_2013_final.csv')
df = df.fillna(0)

df_train, df_test = datawig.utils.random_split(df, split_ratios=[0.8, 0.2])

imputer = datawig.SimpleImputer(
    input_columns = ['Store', 'Fuel_Price', 'Dept', 'Temperature', 'Promotion1', 'Promotion2', 'Promotion3',
          'Promotion4', 'Promotion5', 'Promotion1_imputed', 'Promotion2_imputed', 'Promotion3_imputed',
          'Promotion4_imputed', 'Promotion5_imputed', 'Unemployment', 'Size (sq ft)', 'CPI_imputed'],
    output_column = 'Weekly_Sales',
    output_path = 'imputer_model'
)

imputer.fit(train_df=df_train, num_epochs = 100)

imputed = imputer.predict(df_test)

new_dataframe = pd.DataFrame(data = imputed)

new_dataframe.to_csv('2013_sales_complete.csv')

f1 = score(imputed['Weekly_Sales'], imputed['Weekly_Sales_imputed'])
print('Weekly_Sales f2_score: ', f1)
