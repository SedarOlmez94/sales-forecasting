# Promotion1 f2_score:  0.8796632334818133
# Promotion2 f2_score:  0.6603054641256494
# Promotion3 f2_score:  0.4918742528293464
# Promotion4 f2_score:  0.8915634612104615
# Promotion5 f2_score:  0.5287041047093568
# CPI f2_score:  0.35543474289083

import datawig
import sklearn
import pandas as pd
from sklearn.metrics import r2_score as score
from sklearn.metrics import classification_report as cr

df = pd.read_csv('final_dataframe.csv')
df = df.fillna(0)

df_train, df_test = datawig.utils.random_split(df, split_ratios=[0.8, 0.2])

imputer = datawig.SimpleImputer(
    input_columns = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'Promotion1', 'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5'],
    output_column = 'CPI',
    output_path = 'imputer_model'
)

imputer.fit(train_df=df_train, num_epochs = 50)

imputed = imputer.predict(df_test)

new_dataframe = pd.DataFrame(data = imputed)

new_dataframe.to_csv('CPI_imputed.csv')

f1 = score(imputed['CPI'], imputed['CPI_imputed'])
print('CPI f2_score: ', f1)
