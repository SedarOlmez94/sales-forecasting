import datawig
import sklearn
import pandas as pd

df = pd.read_csv('final_dataframe.csv')
df_train, df_test = datawig.utils.random_split(df, split_ratios=[0.8, 0.2])


imputer_Promo1 = datawig.SimpleImputer(
    input_columns = ['Weekly_Sales', 'Temperature', 'Fuel_Price',
    'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5', 'CPI'],
    output_column = 'Promotion1',
    output_path = 'imputer_model'
)

imputer_Promo1.fit(train_df=df_train, num_epochs = 30)

imputed = imputer_Promo1.predict(df_test)

Promotion1_imputed = pd.DataFrame(data = imputed)
Promotion1_imputed.to_csv('Promotion1_imputed.csv')
