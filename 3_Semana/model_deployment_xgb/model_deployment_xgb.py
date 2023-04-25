import pandas as pd
import joblib
import sys
import os

def predict(var1, var2, var3, var4, var5):

    xgb = joblib.load(os.path.dirname(__file__) + '/xgb_reg_f.pkl') 

    # Create dataframe with input variables
    input_data = pd.DataFrame([[var1, var2, var3, var4, var5]], columns=['Year', 'Mileage', 'State_cod', 'Make_cod', 'Model_cod'])

    # Make prediction
    prediction = xgb.predict(input_data)

    return prediction


if __name__ == "__main__":
    
    if len(sys.argv) <= 5:
        print('Ingrese los cinco atributos')
        
    else:

        Year = float(sys.argv[1])
        Mileage = float(sys.argv[2])
        State_cod = float(sys.argv[3])
        Make_cod = float(sys.argv[4])
        Model_cod = float(sys.argv[5])

        prediction = predict(Year, Mileage, State_cod, Make_cod, Model_cod)
        
        print('Los valores de entrada son: ', Year, Mileage, State_cod, Make_cod, Model_cod)
        print('El valor de predicción es: ', prediction)
