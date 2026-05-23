from flask import Flask,request,render_template
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

application=Flask(__name__)
app=application

## import ridge regression model 
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
scaler_model=pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='POST':
            data=request.form
            Temperature=float(data.get('Temperature', 0))
            RH=float(data.get('RH', 0))
            Ws=float(data.get('Ws', 0))
            Rain=float(data.get('Rain', 0))
            FFMC=float(data.get('FFMC', 0))
            DMC=float(data.get('DMC', 0))
            ISI=float(data.get('ISI', 0))
            CLS=float(data.get('CL', 0))
            REG=float(data.get('REG', 0))


            new_data_scalar=scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,CLS,REG]])
            result=ridge_model.predict(new_data_scalar)
            
            return render_template('home.html',results=result[0])
        
        
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)



