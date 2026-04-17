import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import pandas as pd
import os
#=================flask code starts here
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory


app = Flask(__name__)
app.secret_key = 'welcome'

dataset = pd.read_csv(os.path.join(BASE_DIR, "Dataset", "SourceCodeError.csv"), usecols=['original_src', 'error'])
labels, count = np.unique(dataset['error'].ravel(), return_counts=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
distil_bert_tokenizer = BertTokenizer.from_pretrained(os.path.join(BASE_DIR, "model/DistilBert"))
distil_bert_model = BertForSequenceClassification.from_pretrained(os.path.join(BASE_DIR, "model/DistilBert"), num_labels=4)# num_labels depends on your classification task

@app.route('/Predict', methods=['GET', 'POST'])
def predictView():
    return render_template('Predict.html', msg='')

@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html', msg='')

@app.route('/UserLogin', methods=['GET', 'POST'])
def UserLogin():
    return render_template('UserLogin.html', msg='')

@app.route('/UserLoginAction', methods=['GET', 'POST'])
def UserLoginAction():
    if request.method == 'POST' and 't1' in request.form and 't2' in request.form:
        user = request.form['t1']
        password = request.form['t2']
        if user == "admin" and password == "admin":
            return render_template('UserScreen.html', msg="<font size=3 color=blue>Welcome "+user+"</font>")
        else:
            return render_template('UserLogin.html', msg="<font size=3 color=red>Invalid login details</font>")

@app.route('/Logout')
def Logout():
    return render_template('index.html', msg='')

@app.route('/PredictAction', methods=['GET', 'POST'])
def PredictAction():
    if request.method == 'POST':
       try:  
           global distil_bert_tokenizer, device, distil_bert_model, labels
           distil_bert_model.to(device)
           testData = pd.read_csv(os.path.join(BASE_DIR, "Dataset", "testData.csv"))
           data = testData.iloc[:,0].astype(str).str.strip().tolist()
           #apply bert encoding to get vector    
           encoded_new_data = distil_bert_tokenizer(
                data,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_attention_mask=True,
                return_tensors='pt'
            )
           new_input_ids = encoded_new_data['input_ids'].to(device)
           new_attention_masks = encoded_new_data['attention_mask'].to(device)
           distil_bert_model.eval()
           with torch.no_grad():
            outputs = distil_bert_model(new_input_ids, attention_mask=new_attention_masks)#apply extension distilbert model for classification
           predictions = torch.argmax(outputs.logits, dim=1).flatten().cpu().numpy()
           output='<table border=1 align=center width=100%><tr><th><font size="3" color="black">Test Source Code</th><th><font size="3" color="black">Error Classification Result</th></tr>'
           for i in range(len(predictions)):
            output+='<tr><td><font size="3" color="black">'+testData.iloc[i,0]+'</td><td><font size="3" color="green">'+labels[predictions[i]]+'</td></tr>'
           output += "</table><br/><br/><br/><br/>"        
           return render_template('UserScreen.html', msg=output)
       except Exception as e:
            import traceback
            tb = traceback.format_exc()
            # Catch any server-side errors
            return render_template('UserScreen.html', msg=f"<font color='red'>Server Error: {str(e)}</font>")

if __name__ == '__main__':
    app.run(debug = True)











    
