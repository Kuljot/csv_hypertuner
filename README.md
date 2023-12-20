# CSVToML
## _CSVtoML is a service to convert your csv file to a ML model Powered by_

[![N|Streamlit](https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png)](https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png)

CSVtoML is a cloud-based, mobile-ready,
Streamlit-powered application to let you convert your csv file to ML model and tune the hyperparameters too.

- The app uses streamlit as frontend
- Scikit Learn in the Backend

## Link to the app
https://csvtoml.azurewebsites.net/
## Features

- Import a CSV file and watch it magically convert to ML Model
- Choose the Best Model for your data
- Works for both classification and reression problems
- Select the hyperparameters search range and get the best hyperparamreters
- Use the generated model to predict the results

Any suggestions are welocme you can contact me 
 [Kuljot Singh] on my email [kuljotme035@gmail.com][df1]

> There is one known issue causing the number of
> columns to change due to streamlit's behavior
> of running the entire app from starting 
> upon interaction with buttons
> !Please press [R] on the keyboard or press button
> And it will resolve


## Tech

CSVtoML uses a number of open source libraries to work :

- [Streamlit](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiKzMaevJuDAxXITWwGHQCFArsQFnoECAYQAQ&url=https%3A%2F%2Fstreamlit.io%2F&usg=AOvVaw0COPYHEMKG9SPXbyFDXyMf&opi=89978449) -  For frontend of the web apps!
- [SciKit Learn](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjY5N2RvJuDAxWybmwGHa7uAksQFnoECAcQAQ&url=https%3A%2F%2Fscikit-learn.org%2F&usg=AOvVaw3pidYsGhglQXGDh_4GMetL&opi=89978449) - Make and train the ML models
- [XGBOOST](https://xgboost.readthedocs.io/en/stable/) - A great ensemble type ML model as an option
- [Numpy](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjw6p75u5uDAxXRTWwGHch4AbQQFnoECAoQAQ&url=https%3A%2F%2Fnumpy.org%2F&usg=AOvVaw3L2i9HVc9ZeynETpNrPxO-&opi=89978449) - for numerical processing
- [Pandas](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiVzuHpu5uDAxU5TWwGHRszBdkQFnoECAUQAQ&url=https%3A%2F%2Fpandas.pydata.org%2F&usg=AOvVaw3cD5ulu4AnZcNusojIyttY&opi=89978449) - handeling the dataframes


And CSVtoML itself is open source with a [https://github.com/Kuljot/csv_hypertuner][dill] on GitHub. Built as an educational project. Please treat it for educational purpose only, PLEASE DONT UPLOAD ANY SENSITIVE/PERSONAL INFO.

## Installation on your local Machine

CSVtoML requires Python 3.11+ to run.

Clone the repository on your device
```sh
git clone https://github.com/Kuljot/csv_hypertuner.git
cd csv_hypertuner
```
Create a virtual environment and activate it
```sh
sudo apt install python3.11-venv
python3.11 -m venv env
source env/bin/activate
```

Install the requirements
```sh
pip install -r requirements.txt
```

Run the application
```sh
streamlit run app.py
```

## Docker

CSVtoML can be containerized via docker.

By default, the Docker will expose port 8080 but streamlit uses 8051, so in the
Dockerfile I have exposed 8051 explicitly. Simply use the Dockerfile to
build the image.

```sh
sudo docker build -t csvtoml .
```
Verify the app by navigating to your server address in
your preferred browser.

```sh
http://localhost:8501/
```

## License
CC
**OpenSource!**
