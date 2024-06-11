from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('modelocultivo.pkl')
app.logger.debug('Modelo cargado correctamente.')

# Mapeo de predicciones numéricas a nombres de cultivos
crop_mapping = {
    0: "Apple",
    1: "Banana",
    2: "Blackgram",
    3: "ChickPea",
    4: "Coconut",
    5: "Coffee",
    6: "Cotton",
    7: "Grapes",
    8: "Jute",
    9: "KidneyBeans",
    10: "Lentil",
    11: "Maize",
    12: "Mango",
    13: "MothBeans",
    14: "MungBean",
    15: "Muskmelon",
    16: "Orange",
    17: "Papaya",
    18: "PigeonPeas",
    19: "Pomegranate",
    20: "Rice",
    21: "Watermelon"
}

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        Potassium = float(request.form['Potassium'])
        Nitrogen = float(request.form['Nitrogen'])
        Phosphorus = float(request.form['Phosphorus'])
        Humidity = float(request.form['Humidity'])
        Rainfall = float(request.form['Rainfall'])

        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[Potassium, Nitrogen, Phosphorus, Humidity, Rainfall]], columns=['Potassium', 'Nitrogen', 'Phosphorus', 'Humidity', 'Rainfall'])

        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')

        # Obtener el nombre del cultivo de la predicción
        crop_name = crop_mapping.get(prediction[0], "Cultivo desconocido")
        app.logger.debug(f'Nombre del cultivo: {crop_name}')

        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': crop_name})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
