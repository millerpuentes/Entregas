#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from model_deployment_xgb import predict



app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='API de Predicción de precios en Vehículos',
    description='Esta API es construida con la intención que puedas predecir el precio de un Vehículos conociendo su Año de puesta en marcha, el Millaje recorrido, el Estado en donde se encuentra, el Fabricante y el Modelo')

ns = api.namespace('predict', 
     description='Predicción de precios en Vehículos')
   
parser = api.parser()

parser.add_argument(
    'Year', 
    type=int, 
    required=True, 
    help='Ingrese el año', 
    location='args')

parser.add_argument(
    'Mileage', 
    type=int, 
    required=True, 
    help='Ingrese el Millaje recorrido', 
    location='args')

parser.add_argument(
    'State', 
    type=str, 
    required=True, 
    help='Estado donde está el vehículo', 
    location='args')

parser.add_argument(
    'Make', 
    type=str, 
    required=True, 
    help='Ingrese el Fabricante', 
    location='args')

parser.add_argument(
    'Model', 
    type=str, 
    required=True, 
    help='Ingrese el Modelo', 
    location='args')
resource_fields = api.model('Resource', {
    'result': fields.Float,
})

@ns.route('/')
class RegressionApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict(args['Year'], args['Mileage'], args['State'], args['Make'], args['Model'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)