#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from model_deployment_xgb import predict



app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='API de Predicción de Regresión',
    description='API de Predicción de Regresión')

ns = api.namespace('predict', 
     description='Predicción de Regresión')
   
parser = api.parser()

parser.add_argument(
    'Year', 
    type=int, 
    required=True, 
    help='Valor 1', 
    location='args')

parser.add_argument(
    'Milage', 
    type=int, 
    required=True, 
    help='Valor 2', 
    location='args')

parser.add_argument(
    'State_cod', 
    type=int, 
    required=True, 
    help='Valor 3', 
    location='args')

parser.add_argument(
    'Make_cod', 
    type=int, 
    required=True, 
    help='Valor 4', 
    location='args')

parser.add_argument(
    'Model_cod', 
    type=int, 
    required=True, 
    help='Valor 5', 
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
         "result": predict(args['Year'], args['Milage'], args['State_cod'], args['Make_cod'], args['Model_cod'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
