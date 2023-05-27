
from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from model_deployment import predict

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='MIAD2023-12 - Clasificación de género de películas',
    description='Esta API es construida con la intención que puedas predecir el género de la película introduciendo una pequeña descripción de la misma')

ns = api.namespace('predict', 
     description='Predicción de género de películas')
   
parser = api.parser()

parser.add_argument(
    'Plot', 
    type=str, 
    required=True, 
    help='Descrición de la pelicula', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class ClasificacionApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict(args['Plot'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)