from flask import Flask, request, redirect
from flask_restful import Resource, Api
from flask_cors import CORS
import os
import prediction
from PIL import Image
import io


app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

class Test(Resource):
    def get(self):
        return 'Welcome to, Test App API!'

    def post(self):
        try:
            value = request.get_json()
            if(value):
                return {'Post Values': value}, 201

            return {"error":"Invalid format."}

        except Exception as error:
            return {'error': error}

class GetPredictionOutput(Resource):
    def get(self):
        return {'testing':True}

    def post(self):
        try: 
            bytes_data = io.BytesIO(request.data)
            image = Image.open(bytes_data)
            if request.mimetype == "image/jpg" or "image/jpeg":
                filename = "TestFile.jpg"
            else:   
                return {'error': "File type not supoorted yet"}
            image.save(filename)
            predicted_label, probabilities = prediction.predict(image, filename)
            return {'predicted_document_type':predicted_label, 'probabilities':probabilities}

        except Exception as error:
            return {'error': error}

api.add_resource(Test,'/')
api.add_resource(GetPredictionOutput,'/getPredictionOutput')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)