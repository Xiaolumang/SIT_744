import flask
import pickle
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('D:\Chrome_downloads\model4')
app = flask.Flask(__name__)
@app.route('/model',methods=['POST'])
def serve_model():
  request_data = flask.request.get_json(force=True)

  img = request_data['img']
  img = np.array(img).reshape(-1,150,150,3)
  x = model.predict(img)
  x = ['Recyclables','Hazardous Waste','Kitchen Waste','Other Waste'][x.argmax()]
  return x
if __name__ == "__main__":
  app.run()