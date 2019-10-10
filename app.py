from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#
import os
import socket
import time

# Preparisions for flask
from flask import render_template  
from wtforms import Form, FileField, StringField
from wtforms.validators import InputRequired
from flask_wtf.file import FileRequired, FileAllowed
from flask import Flask, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import CombinedMultiDict

# Prerequisites for cassandra
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
import logging
log = logging.getLogger()
log.setLevel('INFO')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
log.addHandler(handler)

# Preparisions for tensorflow
# Imports
import numpy as np
import tensorflow as tf
from mnist_cnn import cnn_model_fn
#tf.logging.set_verbosity(tf.logging.INFO)
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir="/data/mnist_model")

def predict(file_path):
    image_raw = tf.gfile.FastGFile(file_path,'rb').read()
    img = tf.image.decode_jpeg(image_raw)
    with tf.Session() as sess:
        img_ = img.eval()
        print(img_.shape)
    img_ = img_.astype(np.float32)
    pred_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={"x":img_},
        num_epochs=1,
        shuffle=False)
    pred_results = mnist_classifier.predict(input_fn=pred_input_fn)
    for item in pred_results:
        pred = item['classes']
    return pred

KEYSPACE = "bigdataspace"
os.environ['TZ']='AEST-8AEDT-11,M10.5.0,M3.5.0'
time.tzset()
# cassandra
def insertPic(f,pred,time):
    cluster = Cluster(contact_points=['zjy-cassandra3'],port=9042)
    session = cluster.connect()
    log.info("Using keyspace...")
    try:
        log.info("setting keyspace...")
        session.set_keyspace(KEYSPACE)
        log.info("inserting to table...")
        session.execute("INSERT INTO MNIST (filename,pred,time) VALUES('"+f+"',"+str(pred)+",'"+time+"')")
    except Exception as e:
        log.error("Unable to insert pic")
        log.error(e)

app = Flask(__name__,template_folder=os.path.join(os.path.dirname(__file__)))
app.debug = True

UPLOAD_PATH = os.path.join(os.path.dirname(__file__), 'images')
app.config['SECRET_KEY']='jdlfsdp'

class Myform(Form):
    pic = FileField(u'上传',validators=[FileRequired(), FileAllowed(['jpg', 'png', 'gif'])])
    #submit = SubmitField("Submit")

@app.route("/")
def hello():
    form = Myform()
    return render_template('/index.html',hostname=socket.gethostname(),form=form)

@app.route("/mnist", methods=['POST'])
def mnist():
    form = Myform(CombinedMultiDict([request.form, request.files]))
    if form.validate():
       pic = request.files.get('pic')
       file_path = os.path.join(UPLOAD_PATH,pic.filename)
       pic.save(file_path)
       pred = predict(file_path)
       datetime = time.strftime("%Y-%m-%d %H:%M:%S")
       insertPic(pic.filename,pred,datetime)
       return ("%s%s%s%s%s%s%s%s%s"%("Filename: ",pic.filename,"\n",
                                     "Pred: ",str(pred),"\n",
                                     "UpdateTime: ",datetime,"\n"))
    else:
       return '失败'

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
