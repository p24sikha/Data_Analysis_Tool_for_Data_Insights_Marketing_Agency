import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
from PIL import Image
import io
import os
import gdown
import zipfile
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

gdrive_url = 'https://drive.google.com/uc?id=1zLNe-QteUWjlNLmyWSfiB-D8ZxdPZAFs'
output_file = 'images.zip'

gdown.download(gdrive_url, output_file, quiet=False)

with zipfile.ZipFile(output_file, 'r') as zip_ref:
    zip_ref.extractall('extracted_images')

image_folder = 'extracted_images'
image_paths = []

for root, dirs, files in os.walk(image_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_paths.append(os.path.join(root, file))

num_images_to_show = 10
random_images = random.sample(image_paths, num_images_to_show)


image_folder = 'extracted_images'

IMG_SIZE = (224, 224) 
BATCH_SIZE = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    image_folder,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    image_folder,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, validation_data=validation_generator, epochs=1)
from sklearn.metrics import classification_report, confusion_matrix

val_steps = validation_generator.samples // validation_generator.batch_size
predictions = model.predict(validation_generator, steps=val_steps)
y_pred = np.argmax(predictions, axis=1)
y_true = validation_generator.classes[:len(y_pred)]
conf_matrix = confusion_matrix(y_true, y_pred)
model.save('property_classification_model.h5')





app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

model = load_model('property_classification_model.h5')
IMG_SIZE = (224, 224)
class_names = ['House', 'Apartment', 'Office', 'Architectural Style 1', 'Architectural Style 2', 'Condition 1', 'Condition 2']

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("PropertyVision Dashboard", className="text-center text-primary mb-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Upload(id='upload-image', children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                       style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                              'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                              'textAlign': 'center', 'margin': '10px'}, multiple=False),
            html.Div(id='output-image-upload')
        ], width=6),
        dbc.Col([
            html.Div(id='output-prediction'),
            dcc.Graph(id='probability-graph')
        ], width=6)
    ])
])

def preprocess_image(image_data):
    img = Image.open(io.BytesIO(image_data)).resize(IMG_SIZE)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return decoded

@app.callback([Output('output-image-upload', 'children'),
               Output('output-prediction', 'children'),
               Output('probability-graph', 'figure')],
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'), State('upload-image', 'last_modified')])
def update_output(contents, filename, last_modified):
    if contents is not None:
        image_data = parse_contents(contents)
        img = Image.open(io.BytesIO(image_data))
        prediction_input = preprocess_image(image_data)
        predictions = model.predict(prediction_input)
        predicted_class = class_names[np.argmax(predictions)]
        img_display = html.Img(src=contents, style={'height': '300px'})
        prob_fig = go.Figure(data=[go.Bar(x=class_names, y=predictions[0])])
        prob_fig.update_layout(title='Prediction Probabilities', xaxis_title='Classes', yaxis_title='Probability')
        return img_display, f"Predicted Class: {predicted_class}", prob_fig
    return None, None, go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)
