from flask import Flask
from flask import request
from flask import render_template
from deeplogo import DeepLogo
import os
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/assets')

app = Flask(__name__,  static_folder=ASSETS_DIR)


dl = DeepLogo()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    brand = "NOT CLASSIFIED YET"
    if request.method == 'POST':
        text = request.form['text']
        brand = text
        brand = dl.predict(text)

    return render_template('index.html', brand=brand)


if __name__ == '__main__':
    app.run()





