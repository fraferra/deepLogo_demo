from flask import Flask
from flask import request
from flask import render_template
from deeplogo import DeepLogo
import os
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/assets')

app = Flask(__name__,  static_folder=ASSETS_DIR)


dl = DeepLogo()

def brand2img(brand):
	if brand == "noise":
		return "http://www.selonen.org/arto/netbsd/noise.png"
	if brand == "nike":
		return "http://www.myiconfinder.com/uploads/iconsets/256-256-15f5c0bd367d23e4ed1a1fc800bc2ed6-nike.png"
	if brand == "cocacola":
		return "http://www.iconsdb.com/icons/preview/red/coca-cola-xxl.png"
	if brand == "pepsi":
		return "http://www.myiconfinder.com/uploads/iconsets/256-256-756be8a5c69426cc2552448b6b60fb75-Pepsi.png"
	if brand == "apple":
		return "http://dxf1.com/images/jdownloads/screenshots/apple.png"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    brand = "https://www.shareicon.net/data/256x256/2015/10/02/110418_question_512x512.png"
    if request.method == 'POST':
        text = request.form['text']
        if len(text) > 4:
        #brand = text
	        brand = dl.predict(text)
	        brand = brand2img(brand)
	    else:
	    	brand = "https://www.shareicon.net/data/256x256/2015/10/02/110418_question_512x512.png"

    return render_template('index.html', brand=brand)


if __name__ == '__main__':
    app.run(port=80)





