from flask import *
from flask import request
from flask import render_template
from deeplogo import DeepLogo
import os
from flask import jsonify
from celery import Celery
from tasks import predict
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/assets')


app = Flask(__name__,  static_folder=ASSETS_DIR)

app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery('tasks', backend='redis://localhost', broker='amqp://guest@localhost:5672//')

celery.conf.update(app.config)

dl = DeepLogo()

def brand2img(brand):
	if brand == "noise":
		return "http://www.selonen.org/arto/netbsd/noise.png"
	if brand == "nike":
		return "http://www.myiconfinder.com/uploads/iconsets/256-256-15f5c0bd367d23e4ed1a1fc800bc2ed6-nike.png"
	if brand == "cocacola":
		return "https://www.etu.edu.tr/files/sirket/2016/10/03/0975fc015fa2575a06d1216e2989f6c5.png"
	if brand == "pepsi":
		return "http://www.myiconfinder.com/uploads/iconsets/256-256-756be8a5c69426cc2552448b6b60fb75-Pepsi.png"
	if brand == "apple":
		return "http://dxf1.com/images/jdownloads/screenshots/apple.png"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    brand = "https://www.shareicon.net/data/256x256/2015/10/02/110418_question_512x512.png"

    if request.method == 'POST':
        text = request.form['text']
        print(text)

        if len(text) > 4:
        #brand = text
        	task = predict.apply_async(text)
        	return jsonify({}), 202, {'Location': url_for('taskstatus',
                                                  task_id=task.id)}

            #brand = dl.predict(text)
            #brand = brand2img(brand)
        # else:
        #     brand = "https://www.shareicon.net/data/256x256/2015/10/02/110418_question_512x512.png"

    return render_template('index.html', brand=brand)


@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = predict.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)

@celery.task(bind=True)
def predict(self, url):

	self.update_state(state="PROGRESS", 
					  meta={"current":10, 
					  		"total":100, 
					  		"status":"Dowloading video"})
	frames = dl.downloading_video(url)

	self.update_state(state="PROGRESS", 
					  meta={"current":20, 
					  		"total":100, 
					  		"status":"Processing video"})

	imgs = dl.create_imgs(frames)

	self.update_state(state="PROGRESS", 
					  meta={"current":60, 
					  		"total":100, 
					  		"status":"Using CNN to classify frames"})

	softmaxes = dl.classify_CNN(imgs)

	self.update_state(state="PROGRESS", 
					  meta={"current":60, 
					  		"total":100, 
					  		"status":"Using RNN to classify video"})

	brand = brand2img(dl.classify_RNN(softmaxes))

	return {"current":100, "total":100, "status":"task completed", "result":brand}


if __name__ == '__main__':
    app.run()





