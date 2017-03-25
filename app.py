from flask import *
from flask import request
from flask import render_template
from deeplogo import DeepLogo
import os
from flask import jsonify
from celery import Celery
#from tasks import predict
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/assets')

text = ''
app = Flask(__name__,  static_folder=ASSETS_DIR)



app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'


# Initialize extensions
#dl = DeepLogo()

# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'], backend=app.config['CELERY_RESULT_BACKEND'])
celery.conf.update(app.config)


@celery.task(bind=True)
def long_task(self, url):

	dl = DeepLogo()

	print("###### URL ##### : "+ url)

	self.update_state(state="PROGRESS", 
					  meta={"current":10, 
					  		"total":100, 
					  		"status":"Dowloading video"})
	print('DOWNLOADING VIDEO')
	frames = dl.downloading_video(url)

	self.update_state(state="PROGRESS", 
					  meta={"current":20, 
					  		"total":100, 
					  		"status":"Processing video"})

	imgs = dl.create_imgs(frames)

	self.update_state(state="PROGRESS", 
					  meta={"current":30, 
					  		"total":100, 
					  		"status":"Using CNN to classify frames"})

	softmaxes = dl.classify_CNN(imgs)

	self.update_state(state="PROGRESS", 
					  meta={"current":80, 
					  		"total":100, 
					  		"status":"Using RNN to classify video"})

	brand = dl.brand2img(dl.classify_RNN(softmaxes))

	return {"current":100, "total":100, "status":"task completed", "result":brand}


@app.route('/', methods=['GET', 'POST'])
def index():
    brand = "https://www.shareicon.net/data/256x256/2015/10/02/110418_question_512x512.png"

    if request.method == 'GET':
        return render_template('index.html', brand=brand)

#     if request.method == 'POST':
#         text = request.form['text']
#         print(text)


#         #brand = text
# #        task = predict.apply_async(args=[text])
#         long_task.delay(text)

        # return jsonify({}), 202, {'Location': url_for('taskstatus',
        #                                           task_id=task.id)}

            #brand = dl.predict(text)
            #brand = brand2img(brand)
        # else:
        #     brand = "https://www.shareicon.net/data/256x256/2015/10/02/110418_question_512x512.png"

    #return render_template('index.html', brand=brand)
    return redirect(url_for('index'))


@app.route('/longtask', methods=['POST'])
def longtask():
    #url = request.args.post['text']
    #print(request.post)
    url = request.args.get('text', '')
    print("CHECKING URL: "+url)
    task = long_task.apply_async(args=[url])
    return jsonify({}), 202, {'Location': url_for('taskstatus',
                                                  task_id=task.id)}


@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = long_task.AsyncResult(task_id)
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



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=80)





