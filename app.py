from flask import Flask, render_template, request
import importlib
import predict_model

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)  # 서버에 파일 저장
    predict = importlib.import_module("predict_model")
    result = predict.video_predict(uploaded_file.filename)
    print(result)
    return "asdf"

if __name__ == '__main__':
    app.run(debug=True)