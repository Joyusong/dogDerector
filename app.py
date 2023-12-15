from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)  # 서버에 파일 저장
    return 'File has been uploaded successfully'

if __name__ == '__main__':
    app.run(debug=True)
