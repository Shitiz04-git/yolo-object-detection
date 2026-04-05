from flask import Flask, render_template, Response
from detector import generate_frames, person_count

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', count=person_count)

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)