from flask import Flask, render_template, send_from_directory
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/manifest.json')
def manifest():
    return send_from_directory('.', 'manifest.json')

@app.route('/service-worker.js')
def service_worker():
    return send_from_directory('.', 'service-worker.js')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    # Lanza streamlit en segundo plano
    subprocess.Popen(["streamlit", "run", "app.py"])
    app.run(debug=True, use_reloader=False)
