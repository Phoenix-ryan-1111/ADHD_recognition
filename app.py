from flask import Flask, request, jsonify, render_template, Response
import os
from werkzeug.utils import secure_filename
import create_predict_data
import predict
import json
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 100MB max file size
app.config['MAX_CONTENT_PATH'] = 255  # Maximum length of file path

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def send_estimate_time(estimate_time):
    return f"data: {json.dumps({'type': 'estimate', 'estimate_time': estimate_time})}\n\n"

def send_result(result,sucess):
    return f"data: {json.dumps({'type': 'result','sucess':sucess, 'result': result})}\n\n"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return Response(
                send_result({'success': False, 'result': 'No file part'}),
                mimetype='text/event-stream'
            )
        
        file = request.files['file']
        if file.filename == '':
            return Response(
                send_result({'success': False, 'result': 'No selected file'}),
                mimetype='text/event-stream'
            )
        
        # Check file extension
        allowed_extensions = {'mp3', 'wav'}
        if not '.' in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return Response(
                send_result({'success': False, 'result': 'Invalid file type. Only MP3 and WAV files are allowed.'}),
                mimetype='text/event-stream'
            )
        
        # Check file size
        file_size = len(file.read())
        file.seek(0)  # Reset file pointer
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            return Response(
                send_result({'success': False, 'result': f'File too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'}),
                mimetype='text/event-stream'
            )
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Check if file path is too long
        if len(filepath) > app.config['MAX_CONTENT_PATH']:
            return Response(
                send_result({'success': False, 'message': 'File path too long'}),
                mimetype='text/event-stream'
            )
        
        file.save(filepath)
        
        def generate():
            try:
                estimate_time = 2*(create_predict_data.split_number(filepath) - 1) if create_predict_data.split_number(filepath)>=2 else 3
                yield send_estimate_time(estimate_time)
                # Process audio file
                features_df = create_predict_data.process_audio_files(filepath)

                result = predict.predict_adhd(features_df)
                print(result)

                yield send_result(result,True)
                
            except Exception as e:
                yield send_result({
                    'success': False,
                    'message': f'Error processing file: {str(e)}'
                })
            finally:
                # Clean up
                if os.path.exists(filepath):
                    os.remove(filepath)
        
        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        return Response(
            send_result({'success': False, 'message': f'Server error: {str(e)}'}),
            mimetype='text/event-stream'
        )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
