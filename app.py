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

def send_progress_update(stage, percentage):
    return f"data: {json.dumps({'type': 'progress', 'stage': stage, 'percentage': percentage})}\n\n"

def send_result(result):
    return f"data: {json.dumps({'type': 'result', 'result': result})}\n\n"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return Response(
                send_result({'success': False, 'message': 'No file part'}),
                mimetype='text/event-stream'
            )
        
        file = request.files['file']
        if file.filename == '':
            return Response(
                send_result({'success': False, 'message': 'No selected file'}),
                mimetype='text/event-stream'
            )
        
        # Check file extension
        allowed_extensions = {'mp3', 'wav'}
        if not '.' in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return Response(
                send_result({'success': False, 'message': 'Invalid file type. Only MP3 and WAV files are allowed.'}),
                mimetype='text/event-stream'
            )
        
        # Check file size
        file_size = len(file.read())
        file.seek(0)  # Reset file pointer
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            return Response(
                send_result({'success': False, 'message': f'File too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'}),
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
                # Send upload progress
                yield send_progress_update('upload', 20)
                time.sleep(1)
                
                # Process audio file
                yield send_progress_update('splitting', 30)
                features_df = create_predict_data.process_audio_files(filepath)
                time.sleep(2)
                
                yield send_progress_update('resampling', 50)
                time.sleep(2)
                
                yield send_progress_update('feature_extraction', 70)
                time.sleep(2)
                
                # Make prediction
                yield send_progress_update('analysis', 90)
                time.sleep(1)
                result = predict.predict_adhd(features_df)
                print(result)
                
                # Send final result
                yield send_result(result)
                
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
