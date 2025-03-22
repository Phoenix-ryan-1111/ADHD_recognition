document.addEventListener('DOMContentLoaded', () => {
    const recordButton = document.getElementById('recordButton');
    const uploadButton = document.getElementById('uploadButton');
    const fileInput = document.getElementById('file_input');
    const analysisButton = document.getElementById('analysisButton');
    const progressCircle = document.querySelector('.progress');
    const percentageText = document.querySelector('.percentage');
    const timerText = document.querySelector('.timer');

    let mediaRecorder;
    let audioChunks = [];
    let startTime;
    let timerInterval;
    const MAX_DURATION = 60000; // 2 minutes in milliseconds

    // Request microphone access
    async function setupRecorder() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                analysisButton.disabled = false;
                audioChunks = [];
                 // Confirm before uploading
                const confirmUpload = confirm('Do you want to upload the recorded audio?');
                if (confirmUpload) {
                    uploadAudio(audioBlob); // Call the upload function
                } else {
                    console.log('Upload canceled by user.');
                }
            };
        } catch (err) {
            console.error('Error accessing microphone:', err);
            alert('Unable to access microphone. Please ensure you have granted permission.');
        }
    }


    // Function to upload audio to the backend
    async function uploadAudio(audioBlob) {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav'); // Append the file with a name

        try {
            const response = await fetch('/upload', { 
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const result = await response.json(); // Parse JSON response
                console.log('Audio uploaded successfully:', result);
                alert('Audio uploaded successfully!');
            } else {
                console.error('Failed to upload audio:', response.statusText);
                alert('Failed to upload audio. Please try again.');
            }
        } catch (error) {
            console.error('Error uploading audio:', error);
            alert('Error uploading audio. Please check your network connection.');
        }
    }

    // Update progress and timer
    function updateProgress(elapsed) {
        const progress = Math.min((elapsed / MAX_DURATION) * 100, 100);
        const dashOffset = 283 - (283 * progress / 100);
        progressCircle.style.strokeDashoffset = dashOffset;
        percentageText.textContent = `${Math.round(progress)}%`;
        
        const seconds = Math.floor(elapsed / 1000);
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        timerText.textContent = 
            `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    // Record button events
    recordButton.addEventListener('mousedown', () => {
        setupRecorder().then(() => {
            audioChunks = [];
            mediaRecorder.start();
            startTime = Date.now();
            
            timerInterval = setInterval(() => {
                const elapsed = Date.now() - startTime;
                updateProgress(elapsed);
                
                if (elapsed >= MAX_DURATION) {
                    mediaRecorder.stop();
                    clearInterval(timerInterval);
                }
            }, 100);
        });
    });

    recordButton.addEventListener('mouseup', () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            clearInterval(timerInterval);
        }
    });

    // File upload handling
    uploadButton.addEventListener('click', () => {
        fileInput.click();
    });

    // fileInput.addEventListener('change', (e) => {
    //     if (e.target.files.length > 0) {
    //         analysisButton.disabled = false;
    //     }
    // });

    // // Analysis button
    // analysisButton.addEventListener('click', () => {
    //     window.location.href = 'analysis.html';
    // });

    // JavaScript to handle file selection  
    fileInput.addEventListener('change', function(e) {  
        if (e.target.files.length > 0) {
            analysisButton.disabled = false;
            const fileName = this.files[0] ? this.files[0].name : 'No file chosen'; 
            document.getElementById('fileName').textContent = fileName; 
        }
    });  

    // Handle form submission  
    document.getElementById('uploadForm').addEventListener('submit', function(event) {  
        const fileInput = document.getElementById('file_input');  
        if (!fileInput.files.length) {  
            alert('Please select a file before submitting.');  
            event.preventDefault(); // Prevent form submission  
        }  

        analysisButton.classList.add('loading');
        analysisButton.disabled = true;

        // Simulate analysis process (replace with actual analysis logic)
        setTimeout(() => {
            analysisButton.classList.remove('loading');
            analysisButton.disabled = false;
            // window.location.href = 'analysis.html';
        }, 20000); // Simulating 3 seconds of analysis
    });  

});



