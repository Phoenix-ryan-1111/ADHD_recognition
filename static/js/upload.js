document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file_input');
    const uploadButton = document.getElementById('uploadButton');
    const analysisButton = document.getElementById('analysisButton');
    const resultButton = document.getElementById('resultButton');
    const fileName = document.getElementById('fileName');
    const uploadForm = document.getElementById('uploadForm');
    const progressCircle = document.querySelector('.progress');
    const percentageText = document.querySelector('.percentage');
    const timerElement = document.querySelector('.timer');
    const resultsPortal = document.getElementById('resultsPortal');
    const resultsContent = document.getElementById('resultsContent');
    const overlay = document.getElementById('overlay');
    
    let uploadStartTime;
    let processingStartTime;
    let timerInterval;
    let analysisResult = null;
    
    // Update progress circle
    function updateProgress(percentage, stage) {
        const radius = progressCircle.r.baseVal.value;
        const circumference = radius * 2 * Math.PI;
        const offset = circumference - (percentage / 100) * circumference;

        progressCircle.style.strokeDasharray = `${circumference} ${circumference}`;
        progressCircle.style.strokeDashoffset = offset;
        percentageText.textContent = `${Math.round(percentage)}%`;
        
        // Update progress circle color based on stage
        switch(stage) {
            case 'upload':
                progressCircle.style.stroke = '#4CAF50';
                break;
            case 'splitting':
                progressCircle.style.stroke = '#4CAF50';
                break;
            case 'resampling':
                progressCircle.style.stroke = '#4CAF50';
                break;
            case 'feature_extraction':
                progressCircle.style.stroke = '#4CAF50';
                break;
            case 'analysis':
                progressCircle.style.stroke = '#4CAF50';
                break;
            case 'complete':
                progressCircle.style.stroke = '#4CAF50';
                break;
        }
    }
    
    // Update timer
    function updateTimer() {
        const currentTime = new Date().getTime();
        const elapsedTime = currentTime - uploadStartTime;
        const seconds = Math.floor(elapsedTime / 1000);
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        timerElement.textContent = `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
    
    // Start timer
    function startTimer() {
        uploadStartTime = new Date().getTime();
        timerInterval = setInterval(updateTimer, 1000);
    }
    
    // Stop timer
    function stopTimer() {
        clearInterval(timerInterval);
    }

    // Show results in portal
    function showResults(result) {
        let content = '';
        if (result.success) {
            content = `
                <div class="result-item">
                    <strong>${result.prediction}</strong> 
                </div>
                <div class="result-item">
                    <strong>${result.probability}</strong> 
                </div>
            `;
        } else {
            content = `
                <div class="result-item" style="color: #ff0000;">
                    <strong>Error:</strong> ${result.message}
                </div>
            `;
        }
        resultsContent.innerHTML = content;
        resultsPortal.style.display = 'block';
        overlay.style.display = 'block';
    }

    // Close results portal
    window.closeResults = function() {
        resultsPortal.style.display = 'none';
        overlay.style.display = 'none';
    }
    
    // Handle file selection
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            fileName.textContent = this.files[0].name;
            analysisButton.disabled = false;
            resultButton.style.display = 'none'; // Hide result button when new file is selected
        } else {
            fileName.textContent = 'No file chosen';
            analysisButton.disabled = true;
            resultButton.style.display = 'none';
        }
    });
    
    // Handle upload button click
    uploadButton.addEventListener('click', function() {
        fileInput.click();
    });

    // Handle result button click
    resultButton.addEventListener('click', function() {
        if (analysisResult) {
            showResults(analysisResult);
        }
    });
    
    // Handle form submission
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        // Reset UI
        updateProgress(0, 'upload');
        startTimer();
        analysisButton.disabled = true;
        uploadButton.disabled = true;
        resultButton.style.display = 'none'; // Hide result button during analysis
        
        try {
            // Send file to server
            const response = await fetch('/upload_file', {
                method: 'POST',
                body: formData
            });
            // Handle server-sent events for progress updates
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            while (true) {
                const {value, done} = await reader.read();
                if (done) break;
                
                const text = decoder.decode(value);
                const lines = text.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.slice(6));
                        console.log(data.stage)
                        if (data.type === 'progress') {
                            updateProgress(data.percentage, data.stage);
                        } else if (data.type === 'result') {
                            console.log(data.result)
                            analysisResult = data.result;
                            resultButton.style.display = 'block';
                            updateProgress(100, 'complete');
                            stopTimer();
                        }
                    }
                }
            }
            
        } catch (error) {
            analysisResult = {
                success: false,
                message: 'Error processing file: ' + error.message
            };
            resultButton.style.display = 'block';
        } finally {
            // Reset UI
            analysisButton.disabled = false;
            uploadButton.disabled = false;
            fileInput.value = '';
            fileName.textContent = 'No file chosen';
        }
    });
}); 