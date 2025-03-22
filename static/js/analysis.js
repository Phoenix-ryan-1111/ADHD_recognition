document.addEventListener('DOMContentLoaded', () => {
    // Simple line chart drawing function
    function drawChart(canvas, data) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw line
        ctx.beginPath();
        ctx.strokeStyle = '#4A90E2';
        ctx.lineWidth = 2;
        
        data.forEach((value, index) => {
            const x = (index / (data.length - 1)) * width;
            const y = height - (value / 100 * height);
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
    }

    // Initialize charts
    const charts = document.querySelectorAll('.metric-chart');
    charts.forEach(canvas => {
        // Set canvas size
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        
        // Generate random data for demonstration
        const data = Array.from({length: 20}, () => Math.random() * 100);
        drawChart(canvas, data);
    });

    // Handle window resize
    window.addEventListener('resize', () => {
        charts.forEach(canvas => {
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
            const data = Array.from({length: 20}, () => Math.random() * 100);
            drawChart(canvas, data);
        });
    });
});