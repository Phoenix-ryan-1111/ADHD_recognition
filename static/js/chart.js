document.addEventListener('DOMContentLoaded', function() {
    // Sample data (first few and last few elements from the provided array)


    // Generate labels (indices) for the x-axis
    
    const harmonics_value = document.getElementById("harmonics_chart_value").value();
    
    const labels = Array.from({ length: harmonics_value.length }, (_, i) => i + 1);

    const ctx = document.getElementById('harmonics_chart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Array Values',
                data: harmonics_value,
                borderColor: '#5791e5',
                backgroundColor: 'rgba(87, 145, 229, 0.1)',
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Index'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Value'
                    },
                    ticks: {
                        callback: function(value, index, values) {
                            return value.toExponential(2);
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toExponential(6);
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });
});