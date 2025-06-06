// Main JavaScript file for Music Mood Recommender

// Handle rating hover effects
function setupRatingHover() {
    document.querySelectorAll('.rating-btn').forEach(btn => {
        btn.addEventListener('mouseover', function() {
            const rating = parseInt(this.value);
            const parent = this.parentElement;
            
            parent.querySelectorAll('.rating-btn').forEach((btn, index) => {
                if (index < rating) {
                    btn.classList.add('btn-warning');
                    btn.classList.remove('btn-outline-warning');
                } else {
                    btn.classList.add('btn-outline-warning');
                    btn.classList.remove('btn-warning');
                }
            });
        });
        
        btn.addEventListener('mouseout', function() {
            const parent = this.parentElement;
            parent.querySelectorAll('.rating-btn').forEach(btn => {
                btn.classList.add('btn-outline-warning');
                btn.classList.remove('btn-warning');
            });
        });
    });
}

// Check loading status
function checkLoadingStatus() {
    if (document.getElementById('loading-spinner')) {
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                if (!data.is_loading) {
                    window.location.href = '/';
                }
            });
    }
}

// Document ready
document.addEventListener('DOMContentLoaded', function() {
    setupRatingHover();
    
    // Setup loading check if on loading page
    if (document.getElementById('loading-spinner')) {
        setInterval(checkLoadingStatus, 2000);
    }
});