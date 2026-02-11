const input = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const uploadForm = document.getElementById("uploadForm");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const error = document.getElementById("error");

// Image preview
input.addEventListener("change", function() {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        preview.style.display = "block";
        reader.addEventListener("load", function() {
            preview.setAttribute("src", this.result);
        });
        reader.readAsDataURL(file);
    }
});

// Form submission with AJAX
uploadForm.addEventListener("submit", async function(e) {
    e.preventDefault();
    
    const file = input.files[0];
    if (!file) {
        error.textContent = "Please select an image";
        error.style.display = "block";
        return;
    }
    
    // Hide previous results and show loading
    result.style.display = "none";
    error.style.display = "none";
    loading.style.display = "block";
    
    // Prepare form data
    const formData = new FormData();
    formData.append("image", file);
    
    try {
        // Send request to server
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });
        
        const data = await response.json();
        loading.style.display = "none";
        
        if (data.success) {
            // Display result
            document.getElementById("resultImg").src = data.image_path;
            document.getElementById("predictionText").textContent = data.prediction;
            document.getElementById("confidenceText").textContent = `Confidence: ${data.confidence}%`;
            result.style.display = "block";
        } else {
            // Display error
            error.textContent = data.error || "An error occurred";
            error.style.display = "block";
        }
    } catch (err) {
        loading.style.display = "none";
        error.textContent = "Connection error: " + err.message;
        error.style.display = "block";
    }
});
