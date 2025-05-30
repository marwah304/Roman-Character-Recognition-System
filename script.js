document.getElementById('imageInput').addEventListener('change', function (event) {
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const file = event.target.files[0];

    if (file) {
        previewImage.src = URL.createObjectURL(file);
        previewContainer.style.display = 'block';
    } else {
        previewContainer.style.display = 'none';
    }
});
