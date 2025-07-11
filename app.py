from flask import Flask, render_template, request, send_file
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid memory issues
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import os
import gc

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Max 10MB per upload

MAX_WIDTH = 800  # Max width for uploaded images to limit memory

def read_image_from_bytes(file_storage):
    file_bytes = np.asarray(bytearray(file_storage.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image

def resize_image(img):
    h, w = img.shape[:2]
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def detect_copy_move(img1, img2, match_ratio=0.7):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None:
        raise ValueError("No keypoints/descriptors found.")

    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < match_ratio * n.distance and kp1[m.queryIdx].pt != kp2[m.trainIdx].pt:
            good_matches.append(cv2.DMatch(m.queryIdx, m.trainIdx, m.distance))

    return img1_gray, img2_gray, kp1, kp2, good_matches

def visualize_matches(img1_gray, img2_gray, good_matches, kp1, kp2):
    matches = [[match] for match in good_matches]
    img_match = cv2.drawMatchesKnn(img1_gray, kp1, img2_gray, kp2, matches, None, flags=2)

    img_rgb = cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_rgb)
    ax.axis("off")

    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)

    gc.collect()  # Help free memory

    return buffer

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img1_file = request.files.get('image1')
        img2_file = request.files.get('image2')

        if not img1_file or not img2_file:
            return render_template('index.html', error="Please upload both images.")

        try:
            img1 = read_image_from_bytes(img1_file)
            img2 = read_image_from_bytes(img2_file)

            if img1 is None or img2 is None:
                raise ValueError("Could not decode one or both images.")

            img1 = resize_image(img1)
            img2 = resize_image(img2)

            img1_gray, img2_gray, kp1, kp2, good_matches = detect_copy_move(img1, img2)

            if good_matches:
                result_buffer = visualize_matches(img1_gray, img2_gray, good_matches, kp1, kp2)
                return send_file(result_buffer, mimetype='image/png')
            else:
                return render_template('index.html', error="No copy-move regions found.")

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

# Optional: Run only for local dev, not in production
# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5000))
#     app.run(debug=False, host='0.0.0.0', port=port)
