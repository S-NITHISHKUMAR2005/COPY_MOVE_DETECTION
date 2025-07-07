from flask import Flask, render_template, request, send_file
import cv2
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Max 10MB per upload

def read_image_from_bytes(file_storage):
    file_bytes = np.asarray(bytearray(file_storage.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image

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

    # Convert to RGB for matplotlib
    img_rgb = cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis("off")

    # Save to BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
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

            img1_gray, img2_gray, kp1, kp2, good_matches = detect_copy_move(img1, img2)

            if good_matches:
                result_buffer = visualize_matches(img1_gray, img2_gray, good_matches, kp1, kp2)
                return send_file(result_buffer, mimetype='image/png')
            else:
                return render_template('index.html', error="No copy-move regions found.")

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')
# if __name__ == '__main__':
#     app.run(debug=True)
if __name__ == '__main__':
    app.run()
