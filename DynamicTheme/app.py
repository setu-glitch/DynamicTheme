from flask import Flask, render_template, request
import cv2
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    img = request.files['img']
    clusters = 5
    img.save('img.jpg')
    image = cv2.imread("img.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    kmeans = KMeans(n_clusters = clusters)
    kmeans.fit(image)
    COLORS = kmeans.cluster_centers_
    colors = COLORS.astype(int)

    #print(colors)
    #plt.imshow([colors])
    #plt.savefig('/content/foo.png')
    fig = plt.imshow([colors])
    plt.axis('off')
    #plt.axes.get_xaxis().set_visible(False)
    #plt.axes.get_yaxis().set_visible(False)
    plt.savefig('static/images/foo.jpg', bbox_inches='tight', pad_inches = 0)
    #plt.show()

    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)