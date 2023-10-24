import numpy as np
import pywt
import cv2
from PIL import Image

def SuppWaveletFFT(img, gamma, order, wname, O):
    detail = img.astype(np.float32)
    y1, x1 = detail.shape

    horiz, verti, diag = [], [], []
    for _ in range(order):
        detail, (h, v, d) = pywt.dwt2(detail, wname)
        horiz.append(h)
        verti.append(v)
        diag.append(d)

    for n in range(order):
        if O in [1, 3]:
            fhoriz = np.fft.fftshift(np.fft.fft(horiz[n], axis=1), axes=1)
            y, x = fhoriz.shape
            hmask = (1 - np.exp(-np.arange(-x // 2, x // 2).astype(np.float32) ** 2 / gamma))
            fhoriz = fhoriz * hmask
            horiz[n] = np.fft.ifft(np.fft.ifftshift(fhoriz, axes=1), axis=1)

        if O in [2, 3]:
            fverti = np.fft.fftshift(np.fft.fft(verti[n], axis=0), axes=0)
            y, x = fverti.shape
            vmask = (1 - np.exp(-np.arange(-y // 2, y // 2).astype(np.float32) ** 2 / gamma))
            fverti = fverti * vmask[:, np.newaxis]
            verti[n] = np.fft.ifft(np.fft.ifftshift(fverti, axes=0), axis=0)

    for n in reversed(range(order)):
        detail = detail[:horiz[n].shape[0], :horiz[n].shape[1]]
        detail = pywt.idwt2((detail, (horiz[n], verti[n], diag[n])), wname)

    return detail[:y1, :x1]

# Read PNG image
# image = cv2.imread('Vessels.png', cv2.IMREAD_COLOR)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = image.astype(float) / 255.0

# Read TIFF image
image = Image.open('./Images/Four.tif')
image.seek(0)
image = image.copy()

image = np.array(image)
image = image / image.max()

# Run filtering
gamma = 10
order = 4
wname = 'db20'
reps = 2

O = 3

for r in range(reps):
    image_filtered = SuppWaveletFFT(image, gamma, order, wname, O)
    image = np.minimum(image_filtered, image)

image = image.astype(float)

# Display result
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
