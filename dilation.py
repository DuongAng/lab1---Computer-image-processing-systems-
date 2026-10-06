"""Lab 1, variant 6: phép giãn (dilation) ảnh nhị phân, kernel 3x3.

Chạy:  python dilation.py            -> dùng ảnh test tự sinh
       python dilation.py anh.jpg    -> dùng ảnh của mình
"""
import os
import sys
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt


def dilate_native(img):
    """Tự viết tay: mỗi pixel = giá trị lớn nhất trong ô 3x3 quanh nó."""
    h, w = img.shape
    src = img.tolist()
    dst = [[0] * w for _ in range(h)]
    for y in range(h):
        rows = src[max(y - 1, 0):y + 2]
        for x in range(w):
            x0 = max(x - 1, 0)
            dst[y][x] = max(max(r[x0:x + 2]) for r in rows)
    return np.array(dst, np.uint8)


def dilate_opencv(img):
    """Dùng hàm có sẵn của OpenCV."""
    return cv2.dilate(img, np.ones((3, 3), np.uint8))


def binarize(gray):
    """Nhị phân hóa bằng Otsu (đề cho phép không tự viết)."""
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def test_image(n=512):
    """Ảnh test: chữ, hình tròn, hình chữ nhật, đường thẳng và nhiễu chấm trắng."""
    img = np.zeros((n, n), np.uint8)
    k = n / 512
    p = lambda *v: tuple(int(i * k) for i in v)       # đổi tọa độ theo kích thước ảnh
    cv2.putText(img, "ITMO", p(60, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.5 * k, 255, max(1, int(3 * k)))
    cv2.circle(img, p(130, 330), int(70 * k), 255, max(1, int(2 * k)))
    cv2.rectangle(img, p(260, 260), p(440, 400), 255, 1)
    cv2.line(img, p(30, 470), p(480, 440), 255, 1)
    img[np.random.default_rng(0).random(img.shape) > 0.997] = 255
    return img


def measure(func, img, repeats):
    """Thời gian chạy nhanh nhất trong `repeats` lần, tính bằng mili giây."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        func(img)
        best = min(best, time.perf_counter() - start)
    return best * 1000


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # 1. Đọc ảnh và nhị phân hóa
    gray = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE) if len(sys.argv) > 1 else test_image()
    if gray is None:
        sys.exit("Cannot open image")
    binary = binarize(gray)

    # 2. Giãn ảnh theo hai cách, kiểm tra kết quả có giống nhau không
    native, opencv = dilate_native(binary), dilate_opencv(binary)
    print("Results are identical:", np.array_equal(native, opencv))
    print("White pixels before:", np.count_nonzero(binary), "- after:", np.count_nonzero(opencv))
    # Lưới 2x2: ảnh xám gốc, ảnh nhị phân / kết quả tự viết, kết quả OpenCV
    cv2.imwrite("results/comparison.png", np.vstack([np.hstack([gray, binary]), np.hstack([native, opencv])]))

    # 3. Đo tốc độ trên nhiều kích thước ảnh
    sizes = [128, 256, 512, 1024]
    t_native, t_opencv = [], []
    print("\nSize | Native Python, ms | OpenCV, ms | Speedup")
    for n in sizes:
        img = binarize(test_image(n))
        t_native.append(measure(dilate_native, img, 3))
        t_opencv.append(measure(dilate_opencv, img, 200))
        print(f"{n}x{n} | {t_native[-1]:.1f} | {t_opencv[-1]:.4f} | {t_native[-1] / t_opencv[-1]:.0f}x")

    # 4. Vẽ biểu đồ
    plt.plot(sizes, t_native, "o-", label="Native Python")
    plt.plot(sizes, t_opencv, "s-", label="OpenCV")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xticks(sizes, [f"{n}x{n}" for n in sizes])
    plt.xlabel("Размер изображения")
    plt.ylabel("Время, мс")
    plt.title("Дилатация 3x3: время выполнения")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("results/benchmark.png", dpi=130, bbox_inches="tight")
