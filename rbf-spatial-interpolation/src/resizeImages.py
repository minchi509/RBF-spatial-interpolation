import numpy as np
import cv2
import scipy.spatial.distance
import matplotlib.pyplot as plt
import os
from sklearn.utils.validation import column_or_1d
from sympy import sympify

def getKernelFunc(kernelName='thin_plate_spline', epsilon=1.0):
    if kernelName == 'gaussian':
        return lambda r: np.exp(-(epsilon * r) ** 2)
    elif kernelName == 'multiquadric':
        return lambda r: np.sqrt((epsilon * r) ** 2 + 1)
    elif kernelName == 'inverse_multiquadric':
        return lambda r: 1.0 / np.sqrt((epsilon * r) ** 2 + 1)
    elif kernelName == 'thin_plate_spline':
        return lambda r: np.where(r > 0, r**(2) * np.log(r), 0.0)
    elif  kernelName == 'linear':
        return lambda r: r
    elif kernelName == 'cubic':
        return lambda r: r ** 3

    else:
        raise ValueError(f"Unknown kernel: {kernelName}")

def constructPolynomialBasis(coords, degree):
    if degree < 0:
        return None

    num_terms = (degree + 1) * (degree + 2) // 2
    x = coords[:, 0]
    y = coords[:, 1]
    N = len(x)

    basisMatrix = np.empty((N, num_terms))
    termIndex = 0
    
    for k in range(degree + 1):
        for i in range(k + 1):
            j = k - i
            basisMatrix[:, termIndex] = (x**i) * (y**j)
            termIndex += 1

    return basisMatrix

def RBFSolve(kernel, degree, coords, channelColorVal):
    N = coords.shape[0] 

    distances = scipy.spatial.distance.cdist(coords, coords, metric='euclidean')
    Phi = kernel(distances)
    P = constructPolynomialBasis(coords, degree)
    del distances

    if P is None:
        A = Phi
        C = channelColorVal

    else:
        mHat = P.shape[1]
        O = np.zeros((mHat, mHat))
        A_upper = np.hstack((Phi, P))
        A_lower = np.hstack((P.T, O))
        A = np.vstack((A_upper, A_lower))
        C = np.concatenate((channelColorVal, np.zeros(mHat)))

    del Phi

    b = scipy.linalg.solve(A, C)
    lambda_ = b[:N]
    gamma = b[N:] if P is not None else None

    return (lambda_, gamma)


def chunkedCoordsGenerator(newWidth, newHeight, chunkSize=64):
    xNewAll = np.arange(newWidth)
    yNewAll = np.arange(newHeight)

    for yStart in range(0, newHeight, chunkSize):
        for xStart in range(0, newWidth, chunkSize):
            yEnd = min(yStart + chunkSize, newHeight)
            xEnd = min(xStart + chunkSize, newWidth)

            chunk_x = xNewAll[xStart:xEnd]
            chunk_y = yNewAll[yStart:yEnd]
            X, Y = np.meshgrid(chunk_x, chunk_y)

            chunk_coords = np.stack((X.ravel(), Y.ravel()), axis=1)
            yield chunk_coords, (yStart, yEnd, xStart, xEnd)


def RBFPredict(kernel, lambda_, gamma, degree, coords, newCoords):
    distanceMatrix = scipy.spatial.distance.cdist(newCoords, coords, metric='euclidean')
    Phi = kernel(distanceMatrix)
    del distanceMatrix

    prediction = np.dot(Phi, lambda_)
    del Phi

    if gamma is not None:
        P = constructPolynomialBasis(newCoords, degree)
        prediction += np.dot(P, gamma)

    return prediction


def getEpsilonFromInput(defaultEpsilon=100):
    try:
        eps_input = float(sympify(input(f">> Nhập Epsilon (Mặc định {defaultEpsilon}): ")).evalf())
        epsilon = float(eps_input) if eps_input else defaultEpsilon
    except Exception: 
        print(f"Lỗi nhập! Dùng eps={defaultEpsilon}")
        epsilon = defaultEpsilon
    return epsilon

def RBFPredictChunked(
    kernel, lambda_, gamma,
    polynomialDegree, coordsTrain, scaleFactor,
    targetWidth, targetHeight, targetArray, channelIdx
):
    for newCoordsChunk, (yStart, yEnd, xStart, xEnd) in chunkedCoordsGenerator(targetWidth, targetHeight):
        xSource_chunk = newCoordsChunk[:, 0] / targetWidth
        ySource_chunk = newCoordsChunk[:, 1] / targetHeight
        newSrcCoords = np.stack((xSource_chunk, ySource_chunk), axis=1)

        prediction = RBFPredict(kernel, lambda_, gamma, polynomialDegree, coordsTrain, newSrcCoords)
        prediction = np.clip(prediction, 0, 255).astype(np.uint8)
        
        chunkHeight = yEnd - yStart
        chunkWidth = xEnd - xStart
        targetArray[yStart:yEnd, xStart:xEnd, channelIdx] = prediction.reshape(chunkHeight, chunkWidth)

def main():
    IMAGE_PATH = input('Nhập path đến ảnh: ')
    OUTPUT_FILENAME = "RBF_result.jpg"

    if not os.path.exists(IMAGE_PATH):
        exit(f"Lỗi: Không tìm thấy file ảnh '{IMAGE_PATH}'")

    scaleFactor = float(sympify(input('Nhập scaleFactor: ')).evalf())
    print(f"--- Đang đọc ảnh: {IMAGE_PATH} ---")
    img = cv2.imread(IMAGE_PATH)
    inputHeight, inputWidth, channelCount = img.shape
    
    targetHeight, targetWidth = round(inputHeight * scaleFactor), round(inputWidth * scaleFactor)
    print(f"Kích thước ảnh gốc: {inputHeight}x{inputWidth}\n -> Kích thước ảnh đích: {targetWidth}x{targetHeight} (Scale: x{scaleFactor})")

    # Tạo tọa độ các điểm ảnh
    x = np.arange(inputWidth) / inputWidth
    y = np.arange(inputHeight) / inputHeight
    X, Y = np.meshgrid(x, y)
    coordsTrain = np.column_stack([X.ravel(), Y.ravel()])

    print("\n" + "="*40)
    print("   CHỌN NHÂN RBF   ")
    print("="*40)
    print("1. Gaussian.")
    print("2. Thin Plate Spline.")
    print("3. Multiquadric.")
    print("4. Inverse Multiquadric.")
    print("5. Linear.")
    print("6. Cubic.")

    choice = int(input(">> Nhập số thứ tự (1-6): "))
    choiceToKernelName = {
        1: 'gaussian', 2: 'thin_plate_spline', 3: 'multiquadric',
        4: 'inverse_multiquadric', 5: 'linear', 6: 'cubic',
    }
    doesKernelNeedEpsilon = {
        'gaussian': True, 'multiquadric': True, 'inverse_multiquadric': True,
        'thin_plate_spline': False, 'linear': False, 'cubic': False
    }

    try:
        kernelName = choiceToKernelName[choice]
        kernelNeedsEpsilon = doesKernelNeedEpsilon[kernelName]
        if kernelNeedsEpsilon: epsilon = getEpsilonFromInput()
        else: epsilon = -1
    except KeyError:
        kernelName = 'gaussian'
        epsilon = 100.0
        print(f"Lựa chọn không hợp lệ. Sử dụng mặc định: Gaussian (eps={epsilon})")

    kernelPredeterminedPolDegree = {
        'linear': 1, 'cubic': 2,
        'thin_plate_spline': 1, 'multiquadric': 1
    }
    if kernelName in kernelPredeterminedPolDegree:
        polynomialDegree = kernelPredeterminedPolDegree[kernelName]
    else:
        polynomialDegree = int(input("Nhập bậc đa thức cộng thêm: "))

    kernel = getKernelFunc(kernelName, epsilon)
    rbf_result = np.zeros((targetHeight, targetWidth, channelCount), dtype=np.uint8)

    channelIdxToName = {0: 'Blue', 1: 'Green', 2: 'Red', 4: 'Alpha'}
    print('Đang nội suy ảnh...')
    for channelIdx in range(channelCount):
        channelVals = img[..., channelIdx].ravel().astype(np.float64)
        print(f'Đang xây dựng hàm nội suy cho kênh màu \'{channelIdxToName[channelIdx]}\'...')
        channelLambda, channelGamma = RBFSolve(kernel, polynomialDegree, coordsTrain, channelVals)
        print(f'Đang nội suy điểm ảnh cho kênh màu \'{channelIdxToName[channelIdx]}\'...')
        RBFPredictChunked(
            kernel, channelLambda, channelGamma, polynomialDegree,
            coordsTrain, 
            scaleFactor, targetWidth, targetHeight, rbf_result, channelIdx
        )

    # --- Bước 4: So sánh với Bicubic ---
    print("\n--- Đang chạy Bicubic để đối chứng... ---")
    bicubic_result = cv2.resize(img, (targetWidth, targetHeight), interpolation=cv2.INTER_CUBIC)
    bicubic_result = cv2.cvtColor(bicubic_result, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rbf_result = cv2.cvtColor(rbf_result, cv2.COLOR_BGR2RGB)

    print("--- Hoàn tất! Đang hiển thị kết quả... ---")
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f"Input ({inputWidth}x{inputHeight})")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(bicubic_result)
    plt.title("Bicubic Spline")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(rbf_result)
    plt.title(f"{kernelName}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # Lưu file
    rbf_result = cv2.cvtColor(rbf_result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(OUTPUT_FILENAME, rbf_result)
    print(f"Đã lưu ảnh kết quả vào: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    main()
