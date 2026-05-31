from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata # Dùng để nội suy địa hình nền cho đẹp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sympy import sympify

class RBFInterpolator:
    def __init__(self, kernel='gaussian', epsilon=1.0, degree=1):
        """
        kernel: 'gaussian', 'thin_plate_spline', 'multiquadric', 'inverse_multiquadric'
        epsilon: Shape parameter (quan trọng cho Gaussian/Multiquadric)
        degree: Bậc của đa thức cộng thêm (thường là 1 cho Thin Plate Spline)
        """
        self.kernel_name = kernel
        self.epsilon = epsilon
        self.degree = degree
        self.lambda_ = None
        self.gamma = None
        self.xMin = None
        self.xMax = None
        self.centers = None # Lưu lại để tính khoảng cách khi predict

    def makeKernelFunc(self, r):
        if self.kernel_name == 'gaussian':
            return np.exp(-(self.epsilon * r) ** 2)

        elif self.kernel_name == 'multiquadric':
            return np.sqrt((self.epsilon * r) ** 2 + 1)

        elif self.kernel_name == 'inverse_multiquadric':
            return 1.0 / np.sqrt((self.epsilon * r) ** 2 + 1)

        elif self.kernel_name == 'thin_plate_spline':
            # r = 0 thì trả về 0, tránh log(0)
            res = np.zeros_like(r)
            mask = r > 0
            res[mask] = (r[mask] ** 2) * np.log(r[mask])
            return res

        elif self.kernel_name == 'linear':
            return r

        elif self.kernel_name == 'cubic':
            return r ** 3

        else:
            raise ValueError(f"Nhân không được hỗ trợ: {self.kernel_name}")

    def constructPolynomialBasis(self, X):
        if self.degree < 0:
            return None

        d = self.degree
        num_terms = (d + 3) * (d + 2) * (d + 1) // 6

        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]
        N = len(x)

        basisMatrix = np.empty((N, num_terms), dtype=X.dtype)
        termIndex = 0
        
        for total_degree in range(self.degree + 1):
            for i in range(total_degree + 1):
                for j in range(total_degree - i + 1):
                    k = total_degree - i - j
                    basisMatrix[:, termIndex] = (x**i) * (y**j) * (z**k)
                    termIndex += 1

        return basisMatrix

    def fit(self, X, y):
        # 1. Chuẩn hóa Min-Max (Quan trọng nhất cho bài toán trộn lẫn đơn vị)
        self.xMin = np.min(X)
        self.xMax = np.max(X)
        X_norm = (X - self.xMin) / (self.xMax - self.xMin + 1e-8) # +1e-8 tránh chia 0
        self.centers = X_norm

        N = X_norm.shape[0]

        # 2. Tính ma trận khoảng cách và ma trận Phi
        dists = cdist(X_norm, X_norm)
        Phi = self.makeKernelFunc(dists)

        # 3. Xây dựng phần đa thức
        P = self.constructPolynomialBasis(X_norm)

        if P is None:
            # Hệ đơn giản: Phi * lambda = y
            A = Phi
            C = y

        else:
            # Hệ mở rộng 
            # [ Phi  P ] [ lambda ] = [ y ]
            # [ P.T  O ] [ gamma  ]   [ 0 ]
            mHat = P.shape[1]
            top = np.hstack([Phi, P])
            bot = np.hstack([P.T, np.zeros((mHat, mHat))])
            A = np.vstack([top, bot])
            C = np.concatenate([y, np.zeros(mHat)])

        # Giải hệ phương trình dày đặc
        coeffs = scipy.linalg.solve(A, C, assume_a='sym')
        self.lambda_ = coeffs[:N]
        self.gamma = coeffs[N:]

    def predict(self, X_new):
        X_new_norm = (X_new - self.xMin) / (self.xMax - self.xMin + 1e-8)
        distanceMatrix = cdist(X_new_norm, self.centers)
        Phi = self.makeKernelFunc(distanceMatrix)

        prediction = Phi @ self.lambda_
        P = self.constructPolynomialBasis(X_new_norm)
        if P is not None and self.gamma is not None:
            prediction += P @ self.gamma

        return prediction


def visualize_results(model, X_raw, y_raw, title_suffix=""):
    CMAP = 'gnuplot2'
    lat = X_raw[:, 0]
    lon = X_raw[:, 1]
    elev = X_raw[:, 2]

    grid_lon_pts = np.linspace(lon.min(), lon.max(), 50)
    grid_lat_pts = np.linspace(lat.min(), lat.max(), 50)
    xx, yy = np.meshgrid(grid_lat_pts, grid_lon_pts)

    # Vấn đề: Ta cần Elevation (z) cho lưới để dự đoán Nhiệt độ.
    # Giải pháp: Nội suy tuyến tính đơn giản Elevation từ dữ liệu gốc ra lưới
    # (Để tạo cái nền địa hình cho RBF chạy trên đó)
    zz = griddata((lat, lon), elev, (xx, yy), method='linear')

    # Xử lý vùng NaN (ngoài biên) bằng nearest để lấp đầy hình
    mask = np.isnan(zz)
    zz[mask] = griddata((lat, lon), elev, (xx[mask], yy[mask]), method='nearest')

    # --- DỰ BÁO NHIỆT ĐỘ ---
    # Làm phẳng lưới để đưa vào predict
    X_grid = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    T_pred = model.predict(X_grid).reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(
        xx, yy, T_pred, 
        vmin=y_raw.min(), vmax=y_raw.max(),
        levels=50, cmap=CMAP, alpha=0.8
    )
    plt.colorbar(contour, label=r'Nhiệt độ dự báo ($^\circ$C)')
    plt.scatter(
        lat, lon, c=y_raw,
        cmap=CMAP, edgecolors='k', s=80, label='Trạm đo'
    )

    plt.title(f'Bản đồ nhiệt 2D (RBF) - {title_suffix}')
    plt.ylabel('Kinh độ (Longitude)')
    plt.xlabel('Vĩ độ (Latitude)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.get_cmap(CMAP)
    norm = Normalize(vmin=T_pred.min(),vmax=T_pred.max(), clip=True)
    facecolors = cmap(norm(T_pred))
    surf = ax.plot_surface(
        xx, yy, zz,
        cmap=CMAP,
        facecolors=facecolors,
        rstride=1, cstride=1, shade=False, alpha=0.8
    )

    p = ax.scatter(
        lat, lon, elev,
        vmin=T_pred.min(), vmax=T_pred.max(),
        c=y_raw, cmap=CMAP, s=100, 
        edgecolors='k', depthshade=False
    )

    plt.colorbar(p, ax=ax, shrink=0.5, aspect=10, label=r'Nhiệt độ ($^\circ$C)')

    ax.set_title(f'Mô hình Địa hình - Nhiệt độ 3D - {title_suffix}')
    ax.set_xlabel('Kinh độ')
    ax.set_ylabel('Vĩ độ')
    ax.set_zlabel('Độ cao (m)')

    ax.view_init(elev=30, azim=145)
    plt.show()


def getEpsilonFromInput(defaultEpsilon=1.0):
    try:
        eps_input = float(sympify(input(f">> Nhập Epsilon (Mặc định {defaultEpsilon}): ")).evalf())
        epsilon = float(eps_input) if eps_input else defaultEpsilon
    except Exception: 
        print(f"Lỗi nhập! Dùng eps={defaultEpsilon}")
        epsilon = defaultEpsilon

    return epsilon

if __name__ == "__main__":
    filePath = "airTempDataTexasUS-29-04-2025-metric.csv"
    print(f"--- Đang đọc dữ liệu từ: {filePath} ---")
    df = pd.read_csv(filePath)
    df.rename(columns=str.lower, inplace=True)

    targetCol = 'tobs'
    requiredCols = ['latitude', 'longitude', 'elevation', targetCol]

    if not all(col in df.columns for col in requiredCols):
        print(f"Lỗi thiếu cột. Cần: {requiredCols}")

    else:
        df = df[requiredCols]
        df.dropna(inplace=True)
        X = df[requiredCols[:-1]].to_numpy()
        y = df[targetCol].to_numpy()
        print(f"Tổng số điểm dữ liệu: {len(X)}")

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
            epsilon = 1.0
            print(f"Lựa chọn không hợp lệ. Sử dụng mặc định: Gaussian (eps={epsilon})")

        kernelPredeterminedPolDegree = {
            'linear': 1, 'cubic': 2,
            'thin_plate_spline': 1, 'multiquadric': 1
        }
        if kernelName in kernelPredeterminedPolDegree:
            polynomialDegree = kernelPredeterminedPolDegree[kernelName]
        else:
            polynomialDegree = int(input("Nhập bậc đa thức cộng thêm: "))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Khởi tạo mô hình
        modelEval = RBFInterpolator(kernel=kernelName, epsilon=epsilon, degree=polynomialDegree)
        modelEval.fit(X_train, y_train)

        # Dự báo và tính lỗi
        y_pred = modelEval.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print("\n" + "="*40)
        print(f"   KẾT QUẢ ĐÁNH GIÁ (TEST SET 20%)")
        print(f"   Model: {kernelName}")
        print("="*40)
        print(f"MAE  (Sai số tuyệt đối TB): {mae:.4f} °C")
        print(f"RMSE (Căn bậc 2 sai số BP): {rmse:.4f} °C")
        print(f"R2 Score (Độ phù hợp):      {r2:.4f}")
        print("="*40 + "\n")

        print("--- Đang huấn luyện lại trên toàn bộ dữ liệu để vẽ hình... ---")
        model_full = RBFInterpolator(kernel=kernelName, epsilon=epsilon, degree=1)
        model_full.fit(X, y)

        visualize_results(model_full, X, y, title_suffix=f"{kernelName} Kernel")


