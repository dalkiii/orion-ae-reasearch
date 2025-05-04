import numpy as np
import os

import scipy
from pywt import cwt
from scipy.ndimage import median_filter
from scipy.signal import morlet2, resample
from skimage.transform import resize


# 📌 영점 교차 탐지 함수 (노이즈 감소 포함)
def find_zero_crossings(signal, threshold=0.01):
    smoothed_signal = median_filter(signal, size=5)  # 노이즈 완화
    crossings = np.where((np.diff(np.sign(smoothed_signal)) < 0) & (np.abs(smoothed_signal[:-1]) > threshold))[0]
    return crossings

# 📌 Hanning 윈도우를 적용하지 않고 원본 신호 유지
def apply_hanning_window(signal):
    return signal  # 원본 신호 유지

# 📌 resampling을 이용한 신호 길이 정규화 함수
def normalize_signal_length(signal, target_length):
    """
    주어진 신호를 resampling하여 목표 길이로 조정
    resampling을 통해 원본 신호의 전체 특성을 보존하면서 정규화
    """
    resampled_signal = resample(signal, target_length)
    return resampled_signal

# 📌 기존 CWT 변환 함수 (유지)
def cwt_to_numpy(signal):
    scales = np.arange(1, 128)  # 1부터 128까지 모든 스케일 적용
    cwt_result = cwt(signal, morlet2, scales, w=6)
    return np.abs(cwt_result).astype(np.float32)

# [Added] CWT 결과를 SpecGAN 전처리 방식에 맞게 스펙트로그램 이미지로 변환하는 함수
def cwt_to_spectrogram_image(signal, target_shape=(224, 224)):
    """
    입력 신호에 대해 CWT를 수행하고,
    (1) 로그 스케일 변환,
    (2) 주파수 bin별 정규화 (제로-평균, 단위분산),
    (3) 클리핑 (-3~3) 및 선형 재스케일링하여 [-1,1] 범위로 변환,
    (4) 최종적으로 target_shape 크기의 이미지로 리사이즈.
    """
    # 기존 CWT 변환 (shape: (127, len(signal)))
    cwt_result = cwt_to_numpy(signal)

    # 로그 스케일 적용 (log(1 + x))
    log_cwt = np.log1p(cwt_result)

    # 주파수 bin별로 정규화: 각 row에 대해 평균, std 계산 후 (value - mean)/std, 클리핑, 재스케일
    norm_cwt = np.empty_like(log_cwt)
    for i in range(log_cwt.shape[0]):
        row = log_cwt[i, :]
        mean_val = np.mean(row)
        std_val = np.std(row) if np.std(row) > 0 else 1.0
        row_norm = (row - mean_val) / std_val
        row_norm = np.clip(row_norm, -3, 3)   # 클리핑
        norm_cwt[i, :] = row_norm / 3.0        # [-1, 1] 범위로 재스케일

    # [Modified for 224x224] 최종적으로 target_shape (예: 224x224)로 리사이즈
    spectrogram_image = resize(norm_cwt, target_shape, mode='reflect', anti_aliasing=True)
    return spectrogram_image.astype(np.float32)

# 📌 주기 길이 분석 (Mean 기준)
def calculate_fixed_length(cycle_lengths):
    mean_length = int(np.mean(cycle_lengths))
    print(f"📌 Mean Cycle Length: {mean_length}")
    return mean_length

# [Modified] 파일 처리 함수: GAN 기반 클래스 분류용 스펙트로그램 이미지 생성 (target_sensor 데이터만 사용)
def process_file(filepath, target_sensor, fixed_length, series_name, tightening_level):
    print(f"Processing: {os.path.basename(filepath)} | Series: {series_name} | Tightening Level: {tightening_level}")
    data = scipy.io.loadmat(filepath)

    # 센서 데이터 로드 (총 3가지 - A: Micro80, B: F50A, C: Micro200HF)
    ae_data = data[target_sensor].flatten()  # AE data : 처리 대상
    vibrometer_data = data['D'].flatten()      # 진동데이터 : 영점 교차 기준으로 사용

    # 영점 교차 탐지
    zero_crossings = find_zero_crossings(vibrometer_data)

    # 주기별 데이터 추출
    cycles = [ae_data[zero_crossings[i]:zero_crossings[i+1]] for i in range(len(zero_crossings)-1)]

    # 주기 길이 필터링 (평균 이상)
    mean_length = int(np.mean([len(c) for c in cycles]))
    valid_cycles = [c for c in cycles if len(c) >= mean_length]

    num_cycles = len(valid_cycles)
    # [Modified for 224x224] 최종 출력 이미지 크기를 SpecGAN에 적합한 224x224로 설정
    target_image_shape = (224, 224)

    # NumPy 배열을 사전 할당하여 메모리 절감 (스펙트로그램 이미지 결과 저장)
    numpy_ae = np.empty((num_cycles, target_image_shape[0], target_image_shape[1]), dtype=np.float32)

    # CWT 변환 및 후처리: 기존 CWT 결과를 SpecGAN 전처리 방식에 맞게 스펙트로그램 이미지로 변환
    for idx in range(num_cycles):
        cycle = apply_hanning_window(valid_cycles[idx])
        norm = normalize_signal_length(cycle, fixed_length)
        # [Modified for 224x224] 기존 cwt_to_numpy 대신 cwt_to_spectrogram_image 함수를 사용하여 스펙트로그램 생성
        numpy_ae[idx] = cwt_to_spectrogram_image(norm, target_shape=target_image_shape)

    # 센서별 개별 데이터 저장 경로 설정 (Measurement Series & Tightening Level 포함)
    series_path_single = os.path.join(output_dir_single, series_name, tightening_level)
    os.makedirs(series_path_single, exist_ok=True)

    # 개별 센서 데이터 저장 (float32 변환) - 파일 이름에 'rev2' 추가
    output_filename = f"{os.path.basename(filepath)}_{target_sensor}_rev2.npy"
    np.save(os.path.join(series_path_single, output_filename), numpy_ae.astype(np.float32))

    print(f"Saved Single-Sensor Spectrogram Data for: {os.path.basename(filepath)}")

    return {
        "filename": os.path.basename(filepath),
        "series": series_name,
        "tightening_level": tightening_level,
        "total_cycles": len(cycles),
        "valid_cycles": len(valid_cycles)
    }


if __name__ == "__main__":
    # 📌 전체 데이터가 있는 루트 디렉토리
    root_dir = "./data"
    output_dir_single = f"./data/resized"  # 개별 센서별 저장
    os.makedirs(output_dir_single, exist_ok=True)

    # 📌 FIXED_LENGTH 설정을 위한 주기 길이 데이터 수집
    # (모든 measurementSeries_* 폴더에서 cycle_lengths를 수집)
    cycle_lengths_all = []

    for series_name in os.listdir(root_dir):
        series_path = os.path.join(root_dir, series_name)
        if os.path.isdir(series_path):
            # measurementSeries_ 로 시작하는 폴더 대상으로 함 (measurementSeries_B, C, D, E, F)
            if series_name.startswith("measurementSeries_"):
                for tightening_level in os.listdir(series_path):
                    level_path = os.path.join(series_path, tightening_level)
                    if os.path.isdir(level_path):
                        for filename in os.listdir(level_path):
                            if filename.endswith(".mat"):
                                data = scipy.io.loadmat(os.path.join(level_path, filename))
                                vibrometer_data = data['D'].flatten()
                                zero_crossings = find_zero_crossings(vibrometer_data)
                                # IndexError 방지 (zero_crossings이 너무 짧은 경우)
                                if len(zero_crossings) > 1:
                                    cycle_lengths = [zero_crossings[i+1] - zero_crossings[i] for i in range(len(zero_crossings)-1)]
                                    # 이상치 제거 (10,000 이하 값 필터링)
                                    cycle_lengths_all += [c for c in cycle_lengths if c > 10000]


    FIXED_LENGTH = calculate_fixed_length(cycle_lengths_all)
    # measurementSeries_* 폴더에 대해 처리
    results = []
    # 처리할 시리즈 목록 (필요에 따라 확장/수정 가능)
    target_series_list = ["measurementSeries_B", "measurementSeries_C", "measurementSeries_D", "measurementSeries_E", "measurementSeries_F"]
    # target_series_list = ["measurementSeries_B"]

    for target_sensor in ["A", "B", "C"]:
        for target_series in target_series_list:
            series_path = os.path.join(root_dir, target_series)
            if os.path.isdir(series_path):
                for tightening_level in os.listdir(series_path):
                    level_path = os.path.join(series_path, tightening_level)
                    if os.path.isdir(level_path):
                        for filename in os.listdir(level_path):
                            if filename.endswith(".mat"):
                                result = process_file(
                                    os.path.join(level_path, filename),
                                    target_sensor,
                                    fixed_length=FIXED_LENGTH,
                                    series_name=target_series,
                                    tightening_level=tightening_level
                                )
                                results.append(result)

        # -----------------------------------------------------------------------------
        # 결과 요약
        print(f"\n📊 Summary of Processed Files sensor {target_sensor}:")
        for result in results:
            print(f"🔹 Series: {result['series']} | Level: {result['tightening_level']} | File: {result['filename']}")
            print(f"    Total Cycles: {result['total_cycles']}, Valid Cycles: {result['valid_cycles']}")