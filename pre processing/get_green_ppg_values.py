import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import preprocess 
import cv2 
from sklearn.preprocessing import MinMaxScaler
os.chdir(r'D:\NITT\interns\iiit research summer intern\datasets\bp\bp recordings\bp intern dataset second aug')
directory = os.scandir()
data = np.zeros((1611, 600))
index = 0
for file in directory:
    if file.name[-4: ] != '.mp4':
        continue
    # print(file.name)
    cap = cv2.VideoCapture(file.name)
    green_channel_mean = []
    count = 1
    sampling_frequency = 30

    while(cap.isOpened() and count <= 600):
        if file.name == 'VID_20220228_101451.mp4' and count == 599:
            green_channel_mean.append(green_channel_mean[-1])
            break
        ret, frame = cap.read()
        if ret is not True:
            if len(green_channel_mean) < 600:
                remaining = 600 - len(green_channel_mean)
                for i in range(remaining):
                    green_channel_mean.append(green_channel_mean[-1])
            break
        # print(f'frame {count} {ret}')
        if frame is None:
            continue 
        h, w, r_g_b = frame.shape 
        mid = w // 2
        cv2.rectangle(frame, (mid - 180, 75), (mid + 180, 750), (255, 255, 0), 5)
        green_pixels = frame[75: 750, mid - 180: mid + 180, 1]
        green_channel_mean.append(np.mean(green_pixels))

        # print(count)
        count += 1
        # if ret == True:
        #     cv2.imshow('Frame', frame)
        #     if cv2.waitKey(25) & 0xFF == ord('q'):
        #         break
        # else:
        #     break 
    print(f'video {index} {file.name}')
    data = np.vstack((data, green_channel_mean))
    data[index] = np.array(green_channel_mean)
    cap.release()
    cv2.destroyAllWindows()
    index += 1
    

# print(get_normalized_signal(np.array(green_channel_mean)).ndim)
# time = np.linspace(0, len(green_channel_mean), len(green_channel_mean))
# plt.plot(time, green_channel_mean)
# plt.plot(time, preprocess.get_clean_signal(green_channel_mean))
# plt.show()
# plt.figure()

# green_channel_mean = np.array(green_channel_mean)
# fil = preprocess.get_clean_signal(green_channel_mean)
# t = np.linspace(0, len(fil), len(fil))
# plt.plot(t, fil)
# plt.show()
# fil = fil.squeeze()
# # print(fil.ndim)
# # print(fil.shape)
# systolics = preprocess.systolic_peaks(fil)
# # print(systolics)
# tfns = preprocess.tfn_points(fil)
# # print(tfns)
# t = np.linspace(0, len(fil), len(fil))
# plt.figure(figsize=(17,6))
# plt.plot(t, fil, c="r", alpha=0.6)

# plt.scatter(systolics, fil[systolics], c="g", label="systolic peaks")
# plt.scatter(tfns, fil[tfns], c="m", label="tfn")
# plt.legend(prop={'size':16})
# # plt.xlim(100)
# plt.show()
# beats, systolics = preprocess.beat_segmentation(fil)
# # print(systolics)
# beats_features = preprocess.peaks_detection(beats, systolics)
# # print(beats_features)
# # for beat, systolic, dicrotic, diastolic in beats_features:
# #     plt.figure()
# #     plt.plot(beat)
# #     plt.scatter(systolic,beat[systolic], c="g")
# #     plt.scatter(diastolic,beat[diastolic], c= "r")
# #     plt.scatter(dicrotic,beat[dicrotic], c= "gray")
# #     plt.show()

# extracted_features = preprocess.extract_features(fil, beats_features, sampling_frequency)
# print(extracted_features.shape)
# data = pd.DataFrame(data)
# os.chdir(r'D:\NITT\interns\iiit research summer intern\datasets\bp\blood pressure dl\csv dataset')
# data.to_csv('green_ppg_values_newaug.csv')
# scaler = MinMaxScaler()
# store_address = r'D:\College\interns\iiit research summer intern\datasets\csv dataset'
# preprocess.prepare_dataset(data, scaler, store_address)