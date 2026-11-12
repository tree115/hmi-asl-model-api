# %%time
# features = '' if ling_features else '_only_lmrks'
# # ling_features = True: đặc trưng đã xử lí
# # False: giữa tọa độ landmark thô
# # Path where the extracted features are stored
# features_path = PATHS['features_data'].format(features)

# # If the extracted features are not saved yet, we create them and then store them
# if not os.path.exists(features_path):
#     print('Creating feature-extracted data...')
#     features_data = {}

#     for i, k in enumerate(anns_with_tiers):
#         print('Extracting features for video {}/{}'.format(i+1, len(list(anns_with_tiers.keys()))), end = '\r')
#         # Get the glosses of the file
#         anns_dict = anns_with_tiers[k]
#         # đọc và chuẩn hóa landmarks
#         # Loading the numpy files for a specific video
#         lmrk_dict = load_numpy(cngt_lmrks, k.replace('.eaf', '')) # đọc landmark đã lưu cho video
#         # Normalize the coordinates, also create a horizontally flipped version
#         # chuyển tọa độ sang hệ pixel + chuẩn hóa -> dữ liệu ít phụ thuộc kích thước hình or vị trí người
#         lmrk_dict = normalise_coordinates(get_pixel_coordinates(lmrk_dict))
#         pose = lmrk_dict['pose'] # landmark cơ thể

#         # landmark từ pose để thay thế các điểm tương ứng trong dữ liệu bàn tay
#         lmrk_dict['l_hand'][:, [0,4,5,17], :] = pose[:, [15,17,19,21], :]
#         lmrk_dict['r_hand'][:, [0,4,5,17], :] = pose[:, [16,18,20,22], :]

#         # nếu chỉ lấy landmark thô
#         if 'only_lmrks' in features_path:
#             l_lmrk = lmrk_dict['l_hand']
#             r_lmrk = lmrk_dict['r_hand']
#             shape = l_lmrk.shape
#             l_lmrk = l_lmrk.reshape(shape[0], -1).astype(np.float32)
#             r_lmrk = r_lmrk.reshape(shape[0], -1).astype(np.float32)
#         else:  # lấy đặc trưng
#             # Getting landmarks preprocessed, with new format being (wrist, distances, angles) for each frame
#             l_lmrk = get_wrist_angle_distance(lmrk_dict, 'l_hand', pose).astype(np.float32)
#             r_lmrk = get_wrist_angle_distance(lmrk_dict, 'r_hand', pose).astype(np.float32)
#         # Store in the dictionary
#         features_data[k] = {'l_hand' : l_lmrk, 'r_hand': r_lmrk}

#     # Store the extracted features in a numpy file (faster than pickling)
#     features_data_np = np.array(list(features_data.items()))
#     np.save(features_path, features_data_np)

# else:
#     print('Feature-extracted data exists, loading...')
#     # Allow pickle has to be true because we have nested dictionaries
#     features_data = dict(np.load(features_path, allow_pickle = True))
# print('Done!')
