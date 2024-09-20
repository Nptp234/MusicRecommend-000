# import numpy, keras, cv2
# from load_data import load_dataset_test

# load_model = keras.models.load_model
# Model = keras.models.Model
# saved_model = "saved_model/trained_model.h5"

# loaded_model = load_model(saved_model)
# loaded_model.set_weights(loaded_model.get_weights())

# image, label = load_dataset_test(dataset_size=1)
# # image = numpy.expand_dims(image, axis=3)
# image = image/255

# matrix_size = loaded_model.layers[-2].output.shape[1]
# this_model = Model(inputs=loaded_model.inputs, outputs=loaded_model.layers[-2].output)

# # this_model.summary()

# #----------
# print(f"Songs: {numpy.unique(label)}")
# recommend_input = int(input("Enter your song: "))

# predictions_song = []
# predictions_label = []
# predictions_anchor = numpy.zeros((1, matrix_size))

# counts = []
# count = 0
# distance_array = []

# #----------
# # Calculate the latent feature vectors for all the songs.
# for i in range(0, len(label)):
#     if label[i]==recommend_input:
#         print("this 1")
#         test_image = image[i]
#         test_image = numpy.expand_dims(test_image, axis=0)
#         prediction = this_model.predict(test_image)
#         predictions_anchor += prediction
#         count = count + 1
        
#     elif label[i] not in predictions_label:
#         print("this 2")
#         predictions_label.append(label[i])
#         test_image = image[i]
#         test_image = numpy.expand_dims(test_image, axis=0)
#         prediction = this_model.predict(test_image)
#         predictions_song.append(prediction)
#         counts.append(1)
        
#     elif label[i] in predictions_label:
#         print("this 3")
#         index = predictions_label.index(label[i])
#         test_image = image[i]
#         test_image = numpy.expand_dims(test_image, axis=0)
#         prediction = this_model.predict(test_image)
#         predictions_song[index] += prediction
#         counts[index] = counts[index] + 1

# # Count is used for averaging the latent feature vectors.
# predictions_anchor = predictions_anchor / count
# for i in range(len(predictions_song)):
#     # Cosine Similarity - Computes a similarity score of all songs with respect
#     # to the anchor song.
#     predictions_song[i] = predictions_song[i] / counts[i]
#     distance_array.append(
#         numpy.sum(predictions_anchor*predictions_song[i]) / 
#         (numpy.sqrt(numpy.sum(predictions_anchor**2)) * numpy.sqrt(numpy.sum(predictions_song[i]**2)))
#     )
# distance_array = numpy.array(distance_array)

# recommendations = 0
# print("Recommend songs: ")

# # Set your recommendation number
# while recommendations < 2:
#     index = numpy.argmax(distance_array)
#     value = distance_array[index]
#     print(f"Songs: {predictions_label[index]}")
#     distance_array[index] = -numpy.inf
#     recommendations = recommendations + 1
    

import numpy, keras, cv2
from load_data import load_dataset_test

load_model = keras.models.load_model
Model = keras.models.Model
saved_model = "saved_model/trained_model.h5"

loaded_model = load_model(saved_model)
loaded_model.set_weights(loaded_model.get_weights())

image, label = load_dataset_test(dataset_size=1)
image = image / 255

matrix_size = loaded_model.layers[-2].output.shape[1]
this_model = Model(inputs=loaded_model.inputs, outputs=loaded_model.layers[-2].output)

print(f"Songs: {numpy.unique(label)}")
recommend_input = input("Enter your song: ")

predictions_song = []
predictions_label = []
predictions_anchor = numpy.zeros((1, matrix_size))

counts = []
distance_array = []
count = 0

# Calculate the latent feature vectors for all the songs.
for i in range(0, len(label)):
    test_image = numpy.expand_dims(image[i], axis=0)
    
    # Nếu bài hát hiện tại trùng với bài hát đầu vào (anchor)
    if label[i] == recommend_input:
        print("this 1")
        prediction = this_model.predict(test_image)
        predictions_anchor += prediction
        count += 1  # Đếm số lần xuất hiện của bài hát được chọn
    
    # Nếu bài hát hiện tại chưa xuất hiện trong danh sách các bài hát đã dự đoán
    elif label[i] not in predictions_label:
        print("this 2")
        predictions_label.append(label[i])
        prediction = this_model.predict(test_image)
        predictions_song.append(prediction)
        counts.append(1)  # Khởi tạo số đếm cho bài hát mới
    
    # Nếu bài hát hiện tại đã xuất hiện trong danh sách dự đoán
    else:
        print("this 3")
        index = predictions_label.index(label[i])
        prediction = this_model.predict(test_image)
        predictions_song[index] += prediction
        counts[index] += 1  # Tăng số đếm cho bài hát này

# Kiểm tra nếu `count == 0` để tránh lỗi chia cho 0
if count > 0:
    predictions_anchor = predictions_anchor / count
else:
    raise ValueError("Không tìm thấy bài hát khớp với đầu vào")

# Tính toán độ tương đồng cosine giữa các bài hát và bài hát được chọn
for i in range(len(predictions_song)):
    predictions_song[i] = predictions_song[i] / counts[i]  # Lấy trung bình các dự đoán
    numerator = numpy.sum(predictions_anchor * predictions_song[i])
    denominator = (numpy.sqrt(numpy.sum(predictions_anchor ** 2)) * 
                   numpy.sqrt(numpy.sum(predictions_song[i] ** 2)))
    if denominator != 0:
        similarity = numerator / denominator
    else:
        similarity = 0  # Tránh chia cho 0
    distance_array.append(similarity)

# Chuyển mảng `distance_array` thành numpy array
distance_array = numpy.array(distance_array)

recommendations = 0
print("Recommend songs: ")

# Lấy ra 2 bài hát có độ tương đồng cao nhất
while recommendations < 2:
    index = numpy.argmax(distance_array)
    print(f"Songs: {predictions_label[index]}")
    distance_array[index] = -numpy.inf  # Đảm bảo bài hát này không được chọn lại
    recommendations += 1

