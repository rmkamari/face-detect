
import face_recognition
import cv2
import redis_connection as rd

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
rmk_image = face_recognition.load_image_file("sample_pic/Rasool.jpg")
rmk_face_encoding = face_recognition.face_encodings(rmk_image)[0]
# rd.insert_known_face("face_encoding", str(rmk_face_encoding))
# print(rmk_face_encoding)
# print(type(rmk_face_encoding))

# Load a second sample picture and learn how to recognize it.
masood_image = face_recognition.load_image_file("sample_pic/Masood.jpg")
masood_face_encoding = face_recognition.face_encodings(masood_image)[0]
# rd.insert_known_face("face_encoding", masood_face_encoding)

# Load a second sample picture and learn how to recognize it.
mil_image = face_recognition.load_image_file("sample_pic/Mil.jpg")
mil_face_encoding = face_recognition.face_encodings(mil_image)[0]
# rd.insert_known_face("face_encoding", mil_face_encoding)

# Load a second sample picture and learn how to recognize it.
hos_image = face_recognition.load_image_file("sample_pic/Hossein.jpg")
hos_face_encoding = face_recognition.face_encodings(hos_image)[0]

# Load a second sample picture and learn how to recognize it.
ali_image = face_recognition.load_image_file("sample_pic/Ali.jpg")
ali_face_encoding = face_recognition.face_encodings(ali_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    rmk_face_encoding,
    mil_face_encoding,
    masood_face_encoding,
    hos_face_encoding,
    ali_face_encoding
]
# print(known_face_encodings)
# print(type(known_face_encodings))
# known_face_names = rd.known_face()
# print(known_face_names)
known_face_names = [
    "Rasool Maleki",
    "Milad Salehi",
    "Masoud Sojodi",
    "Hosein Hashemi",
    "Ali Niknafas"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "unKnown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        # ft = cv2.freetype.createFreeType2()
        # ft.loadFontData(fontFileName='Zar.ttf', id=0)

        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

