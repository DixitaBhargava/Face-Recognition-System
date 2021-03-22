import face_recognition

# Load the known images
image_1 = face_recognition.load_image_file("person_1.jpg")
image_2 = face_recognition.load_image_file("person_2.jpg")
image_3 = face_recognition.load_image_file("person_3.jpg")

# Get the face encoding of each person. This can fail if no one is found in the photo.
image_1_face_encoding = face_recognition.face_encodings(image_1)[0]
image_2_face_encoding = face_recognition.face_encodings(image_2)[0]
image_3_face_encoding = face_recognition.face_encodings(image_3)[0]

# Create a list of all known face encodings
known_face_encodings = [
    image_1_face_encoding,
    image_2_face_encoding,
    image_3_face_encoding
]

# Load the image we want to check
unknown = face_recognition.load_image_file("unknown_7.jpg")

# Get face encodings for any people in the picture
face_locations = face_recognition.face_locations(unknown, number_of_times_to_upsample=2)
unknown_face_encodings = face_recognition.face_encodings(unknown, known_face_locations=face_locations)
# There might be more than one person in the photo, so we need to loop over each face we found
for unknown_face_encoding in unknown_face_encodings:

    # Test if this unknown face encoding matches any of the three people we know
    results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)

    name = "Unknown"

    if results[0]:
        name = "Person 1"
    elif results[1]:
        name = "Person 2"
    elif results[2]:
        name = "Person 3"

    print(f"Found {name} in the photo!")
