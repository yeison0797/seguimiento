import numpy as np
import face_recognition as fr
import cv2

video_capture = cv2.VideoCapture(0)

yeison_image = fr.load_image_file("yeisonfoto.jpg")
yeison_face_encoding = fr.face_encodings(yeison_image)[0]

#felipe_image =fr.load_image_file("felipefoto.jpeg")
#felipe_image_encoding = fr.face_encodings(felipe_image)[0]

rector_image =fr.load_image_file("rector.jpg")
rector_image_encoding = fr.face_encodings(rector_image)[0]


known_face_encodings = [yeison_face_encoding, rector_image_encoding]
known_face_names = ["Yeison", "Rector"]

while True:
	ret, frame = video_capture.read()

	rgb_frame = frame[:, :, ::-1]

	face_locations = fr.face_locations(rgb_frame)
	face_encodings = fr.face_encodings(rgb_frame, face_locations)

	for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

		matches = fr.compare_faces(known_face_encodings, face_encoding)

		name = "Unknown"

		face_distances = fr.face_distance(known_face_encodings, face_encoding)

		best_match_index = np.argmin(face_distances)
		if matches[best_match_index]:
			name = known_face_names[best_match_index]

		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255),2)

		cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

	cv2.imshow('segimiento_facial', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video_capture.release()
cv2.destroyAllWindows()
