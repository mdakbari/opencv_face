import numpy as np
import cv2 as cv
import os

haar_cascade = cv.CascadeClassifier('haar_face.xml')

# pepole =['Aaron_Eckhart', 'Aaron_Guiel', 'Aaron_Patterson', 'Aaron_Peirsol', 'Aaron_Pena', 'Aaron_Sorkin', 'Aaron_Tippin', 'Abbas_Kiarostami', 'Abba_Eban', 'Abdel_Aziz_Al-Hakim', 'Abdel_Madi_Shabneh', 'Abdel_Nasser_Assidi', 'Abdoulaye_Wade', 'Abdulaziz_Kamilov', 'Abdullah', 'Abdullah_Ahmad_Badawi', 'Abdullah_al-Attiyah', 'Abdullah_Gul', 'Abdullah_Nasseef', 'Abdullatif_Sener', 'Abdul_Majeed_Shobokshi', 'Abdul_Rahman', 'Abel_Aguilar', 'Abel_Pacheco', 'Abid_Hamid_Mahmud_Al-Tikriti', 'Abner_Martinez', 'Abraham_Foxman', 'Aby_Har-Even', 'Adam_Ant', 'Adam_Freier', 'Adam_Herbert', 'Adam_Kennedy', 'Adam_Mair', 'Adam_Rich', 'Adam_Sandler', 'Adam_Scott', 'Adelina_Avila', 'Adel_Al-Jubeir', 'Adisai_Bodharamik', 'Adolfo_Aguilar_Zinser', 'Adolfo_Rodriguez_Saa', 'Adoor_Gopalakarishnan', 'Adriana_Lima', 'Adriana_Perez_Navarro', 'Adrianna_Zuzic', 'Adrian_Annus', 'Adrian_Fernandez', 'Adrian_McPherson', 'Adrian_Murrell', 'Adrian_Nastase', 'Adrien_Brody', 'Afton_Smith', 'Agbani_Darego', 'Agnelo_Queiroz', 'Agnes_Bruckner', 'Ahmad_Jbarah', 'Ahmad_Masood', 'Ahmed_Ahmed', 'Ahmed_Chalabi', 'Ahmed_Ghazi', 'Ahmed_Ibrahim_Bilal', 'Ahmed_Lopez', 'Ahmed_Qureia', 'Ahmet_Demir', 'Ahmet_Necdet_Sezer', 'Aicha_El_Ouafi', 'Aidan_Quinn', 'Aileen_Riggin_Soule', 'Ainsworth_Dyer', 'Ain_Seppik', 'Aishwarya_Rai', 'Aitor_Gonzalez', 'Aiysha_Smith', 'Ai_Sugiyama', 'Ajit_Agarkar', 'AJ_Cook', 'AJ_Lamas', 'Akbar_Al_Baker', 'Akbar_Hashemi_Rafsanjani', 'Akhmed_Zakayev', 'Akiko_Morigami', 'Akmal_Taher', 'Alain_Cervantes', 'Alain_Ducasse', 'Alanis_Morissette', 'Alanna_Ubach', 'Alan_Ball', 'Alan_Dershowitz', 'Alan_Dreher', 'Alan_Greenspan', 'Alan_Greer', 'Alan_Jackson', 'Alan_Mulally', 'Alan_Stonecipher', 'Alan_Tang_Kwong-wing', 'Alan_Trammell', 'Alan_Zemaitis', 'Alastair_Campbell', 'Alastair_Johnston', 'Albaro_Recoba', 'Alberta_Lee', 'Alberto_Acosta', 'Alberto_Fujimori', 'Alberto_Gonzales', 'Alberto_Ruiz_Gallardon', 'Alberto_Sordi', 'Albert_Brooks', 'Albert_Costa', 'Albert_Montanes', 'Albert_Pujols', 'Albrecht_Mentz', 'Aldo_Paredes', 'Alecos_Markides', 'Alec_Baldwin', 'Alejandro_Atchugarry', 'Alejandro_Avila', 'Alejandro_Fernandez', 'Alejandro_Gonzalez_Inarritu', 'Alejandro_Lembo', 'Alejandro_Lerner', 'Alejandro_Lopez', 'Alejandro_Toledo', 'Aleksander_Kwasniewski', 'Aleksander_Voloshin', 'Alek_Wek', 'Alessandra_Cerna', 'Alessandro_Nesta', 'Alexander_Downer', 'Alexander_Losyukov', 'Alexander_Lukashenko', 'Alexander_Payne', 'Alexander_Rumyantsev', 'Alexandra_Jackson', 'Alexandra_Pelosi', 'Alexandra_Rozovskaya', 'Alexandra_Spann', 'Alexandra_Stevenson', 'Alexandra_Vodjanikova', 'Alexandre_Daigle', 'Alexandre_Despatie', 'Alexandre_Herchcovitch', 'Alexandre_Vinokourov', 'Alexa_Loren', 'Alexa_Vega', 'Alexis_Bledel', 'Alexis_Dennisoff', 'Alex_Barros', 'Alex_Cabrera', 'Alex_Cejka', 'Alex_Corretja', 'Alex_Ferguson', 'Alex_Gonzalez', 'Alex_Holmes', 'Alex_King', 'Alex_Penelas', 'Alex_Popov', 'Alex_Sink', 'Alex_Wallau', 'Alex_Zanardi', 'Alfonso_Cuaron', 'Alfonso_Portillo', 'Alfonso_Soriano', 'Alfredo_di_Stefano', 'Alfredo_Moreno', 'Alfredo_Pena', 'Alfred_Ford', 'Alfred_Sant', 'Alice_Fisher', 'Alicia_Hollowell', 'Alicia_Keys', 'Alicia_Molik', 'Alicia_Silverstone', 'Alicia_Witt', 'Alimzhan_Tokhtakhounov', 'Alina_Kabaeva', 'Aline_Chretien', 'Alisha_Richman', 'Alison_Krauss', 'Alison_Lohman', 'Alistair_MacDonald', 'Ali_Abbas', 'Ali_Abdullah_Saleh', 'Ali_Adbul_Karim_Madani', 'Ali_Ahmeti', 'Ali_Bin_Hussein', 'Ali_Fallahian', 'Ali_Hammoud', 'Ali_Khamenei', 'Ali_Mohammed_Maher', 'Ali_Naimi', 'Allan_Houston', 'Allan_Kemakeza', 'Allan_Wagner', 'Allen_Iverson', 'Allen_Rock', 'Allison_Janney', 'Allison_Searing', 'Allyson_Felix', 'Ally_Sheedy', 'Alma_Powell', 'Almeida_Baptista', 'Alonzo_Mourning', 'Alvaro_Noboa', 'Alvaro_Silva_Calderon', 'Alvaro_Uribe', 'Alyse_Beaupre', 'Alyson_Hannigan', 'Aly_Wagner', 'Al_Cardenas', 'Al_Davis', 'Al_Gore', 'Al_Leiter', 'Al_Pacino', 'Al_Sharpton', 'Amanda_Beard', 'Amanda_Bynes', 'Amanda_Coetzer', 'Amanda_Marsh', 'Amanda_Plumer', 'Amber_Frey', 'Amber_Tamblyn', 'Ambrose_Lee', 'Amelia_Vega', 'Amelie_Mauresmo']

pepole = ['janak', 'manthan']
face_rego = cv.face.LBPHFaceRecognizer_create()
face_rego.read('face_rego.yml')

resize_img = cv.imread(r'c:\Users\Win10\Downloads\IMG_2754.JPG')

img = cv.resize(resize_img, (250, 250))  # Provide your desired width and height here

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

for (x,y,w,h) in face_rect:
    face_roi = gray[y:y+h, x:x+w]
    labals, confidence = face_rego.predict(face_roi)
    print(f'Label = {pepole[labals]} with a confidence of {confidence}')

    cv.putText(img, str(pepole[labals]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)


