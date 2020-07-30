from __future__ import absolute_import, division, print_function
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import unicodedata
import speech_recognition as sr
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys
import webbrowser
import pyttsx3
# import pyttsx
import pyaudio
import gtts
from gtts import gTTS
from pygame import mixer
from urllib.request import urlopen
import xml
import time
import say
import math
import cmath
import re
import image
import html
import html.parser
import numpy
# import dlib
# import voice_assistant as va
# import tesseract
import pytesseract
import PIL
from PIL import Image
# import opencv
import cv2
import wave
import ffmpeg
import subprocess
import bs4
# import pypi
import pypl
# import beautifulsoup4
# import vlc
import youtube_dl
from pyowm import OWM
from bs4 import BeautifulSoup as soup
import wikipedia
# import Image
# from tesseract import image_to_string
# import pillow
# import tesseract-ocr
import django
import smtplib
import requests
import urllib
import urllib3
import json
import _json
import seaborn
import json
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
# from urllib2 import urlopen
import random
from time import strftime
import datetime
import _datetime
from langdetect import detect_langs
import numpy as np
import theano
import argparse
# import imutils
# from imutils.video import VideoStream
# from imutils.video import FPS
import csv
import imageai
import tensorflow as tf
import sympy
import pandas as pd
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib


FLAGS = None
# tf.enable_eager_execution()
print(tf.__version__)

# hello = tf.constant('Hello, TensorFlow')
# sess = tf.Session()
# print(sess.run(hello))

# path_to_zip = tf.keras.utils.get_file(
#     'rus-eng.zip', origin='https://storage.googleapis.com/download.tensorflow.org/data/rus-eng.zip',
#     extract=True)

path_to_file = os.path.dirname('C:/Skynet/ruseng')+"/rus.txt"





def talk(words):
   print(words)
   # os.system("say " + words)
   engine = pyttsx3.init()

   ru_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\EISE-NICOLAI16"
   # ru_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\RHVoice-Aleksandr-Russian"
   # engine.setProperty('voice', ru_voice_id)
   # en_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MS-Anna-1033-20-DSK"
   engine.setProperty('voice', ru_voice_id)
   voices = engine.getProperty('voices')
   # for voice in voices:
   #                      print(voice, voice.id)
   #                      engine.setProperty('voice', voice.id)
   # engine.say("Hello World!")
   #                      engine.runAndWait()
   #                      engine.stop()
   # for voice in voices:
   #     print
   #     voice
   #     # if voice.languages[0] == u'en_US':
   #     #     engine.setProperty('voice', voice.id)
   #     #     break
   # engine.setProperty('voice', voices[0].id)





                     # voices = engine.getProperty('voices')
                     # for voice in voices:
                     #    engine.setProperty('voice', voice.man)
                     #    engine.say('The quick brown fox jumped over the lazy dog.')

   engine.say(words)
   engine.runAndWait()

#talk("Privet, menia zovut skaynet")
talk("Привет")



def talk2(words2):
   print(words2)
   # os.system("say " + words)
   engine2 = pyttsx3.init()
   ru_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\EISE-NICOLAI16"
   engine2.setProperty('voice', ru_voice_id)
   engine2.say(words2)
   engine2.runAndWait()
def talk1(words1):
   print(words1)
   # os.system("say " + words)
   engine1 = pyttsx3.init()

   # en_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MS-Anna-1033-20-DSK"
   en_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
   engine1.setProperty('voice', en_voice_id)

   engine1.say(words1)
   engine1.runAndWait()



def command():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        talk("Говорите")
        r.pause_threshold=1
        r.adjust_for_ambient_noise(source, duration=1)
        # r.adjust_for_ambient_noise(source)
        audio=r.listen(source)
    try:
        zadanie=r.recognize_google(audio, language="ru_RU")
        print("Вы сказали:" + format(zadanie))
        # zadanie = r.recognize_google(audio)
        # print("Вы сказали: " + command)
    except sr.UnknownValueError:
        talk("Я Вас не понял")
        zadanie=command()
    return zadanie






def videocam():
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('C:/Skynet/trainer/trainer.yml')
        cascadePath = "C:/Skynet/haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath);

        font = cv2.FONT_HERSHEY_SIMPLEX

        id = 0

        names = ['None', 'Igor', 'Olga', 'vladimir', 'Sofiya', 'Nikolay', 'Z', 'W']
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video widht
        cam.set(4, 480)  # set video height

        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)

        while True:

            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            for (x, y, w, h) in faces:
                eyeCascade = cv2.CascadeClassifier(
                    "C:/Skynet/OpenCV-Face-Recognition-master-master/OpenCV-Face-Recognition-master-master/FaceDetection/Cascades/haarcascade_eye.xml")
                smileCascade = cv2.CascadeClassifier(
                    "C:/Skynet/OpenCV-Face-Recognition-master-master/OpenCV-Face-Recognition-master-master/FaceDetection/Cascades/haarcascade_smile.xml")

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)



                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]



                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                if (confidence < 100):
                    id = names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

                eyes = eyeCascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.5,
                    minNeighbors=5,
                    minSize=(5, 5),
                )

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 69, 255), 2)

                smile = smileCascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.5,
                    minNeighbors=15,
                    minSize=(25, 25),
                )

                for (xx, yy, ww, hh) in smile:
                    cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (15, 185, 255), 2)


            cv2.imshow('camera', img)

            k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video

            if k == 27:
                break


        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()







# def makeSomethink(zadanie):
#     if 'let is go talk' in zadanie:
#         print("Давай поговорим")
#         talk("Давай поговорим")
def calc(zadanie):
    val = s.group()
    # if not val.strip(): return val
    # return " %s " % eval(val.strip(), {'__builtins__': None})
    # re.sub(r"([0-9\ \.\+\*\-\/(\)]+)", calc, "I have 6 * (2 + 3 ) apples")
def makeSomethink(zadanie):
    if 'Открой контакт' in zadanie:
        talk("Открываю вконтакте")
        url='https://vk.com'
        webbrowser.open(url)
    elif 'включи зрение' in zadanie.lower():
        videocam()
    elif 'открой контакт'.lower() in zadanie:
        talk("Открываю вконтакте")
        url='https://vk.com'
        webbrowser.open(url)
    elif 'Выключи'.lower() in zadanie:
        talk("Выключаюсь")
        sys.exit()
    elif 'Stop the program'.lower() in zadanie:
        talk1("stop the program")
        sys.exit()
    elif 'Как тебя зовут' in zadanie:
        talk("меня зовут скайнет")
    elif 'как тебя зовут' in zadanie:
        talk("меня зовут скайнет")
    elif 'What is your name' in zadanie:
        talk1("my name is skynet")
    elif 'what is your name' in zadanie:
        talk1("my name is skynet")
    elif 'Включи видосы'.lower() in zadanie:
        talk("Открываю ютуб")
        url='https://youtube.com'
        webbrowser.open(url)
    elif 'Открой YouTube' in zadanie:
        talk("Открываю ютуб")
        url='https://youtube.com'
        webbrowser.open(url)
    elif 'открой YouTube' in zadanie:
        talk("Открываю ютуб")
        url='https://youtube.com'
        webbrowser.open(url)
    elif 'Открой Яндекс' in zadanie:
        talk("Открываю яндекс")
        url = 'https://yandex.ru'
        webbrowser.open(url)
    elif 'Открой Google' in zadanie:
        talk("Открываю гугл")
        url = 'https://google.com'
        webbrowser.open(url)
    elif 'Открой Гугл' in zadanie:
        talk("Открываю гугл")
        url = 'https://google.com'
        webbrowser.open(url)
    elif 'Включи деньги' in zadanie:
        talk("Включаю программу для денег")
        os.system("start C:/Users/СеменовыИО/Desktop/RuCaptchaBot.lnk")
    elif 'Открой мой компьютер' in zadanie:
        talk("Открываю компьютер")
        os.system("start C:/Users/Игорь/Desktop/Компьютер.lnk")
    elif 'Открой Косынку' in zadanie:
        talk("Открываю Косынку")
        os.system("start C:/Users/СеменовыИО/Desktop/Карты.lnk")
    elif 'Сделай фото' in zadanie:
        # Включаем первую камеру
        talk("Включаю камеру")
        cap = cv2.VideoCapture(0)
        talk("Камера включена")
        # "Прогреваем" камеру, чтобы снимок не был тёмным
        talk("Прогреваю камеру")
        for i in range(5):
            cap.read()
        talk("Камера прогрета")
        # Делаем снимок
        talk("Делаю снимок")
        ret, frame = cap.read()
        talk("Снимок сделан")
        # Записываем в файл
        talk("Записываю файл")
        cv2.imwrite('C:/Skynet/cam.png', frame)
        talk("Файл сохранен")
        # Отключаем камеру
        talk("Выключаю камеру")
        cap.release()
        talk("Камера выключена")
    elif 'Открой это фото' in zadanie:
        talk("Открываю фото")
        os.system("start C:/Skynet/cam.png")
    elif 'открой это фото' in zadanie:
        talk("Открываю фото")
        os.system("start C:/Skynet/cam.png")
    elif 'сделай фото' in zadanie:
        # Включаем первую камеру
        talk("Включаю камеру")
        cap = cv2.VideoCapture(0)
        talk("Камера включена")
        # "Прогреваем" камеру, чтобы снимок не был тёмным
        talk("Прогреваю камеру")
        for i in range(5):
            cap.read()
        talk("Камера прогрета")
        # Делаем снимок
        talk("Делаю снимок")
        ret, frame = cap.read()
        talk("Снимок сделан")
        # Записываем в файл
        talk("Записываю файл")
        cv2.imwrite('C:/Skynet/cam.png', frame)
        talk("Файл сохранен")
        # Отключаем камеру
        talk("Выключаю камеру")
        cap.release()
        talk("Камера выключена")
    elif 'Запиши звук' in zadanie:
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        RECORD_SECONDS = 5
        WAVE_OUTPUT_FILENAME = "C:/Skynet/output.wav"

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("* recording")

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("* done recording")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    elif 'Запиши звук' in zadanie:
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        RECORD_SECONDS = 5
        WAVE_OUTPUT_FILENAME = "C:/Skynet/output.wav"

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("* recording")

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("* done recording")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    elif 'включи эту запись' in zadanie:
        talk("Включаю аудиозапись")
        os.system("start C:/Skynet/output.wav")
    elif 'Включи эту запись' in zadanie:
        talk("Включаю аудиозапись")
        os.system("start C:/Skynet/output.wav")
    elif 'Запиши видео' in zadanie:
        # import cv2
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        RECORD_SECONDS = 5
        WAVE_OUTPUT_FILENAME = "C:/Skynet/output.wav"

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []

        talk("Записываю")
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('C:/Skynet/captured.avi', fourcc, 20.0, (640, 480))
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
            ret, frame = cap.read()
            if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
                 break
            cv2.imshow('frame', frame)
            out.write(frame)
        out.release()
        cap.release()
        cv2.destroyAllWindows()

        stream.stop_stream()
        stream.close()
        p.terminate()
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        # 'ffmpeg -i C:/Skynet/output.wav -i C:/Skynet/Captured.avi C:/Skynet/video_finale.mpg'
        stroka = 'ffmpeg -y -i C:/Skynet/output.wav -r 30 -i C:/Skynet/Captured.avi -filter:a aresample=async=1 -c:a flac -c:v copy C:/Skynet/av.mkv'
        # stroka = 'ffmpeg -i C:/Skynet/output.wav -i C:/Skynet/Captured.avi C:/Skynet/video_finale.mpg'
        # s = subprocess.call(cmd, shell=True)  # "Muxing Done
        subprocess.call(stroka, shell=True)  # "Muxing Done
        # s.release()
        print('Muxing Done')

        talk("Видео записано")
    elif 'запиши видео' in zadanie:
        # import cv2
        talk("Записываю")
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('C:/Skynet/captured.avi', fourcc, 0.5, (640, 480))
        while (cap.isOpened()):
            ret, frame = cap.read()
            if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
                break
            cv2.imshow('frame', frame)
            out.write(frame)
        out.release()
        cap.release()
        talk("Видео записано")
    # elif 'информация в интернете' in zadanie:
    #     talk2("В чем ищем?")
    #     def command1():
    #         r1 = sr.Recognizer()
    #         with sr.Microphone() as source:
    #             # print("Говорите")
    #             talk2("В чем ищем")
    #             r1.pause_threshold = 1
    #             r1.adjust_for_ambient_noise(source, duration=1)
    #             # r.adjust_for_ambient_noise(source)
    #             audio1 = r1.listen(source)
    #         try:
    #             zadanie1 = r1.recognize_google(audio1, language="ru_RU")
    #             print("Вы сказали:" + format(zadanie1))
    #             # zadanie = r.recognize_google(audio)
    #             # print("Вы сказали: " + command)
    #         except sr.UnknownValueError:
    #             talk2("Я Вас не понял")
    #             zadanie1 = command1()
    #         return zadanie1
    #         def makeSomethink1(zadanie1):
    #             if "Яндекс" in zadanie1:
    #                 url = "https://yandex.ru/yandsearch?text="
    #                 talk2("Что ищем?")
    #                 def command2():
    #                     r2 = sr.Recognizer()
    #                     with sr.Microphone() as source:
    #                         # print("Говорите")
    #                         talk2("В чем ищем")
    #                         r2.pause_threshold = 1
    #                         r2.adjust_for_ambient_noise(source, duration=1)
    #                         # r.adjust_for_ambient_noise(source)
    #                         audio2 = r1.listen(source)
    #                     try:
    #                         zadanie2 = r2.recognize_google(audio2, language="ru_RU")
    #                         print("Вы сказали:" + format(zadanie1))
    #                         # zadanie = r.recognize_google(audio)
    #                         # print("Вы сказали: " + command)
    #                     except sr.UnknownValueError:
    #                         talk2("Я Вас не понял")
    #                         zadanie2 = command2()
    #                         return zadanie2
    #                         url+=zadanie2
    #                         talk2("Открываю")
    #                         webbrowser.open(url)
    elif "Найди в Яндексе" in zadanie:
        url = "https://yandex.ru/yandsearch?text="
        s = zadanie
        a = s[15:]
        url+=a
        talk("Открываю")
        webbrowser.open(url)
    elif "Найди в Гугле" in zadanie:
        url = "https://google.com/search?q="
        s = zadanie
        a = s[13:]
        url+=a
        talk("Открываю")
        webbrowser.open(url)
    elif "Найди в Википедии" in zadanie:
        url = "https://ru.wikipedia.org/w/index.php?search="
        s = zadanie
        a = s[17:]
        url+=a
        talk("Открываю")
        webbrowser.open(url)
    # elif "Вычислить" in zadanie:
    #     per = zadanie
    #     perem = per[9:]
    #     peremm = perem.replace(" ", "")
    #     print(peremm)
    #     resh = []
    #     for char in peremm:
    #         if "0" or "1" or "2" or "3" or "4" or "5" or "6" or "7" or "8" or "9" in char:
    #             # a = int(char)
    #             resh.append(int(char))
    #         elif "+" or "-" or "*" or "/" in char:
    #             a = char
    #             resh.append(a)
    #     return resh
    #     print(resh)
    #     talk(resh)
    elif "вычислить" in zadanie:
        per = zadanie
        per1 = zadanie
        if "корень из" in per:
            kor = per[19:]
            k = int(kor)
            koren = math.sqrt(k)
            talk(koren)
        elif "логарифм" in per:
            logarif = per[19:]
            logarifm = int(logarif)
            itog = math.log(logarifm)
            talk(itog)
        # elif "вычислить интеграл методом прямоугольников" in per:
            # integrall = per[40:]
            #         def pryam_integral(f, xmax, xmin, n):
            #             xmax = int(per[43])
            #             xmin = int(per[44])
            #             n = int(per[45])
            #             dx = (xmax - xmin) / n
            #             area = 0
            #             x = xmin
            #             for i in range(n):
            #                 area += dx * f(x)
            #                 x += dx
            #             return area
            #         # integg = ("pryam_integral = {}".format(pryam_integral(fn,0,math.pi/4,10000)))
            #         print(pryam_integral)
            #     res = resu(f, a, b, n)
            #     a = int(per[43])
            #     b = int(per[44])
            #     n = int(per[45])
            #     h = float(b - a) / n
            #     result = f(a + 0.5 * h)
            #     for i in range(1, n):
            #         result += f(a + 0.5 * h + i * h)
            #     result *= h
            #     return result
            #     if result == 0: print(0)
            #     print(result)
        else:
            perem = per1[9:]
            peremm = perem.replace(" ","")
            if "умножитьна" in peremm:
                peremm = peremm.replace("умножитьна", "*")
            elif "разделитьна" in peremm:
                peremm = peremm.replace("разделитьна", "/")
            elif "минус" in peremm:
                peremm.replace("минус", "-")
            print(peremm)
            a = int(peremm[0])
            b = int(peremm[2])
            if "+" in peremm:
                c = a + b
            elif "-" in peremm:
                c = a - b
            elif "*" in peremm:
                c = a * b
            elif "x" in peremm:
                c = a * b
            elif "/" in peremm:
                c = a / b
            talk(c)


        # resh = []
        # for char in peremm:
        #     if "0" in char:
        #         resh.append(int(char))
        #     elif "1" in char:
        #         resh.append(int(char))
        #     elif "2" in char:
        #         resh.append(int(char))
        #     elif "3" in char:
        #         resh.append(int(char))
        #     elif "4" in char:
        #         resh.append(int(char))
        #     elif "5" in char:
        #         resh.append(int(char))
        #     elif "6" in char:
        #         resh.append(int(char))
        #     elif "7" in char:
        #         resh.append(int(char))
        #     elif "8" in char:
        #         resh.append(int(char))
        #     elif "9" in char:
        #         resh.append(int(char))
        #     elif "+" in char:
        #         resh.append(char)
        #     elif "-" in char:
        #         resh.append(char)
        #     elif "*" in char:
        #         resh.append(char)
        #     elif "/" in char:
        #         resh.append(char)
        # return resh
        # print(resh)
        # a = resh[0]
        # b = resh[2]
        # if "+" in resh:
        #     c = a + b
        #     print(c)
        #     talk(c)
        # elif "-" in resh:
        #     c = a - b
        #     print(c)
        #     talk(c)
        # elif "*" in resh:
        #     c = a * b
        #     print(c)
        #     talk(c)
        # elif "/" in resh:
        #     c = a / b
        #     print(c)
        #     talk(c)
        # for i in perem:



        # if not val.strip(): return val
        # return " %zadanie " % eval(val.strip(), {'__builtins__': None})
        # re.sub(r"([0-9\ \.\+\*\-\/(\)]+)", calc, zadanie)



        # d = ""
        # for char in zadanie:
        #     if char="0" in zadanie
        #     elif char="1"
        #     elif char="2" or char="3" or char="4" or char="5" or char="6" or char="7" or char="8" or char="9"



        # per = zadanie
        # perem = per[9:]
        # print(int(perem))
        # talk(perem)
        # print(2+2)
    elif "зашифруй" in zadanie:
        # massiv=['а','б','в','г','д','е','ё','ж','з','и','й','к','л','м','н','о','п','р','с','т','у','ф','х','ц','ч','ш','щ','ъ','ы','ь','э','ю','я']
        massiv = "я юэьыъщшчцхфутсрпонмлкйизжёедгвба0987654321zyxwvutsrqponmlkjihgfedcba"
        tekst1 = zadanie[9:]
        # print(tekst1)
        k = int(tekst1[0])
        tekst = tekst1[1:]
        shifr = ""
        s = 0
        for i in tekst:
            for j in massiv:
                s = s + 1
                if i == j:
                    # print(s)
                    # print(k)
                    # print(s+k)
                    shifr+=massiv[int(s+k)]
                    s = 0
                    break
        print(shifr)
        my_file = open("C:/Skynet/crypt.txt", "w")
        my_file.write(shifr)
        my_file.close()
    elif "открой кодированный файл" in zadanie:
        talk("Открываю")
        os.system("Start C:/Skynet/crypt.txt")
    elif "Открой кодированный файл" in zadanie:
        talk("Открываю")
        os.system("Start C:/Skynet/crypt.txt")
    elif "Расшифруй кодированный файл" in zadanie:
        massiv = "я юэьыъщшчцхфутсрпонмлкйизжёедгвба0987654321zyxwvutsrqponmlkjihgfedcba"
        tekst2 = zadanie[28:]
        k = int(tekst2[0])
        my_crypt = open("C:/Skynet/crypt.txt")
        my_string = my_crypt.read()
        deshifr = ""
        s = 0
        for i in my_string:
            for j in massiv:
                s = s + 1
                if i == j:
                    deshifr+=massiv[int(s-k-2)]
                    s = 0
                    break
        print(deshifr)
        my_file = open("C:/Skynet/decrypt.txt", "w")
        my_file.write(deshifr)
        my_file.close()
    elif "расшифруй кодированный файл" in zadanie:
        massiv = "я юэьыъщшчцхфутсрпонмлкйизжёедгвба0987654321zyxwvutsrqponmlkjihgfedcba"
        k = int(zadanie[28:])
        my_crypt = open("C:/Skynet/crypt.txt")
        my_string = my_crypt.read()
        deshifr = ""
        s = 0
        for i in my_string:
            for j in massiv:
                s = s + 1
                if i == j:
                    deshifr+=massiv[int(s-k-2)]
                    s = 0
                    break
        print(deshifr)
        my_file = open("C:/Skynet/decrypt.txt", "w")
        my_file.write(deshifr)
        my_file.close()
    elif "открой эту информацию в файле" in zadanie:
        talk("Открываю")
        os.system("Start C:/Skynet/decrypt.txt")
    elif "Открой эту информацию в файле" in zadanie:
        talk("Открываю")
        os.system("Start C:/Skynet/decrypt.txt")
    elif 'сколько время' in zadanie:
        now = datetime.datetime.now()
        talk('Сейчас %d часов %d минут' % (now.hour, now.minute))
    elif 'Сколько время' in zadanie:
        now = datetime.datetime.now()
        talk('Сейчас %d часов %d минут' % (now.hour, now.minute))
    elif 'time now' in zadanie:
        now = datetime.datetime.now()
        talk1('Current time is %d hours %d minutes' % (now.hour, now.minute))
    elif 'Time now' in zadanie:
        now = datetime.datetime.now()
        talk1('Current time is %d hours %d minutes' % (now.hour, now.minute))
    elif 'Time Now' in zadanie:
        now = datetime.datetime.now()
        talk1('Current time is %d hours %d minutes' % (now.hour, now.minute))
    elif 'погода' in zadanie:
        reg_ex = re.search('погода (.*)', zadanie)
        if reg_ex:
            city = reg_ex.group(1)
            owm = OWM(API_key='ab0d5e80e8dafb2cb81fa9e82431c1fa')
            obs = owm.weather_at_place(city)
            w = obs.get_weather()
            k = w.get_status()
            x = w.get_temperature(unit='celsius')
            talk(
                'Погода в %s is %s. Максимальная температура %0.2f и минимальная температура %0.2f по цельсию' % (
                city, k, x['temp_max'], x['temp_min']))
    elif 'current weather' in zadanie:
        reg_ex = re.search('current weather in (.*)', zadanie)
        if reg_ex:
            city = reg_ex.group(1)
            owm = OWM(API_key='ab0d5e80e8dafb2cb81fa9e82431c1fa')
            obs = owm.weather_at_place(city)
            w = obs.get_weather()
            k = w.get_status()
            x = w.get_temperature(unit='celsius')
            talk1(
                'Current weather in %s is %s. The maximum temperature is %0.2f and the minimum temperature is %0.2f degree celcius' % (
                city, k, x['temp_max'], x['temp_min']))
    elif 'Google News' in zadanie:
        z = 0
        try:
            news_url = "https://news.google.com/news/rss"
            Client = urlopen(news_url)
            xml_page = Client.read()
            Client.close()
            soup_page = soup(xml_page, "xml")
            news_list = soup_page.findAll("item")
            for news in news_list[:17]:
                z = z + 1
                talk1(news.title.text.encode('utf-8'))
                if z == 5:
                    z = 0
                    break
        except Exception as e:
            print(e)
    elif 'новости в Яндексе' in zadanie:
        mixer.init()
        tts = gTTS(text='пусто', lang='ru')
        tts.save('0.mp3')

        s = requests.get('https://m.news.yandex.ru/world.html')
        b = bs4.BeautifulSoup(s.text, "html.parser")
        p = b.select('.story__title a')

        z = 0
        for x in p:
            print(x.getText())
            tts = gTTS(text=x.getText(), lang='ru')
            tts.save('1.mp3')
            mixer.music.load('1.mp3')
            mixer.music.play()
            while mixer.music.get_busy():
                time.sleep(0.1)
            mixer.music.stop()
            mixer.music.load('0.mp3')
            z = z + 1
            if z == 5:
                z = 0
                break
    elif 'tell me about' in zadanie:
        reg_ex = re.search('tell me about (.*)', zadanie)
        try:
            if reg_ex:
                topic = reg_ex.group(1)
                ny = wikipedia.page(topic)
                talk1(ny.content[:500].encode('utf-8'))
        except Exception as e:
            talk1(e)
    elif 'кто такой' in zadanie:
        reg_ex = re.search('Кто такой (.*)', zadanie)
        try:
            if reg_ex:
                topic = reg_ex.group(1)
                ny = wikipedia.page(topic)
                talk(ny.content[:500].encode('utf-8'))
        except Exception as e:
            talk(e)
    elif 'Что такое' in zadanie:
        reg_ex = re.search('Что такое (.*)', zadanie)
        try:
            if reg_ex:
                topic = reg_ex.group(1)
                ny = wikipedia.page(topic)
                talk2(ny.content[:500].encode('utf-8'))
        except Exception as e:
            talk(e)
    elif 'Расскажи мне о' in zadanie:
        reg_ex = re.search('Расскажи мне о (.*)', zadanie)
        try:
            if reg_ex:
                topic = reg_ex.group(1)
                ny = wikipedia.page(topic)
                talk(ny.content[:500].encode('utf-8'))
        except Exception as e:
            talk(e)
    elif 'запись лица' in zadanie:
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video width
        cam.set(4, 480)  # set video height

        face_detector = cv2.CascadeClassifier('C:/Skynet/haarcascade_frontalface_default.xml')

        # For each person, enter one numeric face id
        face_id = input('\n enter user id end press <return> ==>  ')

        print("\n [INFO] Initializing face capture. Look the camera and wait ...")
        # Initialize individual sampling face count
        count = 0

        while (True):

            ret, img = cam.read()
            # img = cv2.flip(img, -1)  # flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                count += 1

                # Save the captured image into the datasets folder
                cv2.imwrite("C:/Skynet/dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

                cv2.imshow('image', img)

            k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 30:  # Take 30 face sample and stop video
                break

        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
    elif 'тренировка лица' in zadanie:
        path = 'C:/Skynet/dataset'

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("C:/Skynet/haarcascade_frontalface_default.xml");

        # function to get the images and label data
        def getImagesAndLabels(path):

            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faceSamples = []
            ids = []

            for imagePath in imagePaths:

                PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
                img_numpy = np.array(PIL_img, 'uint8')

                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = detector.detectMultiScale(img_numpy)

                for (x, y, w, h) in faces:
                    faceSamples.append(img_numpy[y:y + h, x:x + w])
                    ids.append(id)

            return faceSamples, ids

        print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces, ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
        recognizer.write('C:/Skynet/trainer/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

        # Print the numer of faces trained and end program
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    elif 'распознавание лиц' in zadanie:
        # cascade = 'C:/Skynet/haarcascade_frontalface_alt.xml'
        # c = 1.6
        # Sr = 15
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('C:/Skynet/trainer/trainer.yml')
        cascadePath = "C:/Skynet/haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath);

        font = cv2.FONT_HERSHEY_SIMPLEX

        # iniciate id counter
        id = 0

        # names related to ids: example ==> Marcelo: id=1,  etc
        names = ['None', 'Igor', 'Olga', 'vladimir', 'Sofiya', 'Nikolay', 'Z', 'W']

        # Initialize and start realtime video capture
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video widht
        cam.set(4, 480)  # set video height

        # Define min window size to be recognized as a face
        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)

        while True:

            ret, img = cam.read()
            # img = cv2.flip(img, -1)  # Flip vertically

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            for (x, y, w, h) in faces:

                # bitmap = cv.fromarray(image)
                # faces = cv.HaarDetectObjects(bitmap, cascade, cv.CreateMemStorage(0))
                # k = float(w) / bitmap.cols
                # S = Sr * c / k
                # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 3)
                # cv2.putText(image, 'S=%s' % (S), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))
                # _, frame = cam.read()
                # frame = numpy.asarray(detect(frame))
                # cv2.imshow("features", frame)


                eyeCascade = cv2.CascadeClassifier(
                    "C:/Skynet/OpenCV-Face-Recognition-master-master/OpenCV-Face-Recognition-master-master/FaceDetection/Cascades/haarcascade_eye.xml")
                smileCascade = cv2.CascadeClassifier(
                    "C:/Skynet/OpenCV-Face-Recognition-master-master/OpenCV-Face-Recognition-master-master/FaceDetection/Cascades/haarcascade_smile.xml")

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)



                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]



                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                # Check if confidence is less them 100 ==> "0" is perfect match
                if (confidence < 100):
                    id = names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

                eyes = eyeCascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.5,
                    minNeighbors=5,
                    minSize=(5, 5),
                )

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 69, 255), 2)

                smile = smileCascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.5,
                    minNeighbors=15,
                    minSize=(25, 25),
                )

                for (xx, yy, ww, hh) in smile:
                    cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (15, 185, 255), 2)


            cv2.imshow('camera', img)

            k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video

            if k == 27:
                break

        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
    elif 'определение лица' in zadanie:
        faceCascade = cv2.CascadeClassifier(
                "C:/Skynet/OpenCV-Face-Recognition-master-master/OpenCV-Face-Recognition-master-master/FaceDetection/Cascades/haarcascade_frontalface_default.xml")
        eyeCascade = cv2.CascadeClassifier(
                "C:/Skynet/OpenCV-Face-Recognition-master-master/OpenCV-Face-Recognition-master-master/FaceDetection/Cascades/haarcascade_eye.xml")
        smileCascade = cv2.CascadeClassifier("C:/Skynet/OpenCV-Face-Recognition-master-master/OpenCV-Face-Recognition-master-master/FaceDetection/Cascades/haarcascade_smile.xml")
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)  # set Width
        cap.set(4, 480)  # set Height

        while True:
            ret, img = cap.read()
            # img = cv2.flip(img, -1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]

                eyes = eyeCascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.5,
                    minNeighbors=5,
                    minSize=(5, 5),
                )

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 69, 255), 2)

                smile = smileCascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.5,
                    minNeighbors=15,
                    minSize=(25, 25),
                )

                for (xx, yy, ww, hh) in smile:
                    cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (15, 185, 255), 2)

                cv2.imshow('video', img)

            k = cv2.waitKey(30) & 0xff
            if k == 27:  # press 'ESC' to quit
                break

        cap.release()
        cv2.destroyAllWindows()
    elif 'Найди пост в группе ВКонтакте' in zadanie:
        aid = input('\n Введите адрес группы:')
        keyword = input('\n Введите ключевые слова:')
        filename = input('\n Введите название файла:')
        def take_100_posts():
            token = 'Сервисный ключ доступа'
            version = 5.92
            count = 1
            offset = 0
            all_posts = []
            error = 'error'
            talk('ищу')
            while offset < 5:
                response = requests.get('https://api.vk.com/method/wall.get',
                                params={
                                'access_token': token,
                                'v': version,
                                'domain': aid,
                                'keyword': keyword,
                                'count': count,
                                'offset': offset
                                }
                                )

                data = response.json()['response']['items']
                all_posts.extend(data)
                offset+= 1
                # talk('дальше')
            return all_posts
        def file_writer(data):
            talk('записываю')
            with open("c:/skynet/"+filename+".csv", "w") as file:
                a_pen = csv.writer(file)
                a_pen.writerow(('likes', 'body', 'url'))
                for post in data:
                    if keyword in post['text']:
                        a_pen.writerow((post['likes']['count'], post['text']))
                    else:
                        pass
                        # try:
                        #     if post['attachments'][0]['type']:
                        #         img_url = post['attachments'][0]['photo']['sizes'][-1]['url']
                        #     else:
                        #         img_url = 'pass'
                        # except:
                        #     pass
        all_posts = take_100_posts()
        file_writer(all_posts)
        # open("c:/skynet/"+filename+".csv")
        os.system("Start C:/Skynet/"+filename+".csv")


    elif 'translate to Russian' in zadanie:
        k = (zadanie[21:])
        url = 'https://translate.yandex.net/api/v1.5/tr.json/translate?'
        key = 'APl ключ'
        text = k
        lang = 'en-ru'
        r = requests.post(url, data={'key': key, 'text': text, 'lang': lang}).json()
        # print(r["text"])
        talk(r["text"])

    elif 'перевод на английский' in zadanie:
        k = (zadanie[22:])
        url = 'https://translate.yandex.net/api/v1.5/tr.json/translate?'
        key = 'APl ключ'
        text = k
        lang = 'ru-en'
        r = requests.post(url, data={'key': key, 'text': text, 'lang': lang}).json()
        # print(r["text"])
        talk1(r["text"])








    elif 'Переведи на английский' in zadanie:
        kk = (zadanie[23:])
        # Converts the unicode file to ascii
        def unicode_to_ascii(s):
            return ''.join(c for c in unicodedata.normalize('NFD', s)
                           if unicodedata.category(c) != 'Mn')

        def preprocess_sentence(w):
            w = unicode_to_ascii(w.lower().strip())

            # creating a space between a word and the punctuation following it
            # eg: "he is a boy." => "he is a boy ."
            # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
            w = re.sub(r"([?.!,¿])", r" \1 ", w)
            w = re.sub(r'[" "]+', " ", w)

            # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
            w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

            w = w.rstrip().strip()

            # adding a start and an end token to the sentence
            # so that the model know when to start and stop predicting.
            w = '<start> ' + w + ' <end>'
            return w

        # 1. Remove the accents
        # 2. Clean the sentences
        # 3. Return word pairs in the format: [ENGLISH, SPANISH]

        def create_dataset(path, num_examples):
            lines = open(path, encoding='UTF-8').read().strip().split('\n')

            word_pairs = [[preprocess_sentence(w) for w in l.split(
                '\t')] for l in lines[:num_examples]]

            return word_pairs

        # This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa
        # (e.g., 5 -> "dad") for each language,

        class LanguageIndex():
            def __init__(self, lang):
                self.lang = lang
                self.word2idx = {}
                self.idx2word = {}
                self.vocab = set()

                self.create_index()

            def create_index(self):
                for phrase in self.lang:
                    self.vocab.update(phrase.split(' '))

                self.vocab = sorted(self.vocab)

                self.word2idx['<pad>'] = 0
                for index, word in enumerate(self.vocab):
                    self.word2idx[word] = index + 1

                for word, index in self.word2idx.items():
                    self.idx2word[index] = word

        def max_length(tensor):
            return max(len(t) for t in tensor)

        def load_dataset(path, num_examples):
            # creating cleaned input, output pairs
            pairs = create_dataset(path, num_examples)

            # index language using the class defined above
            inp_lang = LanguageIndex(sp for en, sp in pairs)
            targ_lang = LanguageIndex(en for en, sp in pairs)

            # Vectorize the input and target languages

            # Spanish sentences
            input_tensor = [[inp_lang.word2idx[s]
                             for s in sp.split(' ')] for en, sp in pairs]

            # English sentences
            target_tensor = [[targ_lang.word2idx[s]
                              for s in en.split(' ')] for en, sp in pairs]

            # Calculate max_length of input and output tensor
            # Here, we'll set those to the longest sentence in the dataset
            max_length_inp, max_length_tar = max_length(
                input_tensor), max_length(target_tensor)

            # Padding the input and output tensor to the maximum length
            input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                                         maxlen=max_length_inp,
                                                                         padding='post')

            target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                                                          maxlen=max_length_tar,
                                                                          padding='post')

            return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar

        # Try experimenting with the size of that dataset
        num_examples = 30000
        input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(path_to_file,
                                                                                                         num_examples)
        # Creating training and validation sets using an 80-20 split
        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                        target_tensor,
                                                                                                        test_size=0.2)

        # Show length
        len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val)
        BUFFER_SIZE = len(input_tensor_train)
        BATCH_SIZE = 64
        N_BATCH = BUFFER_SIZE // BATCH_SIZE
        embedding_dim = 256
        units = 1024
        vocab_inp_size = len(inp_lang.word2idx)
        vocab_tar_size = len(targ_lang.word2idx)

        dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

        def gru(units):
            # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
            # the code automatically does that.
            if tf.test.is_gpu_available():
                return tf.keras.layers.CuDNNGRU(units,
                                                return_sequences=True,
                                                return_state=True,
                                                recurrent_initializer='glorot_uniform')
            else:
                return tf.keras.layers.GRU(units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_activation='sigmoid',
                                           recurrent_initializer='glorot_uniform')

        class Encoder(tf.keras.Model):
            def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
                super(Encoder, self).__init__()
                self.batch_sz = batch_sz
                self.enc_units = enc_units
                self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
                self.gru = gru(self.enc_units)

            def call(self, x, hidden):
                x = self.embedding(x)
                output, state = self.gru(x, initial_state=hidden)
                return output, state

            def initialize_hidden_state(self):
                return tf.zeros((self.batch_sz, self.enc_units))

        class Decoder(tf.keras.Model):
            def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
                super(Decoder, self).__init__()
                self.batch_sz = batch_sz
                self.dec_units = dec_units
                self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
                self.gru = gru(self.dec_units)
                self.fc = tf.keras.layers.Dense(vocab_size)

                # used for attention
                self.W1 = tf.keras.layers.Dense(self.dec_units)
                self.W2 = tf.keras.layers.Dense(self.dec_units)
                self.V = tf.keras.layers.Dense(1)

            def call(self, x, hidden, enc_output):
                # enc_output shape == (batch_size, max_length, hidden_size)

                # hidden shape == (batch_size, hidden size)
                # hidden_with_time_axis shape == (batch_size, 1, hidden size)
                # we are doing this to perform addition to calculate the score
                hidden_with_time_axis = tf.expand_dims(hidden, 1)

                # score shape == (batch_size, max_length, 1)
                # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V

                score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
                print(score.shape)
                # attention_weights shape == (batch_size, max_length, 1)
                attention_weights = tf.nn.softmax(score, axis=1)

                # context_vector shape after sum == (batch_size, hidden_size)
                context_vector = attention_weights * enc_output
                context_vector = tf.reduce_sum(context_vector, axis=1)

                # x shape after passing through embedding == (batch_size, 1, embedding_dim)
                x = self.embedding(x)

                # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
                x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

                # passing the concatenated vector to the GRU
                output, state = self.gru(x)

                # output shape == (batch_size * 1, hidden_size)
                output = tf.reshape(output, (-1, output.shape[2]))
                # output shape == (batch_size * 1, vocab)
                x = self.fc(output)

                return x, state, attention_weights

            def initialize_hidden_state(self):
                return tf.zeros((self.batch_sz, self.dec_units))

        encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
        decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
        # optimizer = tf.train.AdamOptimizer()
        # optimizer = tf.compat.v1.train.AdamOptimizer()
        optimizer = tf.optimizers.Adam()
        def loss_function(real, pred):
            mask = 1 - np.equal(real, 0)
            loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=real, logits=pred) * mask
            return tf.reduce_mean(loss_)

        # checkpoint_dir = './training_checkpoints'
        # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        # checkpoint = tf.train.Checkpoint(optimizer=optimizer,
        #                                  encoder=encoder,
        #                                  decoder=decoder)


        # EPOCHS = 10
        #
        # for epoch in range(EPOCHS):
        #     start = time.time()
        #
        #     hidden = encoder.initialize_hidden_state()
        #     total_loss = 0
        #
        #     for (batch, (inp, targ)) in enumerate(dataset):
        #         loss = 0
        #
        #         with tf.GradientTape() as tape:
        #             enc_output, enc_hidden = encoder(inp, hidden)
        #
        #             dec_hidden = enc_hidden
        #             #             print(dec_hidden)
        #             dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)
        #             #             print(dec_input)
        #
        #             # Teacher forcing - feeding the target as the next input
        #             for t in range(1, targ.shape[1]):
        #                 # passing enc_output to the decoder
        #                 predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
        #
        #                 loss += loss_function(targ[:, t], predictions)
        #
        #                 # using teacher forcing
        #                 dec_input = tf.expand_dims(targ[:, t], 1)
        #
        #         batch_loss = (loss / int(targ.shape[1]))
        #
        #         total_loss += batch_loss
        #
        #         variables = encoder.variables + decoder.variables
        #
        #         gradients = tape.gradient(loss, variables)
        #
        #         optimizer.apply_gradients(zip(gradients, variables))
        #
        #         if batch % 100 == 0:
        #             print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
        #                                                          batch,
        #                                                          batch_loss.numpy()))
        #     # saving (checkpoint) the model every 2 epochs
        #     if (epoch + 1) % 2 == 0:
        #         checkpoint.save(file_prefix=checkpoint_prefix)
        #
        #     print('Epoch {} Loss {:.4f}'.format(epoch + 1,
        #                                         total_loss / N_BATCH))
        #     print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

        def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
            attention_plot = np.zeros((max_length_targ, max_length_inp))

            sentence = preprocess_sentence(sentence)

            inputs = [inp_lang.word2idx[i] for i in sentence.split(' ')]
            inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
            inputs = tf.convert_to_tensor(inputs)

            result = ''

            hidden = [tf.zeros((1, units))]
            enc_out, enc_hidden = encoder(inputs, hidden)

            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)

            for t in range(max_length_targ):
                predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

                # storing the attention weights to plot later on
                attention_weights = tf.reshape(attention_weights, (-1,))
                attention_plot[t] = attention_weights.numpy()

                predicted_id = tf.argmax(predictions[0]).numpy()

                result += targ_lang.idx2word[predicted_id] + ' '

                if targ_lang.idx2word[predicted_id] == '<end>':
                    return result, sentence, attention_plot

                # the predicted ID is fed back into the model
                dec_input = tf.expand_dims([predicted_id], 0)

            return result, sentence, attention_plot

        # function for plotting the attention weights
        def plot_attention(attention, sentence, predicted_sentence):
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1)
            ax.matshow(attention, cmap='viridis')

            fontdict = {'fontsize': 14}

            ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
            ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

            plt.show()

        def translate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
            result, sentence, attention_plot = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp,
                                                        max_length_targ)

            # print('Input: {}'.format(sentence))
            # print('Predicted translation: {}'.format(result))

            print(kk.format(sentence))
            print(kk.format(result))




            attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
            plot_attention(attention_plot, sentence.split(' '), result.split(' '))

        # # restoring the latest checkpoint in checkpoint_dir
        # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        translate(u'Здесь очень холодно.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
        print(translate)
        translate(u'Мы тебя любим.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
        print(translate)
        max_length_targ
        translate(kk, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
        print(translate)
        translate(""+kk+"", encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
        print(translate)
        # translate(u'esta es mi vida.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
        # translate(u'todavia estan en casa?', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
        # # wrong translation
        # translate(u'trata de averiguarlo.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)



    elif 'выучи оружие' in zadanie:
        # dimensions of our images.
        img_width, img_height = 150, 150

        train_data_dir = 'data/train'
        validation_data_dir = 'data/validation'
        nb_train_samples = 2000
        nb_validation_samples = 800
        epochs = 5
        batch_size = 16

        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape = input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary')

        validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary')

        model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)

        model.save_weights('first_try.h5')


    elif 'определение объекта' in zadanie:
        faceCascade = cv2.CascadeClassifier(
                "C:\Skynet\inception-2015-12-05\classify_image_graph_def.pb")
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)  # set Width
        cap.set(4, 480)  # set Height

        while True:
            ret, img = cap.read()
            # img = cv2.flip(img, -1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]

                for (xx, yy, ww, hh) in smile:
                    cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (15, 185, 255), 2)

                cv2.imshow('video', img)

            k = cv2.waitKey(30) & 0xff
            if k == 27:  # press 'ESC' to quit
                break

        cap.release()
        cv2.destroyAllWindows()

while True:
    makeSomethink(command())
