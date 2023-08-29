from alpaca import Alpaca, InferenceRequest
from gtts import gTTS
import simpleaudio as sa
import os
from pydub import AudioSegment
import speech_recognition as sr

# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()

# Reading Microphone as source
# listening the speech and store in audio_text variable

with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source, duration=0.2)
    print("Talk")
    audio_text = r.listen(source)
# recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
    spch = r.recognize_google(audio_text)
    try:
        # using google speech recognition
        print("I Heard: "+spch)
    except:
         print("Sorry, I didn't catch that")
         
         
print("Now Loading Model...")

alpaca = Alpaca('./alpaca', 'ggml-alpaca-7b-q4.bin')
try:
    output = alpaca.run_simple(InferenceRequest(input_text=spch))["output"]
finally:
    alpaca.stop()
    
print(output)

mytext = output
  
# Language in which you want to convert
language = 'en'
  
# Passing the text and language to the engine, 
# here we have marked slow=False. Which tells 
# the module that the converted audio should 
# have a high speed
myobj = gTTS(text=mytext, lang=language, slow=False)
  
# Saving the converted audio in a mp3 file named
# welcome 
myobj.save("response.mp3")
src = "response.mp3"
dst = "response.wav"
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")
os.remove("response.mp3") 
  
# Playing the converted file
wave_obj = sa.WaveObject.from_wave_file("response.wav")
play_obj = wave_obj.play()
play_obj.wait_done()
os.remove("response.wav")
print("Done!")
