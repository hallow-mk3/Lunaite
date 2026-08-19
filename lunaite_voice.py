"""
Lunaite AI 10T — Two-Way Voice Communication & Speech Engine
============================================================
Provides real-time Speech-to-Text (STT) and natural Text-to-Speech (TTS)
for seamless, hands-free voice-in and voice-out interaction with Lunaite AI.

Created by Swasthik Shetty for Lunaite AI.
"""

import os
import sys
import re
import time
import queue
import threading
from typing import Optional, Callable

# TTS Engine Imports
try:
    import win32com.client
    HAS_SAPI = True
except ImportError:
    HAS_SAPI = False

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False

# STT Engine Imports
try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False


# ─── 1. TEXT-TO-SPEECH (TTS) SUBSYSTEM ────────────────────────────────────────

class LunaiteTTS:
    """Fast, natural speech synthesis engine with text sanitation."""
    def __init__(self, rate: int = 195, volume: float = 1.0):
        self.rate = rate
        self.volume = volume
        self.sapi_voice = None
        self.pyttsx3_engine = None
        self._speech_queue = queue.Queue()
        self._is_speaking = False
        self._stop_requested = False
        
        # Initialize native Windows SAPI voice first for zero-latency
        if HAS_SAPI:
            try:
                self.sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
                # Rate is -10 to +10 in SAPI
                sapi_rate = int((rate - 150) / 15)
                sapi_rate = max(-10, min(10, sapi_rate))
                self.sapi_voice.Rate = sapi_rate
                self.sapi_voice.Volume = int(volume * 100)
            except Exception as e:
                self.sapi_voice = None

        # Fallback to pyttsx3
        if not self.sapi_voice and HAS_PYTTSX3:
            try:
                self.pyttsx3_engine = pyttsx3.init()
                self.pyttsx3_engine.setProperty('rate', rate)
                self.pyttsx3_engine.setProperty('volume', volume)
            except Exception:
                self.pyttsx3_engine = None

        # Start background worker thread for non-blocking speech
        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker_thread.start()

    @staticmethod
    def sanitize_for_speech(text: str) -> str:
        """Strip code blocks, markdown symbols, and URLs for fluid spoken audio."""
        if not text:
            return ""
        
        # Remove code blocks entirely or summarize them
        text = re.sub(r'```.*?```', ' I have written the code block for you. ', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove tool tags
        text = re.sub(r'<tool:[^>]+>.*?</tool:[^>]+>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://[^\s]+', ' web link ', text)
        
        # Remove markdown symbols
        text = re.sub(r'[#*_~>\[\]|]', '', text)
        text = re.sub(r'\(file:///[^\)]+\)', '', text)
        
        # Replace multiple newlines or spaces
        text = re.sub(r'\n+', '. ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _speech_worker(self):
        while True:
            text = self._speech_queue.get()
            if text is None or self._stop_requested:
                self._speech_queue.task_done()
                continue
            
            self._is_speaking = True
            try:
                if self.sapi_voice:
                    # SAPI speak (1 = SVSFlagsAsync)
                    self.sapi_voice.Speak(text, 0)
                elif self.pyttsx3_engine:
                    self.pyttsx3_engine.say(text)
                    self.pyttsx3_engine.runAndWait()
                else:
                    # Windows PowerShell Speech fallback
                    escaped = text.replace('"', '`"').replace("'", "''")
                    ps_cmd = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{escaped}")'
                    import subprocess
                    subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True)
            except Exception as e:
                pass
            finally:
                self._is_speaking = False
                self._speech_queue.task_done()

    def speak(self, text: str, block: bool = False):
        """Speak the given text."""
        clean_text = self.sanitize_for_speech(text)
        if not clean_text:
            return
        
        if block:
            # Synchronous speak
            if self.sapi_voice:
                self.sapi_voice.Speak(clean_text, 0)
            elif self.pyttsx3_engine:
                self.pyttsx3_engine.say(clean_text)
                self.pyttsx3_engine.runAndWait()
            else:
                self._speech_queue.put(clean_text)
                self._speech_queue.join()
        else:
            self._speech_queue.put(clean_text)

    def stop(self):
        """Halt any ongoing speech."""
        try:
            while not self._speech_queue.empty():
                self._speech_queue.get_nowait()
                self._speech_queue.task_done()
        except Exception:
            pass


tts_engine = LunaiteTTS()


# ─── 2. SPEECH-TO-TEXT (STT) SUBSYSTEM ────────────────────────────────────────

class LunaiteSTT:
    """Robust Speech-to-Text Listener with microphone calibration."""
    def __init__(self):
        self.recognizer = sr.Recognizer() if HAS_SR else None
        if self.recognizer:
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            self.recognizer.phrase_threshold = 0.3
            self.recognizer.non_speaking_duration = 0.5

    def calibrate_ambient_noise(self, duration: float = 1.0):
        """Calibrate microphone for room background noise."""
        if not HAS_SR:
            return
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=duration)
        except Exception as e:
            pass

    def listen(self, timeout: int = 8, phrase_time_limit: int = 15) -> Optional[str]:
        """Listen to the microphone and return transcribed text."""
        if not HAS_SR:
            print("\033[91m[STT Warning]: SpeechRecognition is not installed.\033[0m")
            return None

        try:
            with sr.Microphone() as source:
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
            
            # Use Google Speech Recognition API (fast and high accuracy)
            try:
                text = self.recognizer.recognize_google(audio, language="en-US")
                return text.strip()
            except sr.UnknownValueError:
                return None
            except sr.RequestError:
                # Fallback to local Windows Speech / Sphinx if available
                try:
                    return self.recognizer.recognize_sphinx(audio).strip()
                except Exception:
                    return None

        except (KeyboardInterrupt, SystemExit):
            raise
        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            return None


stt_engine = LunaiteSTT()


# ─── 3. HIGH LEVEL VOICE API ──────────────────────────────────────────────────

def speak_out(text: str, block: bool = False):
    """Global helper to speak out text with Lunaite's voice."""
    tts_engine.speak(text, block=block)


def listen_voice(timeout: int = 8, phrase_time_limit: int = 15) -> Optional[str]:
    """Global helper to capture and transcribe user voice."""
    return stt_engine.listen(timeout=timeout, phrase_time_limit=phrase_time_limit)


# ─── 4. CONTINUOUS VOICE AGENT LOOP ───────────────────────────────────────────

def run_voice_assistant(query_callback: Callable[[str], str], wake_word: Optional[str] = None):
    """
    Run continuous hands-free voice loop:
    User speaks -> Lunaite transcribes -> Agent processes -> Lunaite speaks reply.
    """
    print("\033[96m\033[1m" + "=" * 65 + "\033[0m")
    print("\033[97m\033[1m 🎙️  LUNAITE AI 10T — HANDS-FREE VOICE AGENT ACTIVE\033[0m")
    print("\033[90m Speak naturally into your microphone to control apps and chat.\033[0m")
    print("\033[90m Say 'exit', 'quit', or 'goodbye' to end voice session.\033[0m")
    print("\033[96m\033[1m" + "=" * 65 + "\033[0m\n")

    speak_out("Lunaite Voice Assistant activated. I am listening.")
    
    # Noise calibration
    print("\033[36m[Calibrating microphone for background noise...]\033[0m")
    stt_engine.calibrate_ambient_noise(1.2)
    print("\033[92m[Microphone Ready!]\033[0m\n")

    while True:
        try:
            print("\033[96m● [Listening...] 🎙️\033[0m", end="\r", flush=True)
            user_speech = listen_voice(timeout=10, phrase_time_limit=20)
            
            if not user_speech:
                continue

            print(f"\n\033[97m\033[1m👤 You said:\033[0m \033[96m\"{user_speech}\"\033[0m")

            # Check exit
            if user_speech.lower().strip() in ["exit", "quit", "goodbye", "stop listening", "terminate"]:
                print("\n\033[93mShutting down voice mode. Have a great day!\033[0m")
                speak_out("Goodbye! Shutting down voice mode.", block=True)
                break

            # Process with callback
            print("\033[95m🧠 Lunaite Thinking & Executing...\033[0m")
            response = query_callback(user_speech)

            # Speak out response
            print(f"\n\033[92m\033[1m🌙 Lunaite Replied & Spoke:\033[0m\n{response}\n")
            speak_out(response, block=False)

        except KeyboardInterrupt:
            print("\n\033[93m[Voice session interrupted by user]\033[0m")
            speak_out("Voice session paused.", block=True)
            break
        except Exception as e:
            print(f"\n\033[91m[Voice Loop Error]: {e}\033[0m")
            time.sleep(1)
