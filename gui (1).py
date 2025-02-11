import os
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
from audio_processing import process_audio, clean_temp_files
import torch
import threading
import vlc
import time
import pyaudio
import wave
import numpy as np
import soundfile as sf
import shutil
from tkinter.ttk import Progressbar
from tkinter import messagebox

class DiarizationApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        clean_temp_files()

        self.title("Audio Diarization and Transcription")
        screen_height = self.winfo_screenheight()
        self.geometry(f"1200x{screen_height-100}+0+0")
        self.minsize(1200, 600)
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.audio_path = None
        self.transcriptions = []
        self.speaker_mapping = {}
        self.speaker_files = {}
        self.model_choice = tk.StringVar(value="small")  # Default model
        self.number_of_speakers = tk.StringVar(value="Auto")

        self.create_widgets()
        self.check_cuda_availability()
        self.current_position = 0
        self.is_recording = False
        self.audio_stream = None
        self.audio_frames = []

    def create_widgets(self):
        # Cadre gauche
        left_frame = ctk.CTkFrame(self, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)



        self.audio_label = ctk.CTkLabel(left_frame, text="Load an audio file for processing or Start a Live Session")
        self.audio_label.pack(pady=10)

        load_audio_frame = ctk.CTkFrame(left_frame)
        load_audio_frame.pack(fill="x", padx=10, pady=5)

        self.select_button = ctk.CTkButton(load_audio_frame, text="Load Audio File", command=self.load_audio_file)
        self.select_button.pack(side=tk.LEFT, padx=40, pady=10)

        self.start_button = ctk.CTkButton(
            load_audio_frame,
            text="Start Record",
            command=self.start_recording,
            fg_color="#FF0000",  # Rouge
            hover_color="#CC0000",
            text_color="white"
        )
        self.start_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Bouton Pause Record en jaune
        pause_button = ctk.CTkButton(
            load_audio_frame,
            text="Pause/Resume Record",
            command=self.toggle_pause_resume,
            fg_color="#FFFF00",  # Jaune
            hover_color="#CCCC00",
            text_color="black"
        )
        pause_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Bouton Stop Record en noir
        stop_button = ctk.CTkButton(
            load_audio_frame,
            text="Stop Record",
            command=self.stop_recording,
            fg_color="#000000",  # Noir
            hover_color="#333333",
            text_color="white"
        )
        stop_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Cadre pour les contrôles audio en dessous
        audio_controls_frame = ctk.CTkFrame(left_frame)
        audio_controls_frame.pack( fill="x", padx=10, pady=10)

        # Label pour afficher la position en minutes:secondes
        self.audio_slider_label = ctk.CTkLabel(audio_controls_frame, text="0m0s", font=("Arial", 12))
        self.audio_slider_label.pack(side=tk.TOP, pady=5)

        self.audio_slider = tk.Scale(
            audio_controls_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            length=300,
            command=self.update_audio_position,
            showvalue=0
        )
        self.audio_slider.pack(side=tk.TOP, fill=tk.X, expand=True, padx=5)

        # Lier l'événement pour détecter quand l'utilisateur relâche le curseur
        self.audio_slider.bind("<ButtonRelease-1>", self.on_slider_release)

        # Boutons Play et Stop centrés
        button_frame = ctk.CTkFrame(audio_controls_frame)
        button_frame.pack(side=tk.TOP, fill=tk.X, pady=5)  # Cadre pour les boutons

        # Ajouter des espaces flexibles pour centrer les boutons
        button_frame.columnconfigure(0, weight=1)  # Espace à gauche
        button_frame.columnconfigure(1, weight=0)  # Bouton Play
        button_frame.columnconfigure(2, weight=0)  # Bouton Stop
        button_frame.columnconfigure(3, weight=1)  # Espace à droite

        self.play_original_button = ctk.CTkButton(button_frame, text="Play Main Audio",
                                                  command=self.play_original_audio, width=100)
        self.play_original_button.grid(row=0, column=1, padx=10, pady=5)

        self.stop_original_button = ctk.CTkButton(button_frame, text="Stop",
                                                  command=self.stop_original_audio, width=100)
        self.stop_original_button.grid(row=0, column=2, padx=10, pady=5)

        # Centrer le cadre contenant les boutons
        button_frame.pack(fill=tk.X, pady=5)
        button_frame.pack_propagate(False)  # Désactiver la propagation pour maintenir le cadre






        # Initialisation de `self.processing_enabled`
        self.postprocessing_enabled = tk.BooleanVar(value=False)
        self.postprocessing_toggle = ctk.CTkSwitch(
            left_frame,
            text="Enable Post Processing Filter",
            variable=self.postprocessing_enabled,
            onvalue=True,
            offvalue=False
        )
        # Ajout du commutateur au cadre avec pack()
        self.postprocessing_toggle.pack(pady=10)

        self.postprocessing_enabled.trace_add("write", lambda *args: self.process_and_normalize_audio())

        self.model_label = ctk.CTkLabel(left_frame, text="Select the model for Speech To Text")
        self.model_label.pack(pady=10)  # Assurez que ce widget est bien ajouté avec pack

        self.model_menu = ctk.CTkOptionMenu(
            left_frame,
            variable=self.model_choice,
            values=["tiny", "base", "small", "medium", "large"],
            command=self.show_model_description
        )
        self.model_menu.pack(pady=10)  # Ajout correct avec pack

        self.model_description = ctk.CTkLabel(
            left_frame,
            text="",
            wraplength=600,  # Largeur pour les longues descriptions
            justify="center",
            anchor="center",
            font=("Arial", 12)
        )
        self.model_description.pack(pady=10, fill=tk.X)  # Description étendue sur la largeur

        # Initialisation de `self.diarization_enabled`
        self.diarization_enabled = tk.BooleanVar(value=True)

        # Cadre pour "Enable Voice Separation" et "Nombre de locuteurs"
        toggle_frame = ctk.CTkFrame(left_frame)
        toggle_frame.pack(fill=tk.X, padx=30, pady=10)

        self.diarization_toggle = ctk.CTkSwitch(
            toggle_frame, text="Enable Voice Separation", variable=self.diarization_enabled, onvalue=True,
            offvalue=False
        )
        self.diarization_toggle.grid(row=0, column=0, padx=(0, 10))  # Place à gauche

        num_speakers_menu = ctk.CTkOptionMenu(
            toggle_frame,
            variable=self.number_of_speakers,
            values=["Auto", "1", "2", "3", "4", "5" , "6", "7", "8", "9", "10"]  # Options possibles
        )

        num_speakers_label = ctk.CTkLabel(toggle_frame, text="Number of Speakers: ")
        num_speakers_label.grid(row=0, column=1, padx=(10, 5))  # Place à droite de la switch

        # Ajustez la position de la boîte de sélection
        num_speakers_menu.grid(row=0, column=2, padx=(5, 0))  # Place après le label


        self.process_button = ctk.CTkButton(left_frame, text="Start Processing", command=self.start_processing,
                                            state="disabled")
        self.process_button.pack(pady=10)

        # Ajouter la barre de progression
        self.progress_bar = Progressbar(left_frame, orient="horizontal", mode="determinate", length=300)
        self.progress_bar.pack(pady=10)

        self.export_button = ctk.CTkButton(left_frame, text="Export Results", command=self.save_results,
                                           state="disabled")
        self.export_button.pack(pady=10)

        self.cuda_status_label = ctk.CTkLabel(left_frame, text="Checking CUDA availability...")
        self.cuda_status_label.pack(pady=10)

        # Panneau de droite
        right_frame = ctk.CTkFrame(self, border_width=2, corner_radius=10,
                                   fg_color="#003366")  # Bordure et couleur de fond
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)



        # Section supérieure : Texte transcrit
        transcription_frame = ctk.CTkFrame(right_frame)
        transcription_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        transcription_label = ctk.CTkLabel(transcription_frame, text="Transcription", font=("Arial", 12, "bold"),
                                           text_color="white", fg_color="#005FA1", corner_radius=5)
        transcription_label.pack(fill=tk.X, pady=5)

        self.transcription_text = tk.Text(transcription_frame, wrap="word", font=("Arial", 14))
        self.transcription_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        transcription_scrollbar = tk.Scrollbar(transcription_frame, orient=tk.VERTICAL,
                                               command=self.transcription_text.yview)
        transcription_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.transcription_text.config(yscrollcommand=transcription_scrollbar.set)

        # Section intermédiaire : Entrée pour le prompt
        prompt_frame = ctk.CTkFrame(right_frame, fg_color="#003366", corner_radius=10)  # Style pour le prompt
        prompt_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        prompt_label = ctk.CTkLabel(prompt_frame, text="Type your prompt here for AI", font=("Arial", 12, "bold"),
                                    text_color="white", fg_color="#005FA1", corner_radius=5)
        prompt_label.pack(fill=tk.X, pady=5)

        self.prompt_entry = ctk.CTkEntry(prompt_frame, placeholder_text="Enter your prompt here...")
        self.prompt_entry.pack(fill=tk.X, padx=10, pady=5)

        # Section inférieure : Résultat généré
        result_frame = ctk.CTkFrame(right_frame)
        result_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self.result_text = tk.Text(result_frame, wrap="word", font=("Arial", 14))
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        result_scrollbar = tk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        result_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=result_scrollbar.set)

        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        listbox_frame = ctk.CTkFrame(left_frame)
        listbox_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.speaker_listbox = tk.Listbox(listbox_frame, height=8, selectmode=tk.SINGLE, font=("Arial", 14))
        self.speaker_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.speaker_listbox.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.speaker_listbox.config(yscrollcommand=self.scrollbar.set)

        button_frame = ctk.CTkFrame(left_frame)
        button_frame.pack(fill="x", padx=10, pady=5)

        self.name_entry = ctk.CTkEntry(button_frame, placeholder_text="Enter a new name for the speaker")
        self.name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

        self.rename_button = ctk.CTkButton(button_frame, text="Rename", command=self.rename_speaker)
        self.rename_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.play_button = ctk.CTkButton(button_frame, text="Play Speaker Audio", command=self.play_speaker_audio)
        self.play_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.stop_button = ctk.CTkButton(button_frame, text="Stop", command=self.stop_audio)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)

    def toggle_record_button_color(self):
        """Fait clignoter le bouton Start Record pendant l'enregistrement."""
        if self.is_recording:
            # Alterne entre rouge et gris
            current_color = self.start_button.cget("fg_color")
            new_color = "#FF0000" if current_color != "#FF0000" else "#333333"
            self.start_button.configure(fg_color=new_color)
            # Relancer la fonction après 500 ms
            self.after(500, self.toggle_record_button_color)
        else:
            # Réinitialiser la couleur à rouge lorsque l'enregistrement s'arrête
            self.start_button.configure(fg_color="#FF0000")
    def reset_interface(self):
        """Réinitialise l'interface après confirmation de l'utilisateur."""
        confirm = messagebox.askyesno(
            "Confirmation",
            "Êtes-vous sûr de vouloir réinitialiser l'interface ? Toutes les données non enregistrées seront perdues."
        )
        if confirm:
            # Réinitialiser les variables et widgets
            self.audio_path = None
            self.transcriptions = []
            self.speaker_mapping = {}
            self.speaker_files = {}
            self.current_position = 0
            self.audio_slider.set(0)
            self.audio_slider_label.configure(text="0m0s")
            self.transcription_text.delete("1.0", tk.END)
            self.populate_speaker_list()
            self.audio_label.configure(text="Interface réinitialisée.")
            if hasattr(self, 'vlc_player'):
                self.vlc_player.stop()
                self.vlc_player = None

    def play_original_audio(self):
        """Lit le fichier audio principal."""
        if not self.audio_path:
            self.audio_label.configure(text="No audio file loaded.")
            return

        try:
            if not hasattr(self, 'vlc_player'):
                self.vlc_player = vlc.MediaPlayer(self.audio_path)

            self.vlc_player.play()
            self.audio_label.configure(text=f"Playing: {os.path.basename(self.audio_path)}")
        except Exception as e:
            self.audio_label.configure(text=f"Error playing audio: {str(e)}")
    def play_audio(self):
        """Joue le fichier audio chargé."""
        if not self.audio_path:
            self.audio_label.configure(text="No audio file loaded.")
            return

        try:
            if not hasattr(self, 'vlc_player'):
                self.vlc_player = vlc.MediaPlayer(self.audio_path)

            # Démarre la lecture
            self.vlc_player.play()
            self.audio_label.configure(text=f"Playing: {os.path.basename(self.audio_path)}")
        except Exception as e:
            self.audio_label.configure(text=f"Error playing audio: {str(e)}")

    def play_speaker_audio(self):
        """Lit le fichier audio associé à un locuteur sélectionné."""
        selection = self.speaker_listbox.curselection()
        if not selection:
            self.audio_label.configure(text="No speaker selected.")
            return

        # Récupérer le nom du locuteur sélectionné
        selected_speaker = self.speaker_listbox.get(selection[0])

        # Récupérer les fichiers associés au locuteur
        audio_files = self.speaker_files.get(selected_speaker)

        if not audio_files:
            self.audio_label.configure(text=f"No audio files found for {selected_speaker}.")
            return

        # Jouer le premier fichier du locuteur
        speaker_audio_path = audio_files[0]
        try:
            if not hasattr(self, 'vlc_player') or self.vlc_player is None:
                self.vlc_player = vlc.MediaPlayer(speaker_audio_path)
            else:
                self.vlc_player.set_mrl(speaker_audio_path)

            self.vlc_player.play()
            self.audio_label.configure(text=f"Playing segment for: {selected_speaker}")
        except Exception as e:
            self.audio_label.configure(text=f"Error playing speaker audio: {str(e)}")
    def update_audio_slider(self):
        """Met à jour le slider en fonction de la position actuelle de l'audio."""
        if self.vlc_player.is_playing():
            current_time = self.vlc_player.get_time() / 1000  # En secondes
            total_length = self.vlc_player.get_length() / 1000  # En secondes
            slider_position = (current_time / total_length) * 100 if total_length > 0 else 0
            self.audio_slider.set(slider_position)

            # Relancer la mise à jour après 500 ms
            self.after(500, self.update_audio_slider)

    def generate_summary(self):
        """Méthode placeholder pour la génération de résumé."""
        pass
    def process_and_normalize_audio(self):
        """Traite et normalise un fichier audio si le post-processing est activé."""
        if not self.audio_path:
            self.audio_label.configure(text="No audio file to process.")
            return

        try:
            # Vérifier si le post-processing est activé
            if self.postprocessing_enabled.get():
                # Chargement du fichier audio
                data, samplerate = sf.read(self.audio_path)

                # Si le fichier est stéréo, convertissez-le en mono (moyenne des canaux)
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)

                # Normalisation des niveaux audio entre -1 et 1
                max_amplitude = np.max(np.abs(data))
                if max_amplitude > 0:
                    data = data / max_amplitude

                # Sauvegarde du fichier normalisé
                normalized_file = "normalized_audio.wav"
                sf.write(normalized_file, data, samplerate)

                # Mettre à jour le chemin pour pointer vers le fichier normalisé
                self.audio_path = normalized_file
                self.audio_label.configure(text=f"Audio normalized and saved as {normalized_file}")

            else:
                # Si désactivé, récupérer le fichier original
                if os.path.exists("original_audio.wav"):
                    self.audio_path = "original_audio.wav"
                    self.audio_label.configure(text="Using the original audio file.")
                else:
                    self.audio_label.configure(text="Original file not found. Normalization disabled.")

            # Réinitialisation du lecteur VLC avec le fichier actuel (normalisé ou original)
            self.vlc_player = vlc.MediaPlayer(self.audio_path)
            self.process_button.configure(state="normal")

        except Exception as e:
            self.audio_label.configure(text=f"Error during processing: {str(e)}")

    def update_audio_progress(self):
        """Met à jour le curseur et le temps en fonction de la position actuelle."""
        if hasattr(self, 'vlc_player') and self.vlc_player.is_playing():
            # Obtenir la position actuelle en millisecondes
            current_time = self.vlc_player.get_time() / 1000  # Convertir en secondes
            total_length = self.vlc_player.get_length() / 1000  # Convertir en secondes

            # Mettre à jour le curseur
            if total_length > 0:
                slider_position = (current_time / total_length) * 100
                self.audio_slider.set(slider_position)

            # Mettre à jour le label du temps
            minutes = int(current_time // 60)
            seconds = int(current_time % 60)
            self.audio_slider_label.configure(text=f"{minutes}m{seconds}s")

            # Relancer la mise à jour après 500 ms
            self.after(500, self.update_audio_progress)

    def on_slider_release(self, event):
        """Reprend la lecture à la position définie par le curseur."""
        if self.audio_path and hasattr(self, 'vlc_player'):
            try:
                # Convertir la position en millisecondes
                new_position_ms = int(self.current_position * 1000)
                self.vlc_player.set_time(new_position_ms)

                # Reprendre la lecture
                if not self.vlc_player.is_playing():
                    self.vlc_player.play()

                self.audio_label.configure(
                    text=f"Playing from {int(self.current_position // 60)}m{int(self.current_position % 60)}s.")
            except Exception as e:
                self.audio_label.configure(text=f"Error setting new position: {str(e)}")
    def update_audio_position(self, value):
        """Mise à jour de la position de lecture en fonction du curseur."""
        try:
            if self.audio_path:
                total_length = self.vlc_player.get_length() / 1000  # En secondes
                self.current_position = (float(value) / 100) * total_length

                # Mettre à jour le label
                minutes = int(self.current_position // 60)
                seconds = int(self.current_position % 60)
                self.audio_slider_label.configure(text=f"{minutes}m{seconds}s")
        except Exception as e:
            self.audio_label.configure(text=f"Error updating position: {str(e)}")

    def start_recording(self):
        """Démarre l'enregistrement audio en direct après confirmation de réinitialisation."""
        if self.audio_path or self.transcriptions:
            confirm = messagebox.askyesno(
                "Confirmation",
                "Êtes-vous sûr de vouloir réinitialiser l'interface et démarrer un nouvel enregistrement ? "
                "Toutes les données non enregistrées seront perdues."
            )
            if confirm:
                # Réinitialiser l'interface
                self.reset_interface()
            else:
                return  # Annule l'action si l'utilisateur refuse de réinitialiser

        # Démarre un nouvel enregistrement
        if self.is_recording:
            self.audio_label.configure(text="Already recording.")
            return

        self.is_recording = True
        self.toggle_record_button_color()
        self.audio_frames = []

        # Configuration de PyAudio
        self.audio = pyaudio.PyAudio()
        self.audio_stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024
        )
        self.audio_label.configure(text="Recording started.")

        # Lance le thread pour capturer l'audio
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()

    def record_audio(self):
        """Capturer l'audio tant que l'enregistrement est actif."""
        while self.is_recording:
            data = self.audio_stream.read(1024)
            self.audio_frames.append(data)

    def pause_recording(self):
        """Suspendre temporairement l'enregistrement audio."""
        if not self.is_recording:
            self.audio_label.configure(text="Recording is not active.")
            return

        self.is_recording = False
        if self.audio_stream is not None:
            self.audio_stream.stop_stream()  # Met en pause le flux sans le fermer
        self.audio_label.configure(text="Recording paused.")

    def toggle_pause_resume(self):
        """Alterner entre pause et reprise de l'enregistrement."""
        if self.is_recording:
            self.pause_recording()
        else:
            self.resume_recording()

    from scipy.signal import butter, filtfilt
    import soundfile as sf

    def high_pass_filter(audio_path, cutoff_freq, output_path):
        """
        Applique un filtre passe-haut à un fichier audio.

        Args:
            audio_path (str): Chemin du fichier audio d'entrée.
            cutoff_freq (float): Fréquence de coupure du filtre en Hz.
            output_path (str): Chemin du fichier audio filtré en sortie.

        Returns:
            None
        """
        try:
            # Charger l'audio
            data, samplerate = sf.read(audio_path)

            # Si l'audio est stéréo, convertir en mono
            if len(data.shape) > 1:
                data = data.mean(axis=1)  # Moyenne des canaux

            # Normalisation de la fréquence de coupure
            nyquist = 0.5 * samplerate
            normalized_cutoff = cutoff_freq / nyquist

            # Créer un filtre Butterworth passe-haut
            b, a = butter(N=4, Wn=normalized_cutoff, btype='high', analog=False)

            # Appliquer le filtre
            filtered_data = filtfilt(b, a, data)

            # Sauvegarder l'audio filtré
            sf.write(output_path, filtered_data, samplerate)
            print(f"Filtre passe-haut appliqué avec succès. Fichier sauvegardé à : {output_path}")

        except Exception as e:
            print(f"Erreur lors de l'application du filtre passe-haut : {e}")

    def resume_recording(self):
        """Reprendre l'enregistrement audio après une pause."""
        if self.is_recording:
            self.audio_label.configure(text="Recording is already active.")
            return

        self.is_recording = True

        # Recréez le flux si nécessaire
        if self.audio_stream is None or self.audio_stream.is_stopped():
            self.audio_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                frames_per_buffer=1024
            )

        self.audio_label.configure(text="Recording started.")
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()

        # Recommencer le clignotement
        self.toggle_record_button_color()

    def stop_recording(self):
        """Arrêter l'enregistrement audio."""
        if not self.audio_stream:
            self.audio_label.configure(text="No recording in progress.")
            return

        self.is_recording = False
        self.start_button.configure(fg_color="#FF0000")
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.audio_stream = None  # Assurez-vous que le flux est réinitialisé
        self.audio.terminate()

        # Sauvegarder l'audio dans un fichier WAV temporaire
        temp_file = "live_session.wav"
        try:
            wf = wave.open(temp_file, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b''.join(self.audio_frames))
            wf.close()
            self.audio_label.configure(text="Recording saved as a temporary file.")

            # Proposer de sauvegarder le fichier avec un nom personnalisé
            save_file = filedialog.asksaveasfilename(
                title="Save Recording As",
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
            )
            if save_file:
                shutil.move(temp_file, save_file)
                self.audio_label.configure(text=f"Recording saved as: {os.path.basename(save_file)}")
                self.audio_path = save_file
            else:
                self.audio_label.configure(text="Recording not saved, using temporary file.")
                self.audio_path = temp_file

            # Charger automatiquement le fichier enregistré
            self.vlc_player = vlc.MediaPlayer(self.audio_path)  # Initialise le lecteur VLC
            self.process_button.configure(state="normal")
        except Exception as e:
            self.audio_label.configure(text=f"Error saving or loading recording: {str(e)}")

    def update_progress_bar(self, value):
        """Met à jour la barre de progression avec une valeur entre 0 et 100."""
        self.progress_bar["value"] = value
        self.update_idletasks()  # Met à jour l'interface utilisateur
    def play_original_audio(self):
        """Démarre la lecture à partir de la position actuelle."""
        if not self.audio_path:
            self.audio_label.configure(text="No audio file loaded.")
            return

        try:
            if not hasattr(self, 'vlc_player'):
                self.vlc_player = vlc.MediaPlayer(self.audio_path)

            # Démarrer la lecture
            self.vlc_player.play()

            # Attendre brièvement pour initialiser
            time.sleep(0.1)

            # Positionner à la position actuelle en millisecondes
            self.vlc_player.set_time(int(self.current_position * 1000))

            # Démarrer la mise à jour du curseur et du temps
            self.update_audio_progress()

            minutes = int(self.current_position // 60)
            seconds = int(self.current_position % 60)
            self.audio_label.configure(text=f"Playing from {minutes}m{seconds}s.")
        except Exception as e:
            self.audio_label.configure(text=f"Error playing audio: {str(e)}")

    def stop_original_audio(self):
        """Arrête la lecture sans réinitialiser la position."""
        if hasattr(self, 'vlc_player') and self.vlc_player.is_playing():
            self.vlc_player.stop()
            self.audio_label.configure(text="Audio stopped.")

        # Réinitialiser l'état du slider
        self.audio_slider.set(0)
        self.audio_slider_label.configure(text="0m0s")
    def check_cuda_availability(self):
        """Check if CUDA is available and update the label."""
        if torch.cuda.is_available():
            self.cuda_status_label.configure(text="CUDA available", text_color="green")
        else:
            self.cuda_status_label.configure(text="CUDA unavailable", text_color="red")

    def show_model_description(self, model):
        """Display model description based on selection."""
        descriptions = {
            "tiny": "Tiny (~75 MB): Ultra-light model for fast but less precise transcriptions.",
            "base": "Base (~145 MB): Balanced between speed and quality, ideal for general cases.",
            "small": "Small (~502 MB): Precise model for high-quality multilingual transcriptions.",
            "medium": "Medium (~1.42 GB): High precision, suitable for accents and complex environments.",
            "large": "Large (~2.87 GB): The most precise model, recommended for complex transcriptions.",
            "Medical-Whisper": "Medical-Whisper: Specialized model for medical audio transcription."
        }
        self.model_description.configure(text=descriptions.get(model, ""))

    def format_duration(self, seconds):
        """Convert duration in seconds to 'XmYs' format."""
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m{remaining_seconds}s"

    def populate_speaker_list(self):
        """Add speakers to the list."""
        self.speaker_listbox.delete(0, "end")

        if not self.diarization_enabled.get():
            self.speaker_listbox.insert("end", "Speaker 1")
        else:
            speakers = list(set([self.speaker_mapping.get(spk, spk) for _, _, spk, _ in self.transcriptions]))
            for speaker in speakers:
                self.speaker_listbox.insert("end", speaker)

    def rename_speaker(self):
        """Rename a speaker."""
        selection = self.speaker_listbox.curselection()
        if not selection:
            self.audio_label.configure(text="No speaker selected for renaming.")
            return

        idx = selection[0]
        speaker = self.speaker_listbox.get(idx)
        new_name = self.name_entry.get().strip()

        if not new_name:
            self.audio_label.configure(text="Rename field is empty.")
            return

        # Mettre à jour le mapping des speakers
        self.speaker_mapping[speaker] = new_name
        if speaker in self.speaker_files:
            self.speaker_files[new_name] = self.speaker_files.pop(speaker)

        # Rafraîchir la liste des speakers
        self.populate_speaker_list()

        # Effacer et réinsérer les transcriptions avec le format correct
        self.transcription_text.delete("1.0", tk.END)

        for i, (start, end, spk, text) in enumerate(self.transcriptions):
            if spk == speaker:
                self.transcriptions[i] = (start, end, new_name, text)

            # Conversion des durées au format minutes:secondes
            start_formatted = self.format_duration(start)
            end_formatted = self.format_duration(end)
            display_name = self.speaker_mapping.get(spk, spk)

            # Réinsertion dans la zone de texte
            self.transcription_text.insert(tk.END, f"{start_formatted} - {end_formatted}: {display_name}\n{text}\n\n")

        # Réinitialiser le champ d'entrée
        self.name_entry.delete(0, tk.END)
        self.audio_label.configure(text=f"Speaker {speaker} renamed to {new_name}.")

    def load_audio_file(self):
        """Charge un fichier audio et initialise le lecteur VLC."""
        self.audio_path = filedialog.askopenfilename(
            title="Select an Audio File",
            filetypes=(("Audio Files", "*.mp3 *.wav *.ogg *.flac"), ("All Files", "*.*"))
        )
        if self.audio_path:
            # Sauvegarder une copie de l'original
            shutil.copy(self.audio_path, "original_audio.wav")
            self.audio_label.configure(text=f"File loaded: {os.path.basename(self.audio_path)}")
            self.vlc_player = vlc.MediaPlayer(self.audio_path)
            self.process_button.configure(state="normal")
        else:
            self.audio_label.configure(text="No file selected.")
            self.process_button.configure(state="disabled")

    def stop_audio(self):
        """Stop audio playback using VLC."""
        try:
            if hasattr(self, 'vlc_player') and self.vlc_player.is_playing():
                self.vlc_player.stop()
                self.audio_label.configure(text="Playback stopped.")
            else:
                self.audio_label.configure(text="No audio is currently playing.")
        except Exception as e:
            self.audio_label.configure(text=f"Error during stop: {str(e)}")

    def save_results(self):
        """Save results to a text file."""
        output_file = filedialog.asksaveasfilename(
            title="Save Text File", defaultextension=".txt", filetypes=(("Text Files", "*.txt"), ("All Files", "*.*"))
        )
        if output_file:
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    for start, end, spk, text in self.transcriptions:
                        f.write(f"{start} - {end}: {spk}\n{text}\n\n")
                self.audio_label.configure(text=f"Results saved to {output_file}")
            except Exception as e:
                self.audio_label.configure(text=f"Save error: {str(e)}")

    def start_processing(self):
        """Start processing the audio."""
        self.process_button.configure(state="disabled")
        threading.Thread(target=self.process_audio).start()

    def process_audio(self):
        """Process the audio file."""
        try:
            selected_model = self.model_choice.get()  # Récupère le modèle sélectionné dans l'interface
            self.transcriptions, self.speaker_files = process_audio(
                self.audio_path, self.diarization_enabled.get(), model_name=selected_model
            )

            self.transcription_text.delete("1.0", tk.END)

            if not self.diarization_enabled.get():
                for text in self.transcriptions:
                    self.transcription_text.insert(tk.END, f"{text}\n\n")
            else:
                for start, end, spk, text in self.transcriptions:
                    start_formatted = self.format_duration(start)
                    end_formatted = self.format_duration(end)
                    self.transcription_text.insert(tk.END, f"{start_formatted} - {end_formatted}: {spk}\n{text}\n\n")

            # Réinitialiser le lecteur VLC
            if hasattr(self, 'vlc_player'):
                self.vlc_player.stop()  # Arrêter la lecture si en cours
                self.vlc_player = vlc.MediaPlayer(self.audio_path)  # Recharger le fichier audio

            # Réinitialiser le curseur et l'état
            self.current_position = 0
            self.audio_slider.set(0)
            self.audio_slider_label.configure(text="0m0s")
            self.audio_label.configure(text="Audio processed and ready.")
            self.populate_speaker_list()
            self.export_button.configure(state="normal")
        except Exception as e:
            self.audio_label.configure(text=f"Error: {str(e)}")
        finally:
            self.process_button.configure(state="normal")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    app = DiarizationApp()
    app.mainloop()