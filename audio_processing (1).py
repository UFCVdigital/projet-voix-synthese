import os
from pathlib import Path
from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisperx
import torch
import io
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def transcribe_with_whisperx(audio_path, model_name="small", batch_size=16, compute_type="float16"):
    """Transcription audio avec WhisperX."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initialisation de WhisperX sur l'appareil : {device} avec le modèle : {model_name}")

    # Chargement du modèle WhisperX
    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    print(f"Modèle WhisperX '{model_name}' chargé.")

    # Chargement de l'audio
    audio = whisperx.load_audio(audio_path)
    print(f"Fichier audio chargé depuis : {audio_path}")

    # Transcription avec WhisperX
    result = model.transcribe(audio, batch_size=batch_size, language="fr")
    print("Transcription réalisée avec succès.")

    # Collecte des segments transcrits
    transcriptions = []
    for segment in result["segments"]:
        transcriptions.append(segment['text'])
        print(f"Segment transcrit : {segment['text']}")

    return transcriptions

def clean_brackets(transcriptions):
    """
    Supprime les crochets au début et à la fin des transcriptions.
    """
    return [text.strip('[" ]') for text in transcriptions]


def transcribe_with_medical_whisper(audio_path, model_path="/path/to/Medical-Whisper"):
    """Transcription avec Medical-Whisper."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Chargement de Medical-Whisper depuis : {model_path} sur l'appareil : {device}")

    # Charger le modèle et le processeur
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    print("Medical-Whisper chargé avec succès.")

    # Charger et prétraiter l'audio
    audio_input = processor.load_audio(audio_path, sampling_rate=16000)
    inputs = processor(audio_input, return_tensors="pt").to(device)

    # Générer la transcription
    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print(f"Transcription avec Medical-Whisper : {transcription}")
    return transcription

def load_pipeline_from_pretrained(path_to_config: str | Path) -> Pipeline:
    """Charge le pipeline Pyannote en local."""
    path_to_config = Path(path_to_config)

    print(f"Chargement du pipeline Pyannote depuis : {path_to_config}")
    cwd = Path.cwd().resolve()  # Répertoire de travail actuel
    cd_to = path_to_config.parent  # Répertoire contenant le fichier de configuration

    # Changer de répertoire pour garantir les chemins relatifs
    print(f"Changement du répertoire de travail vers : {cd_to}")
    os.chdir(cd_to)

    # Charger le pipeline depuis le fichier local
    pipeline = Pipeline.from_pretrained(path_to_config)

    # Revenir au répertoire de travail initial
    print(f"Retour au répertoire initial : {cwd}")
    os.chdir(cwd)

    return pipeline

def filter_short_lines(transcriptions, min_length=2):
    """Filtre les transcriptions pour ne garder que les lignes avec au moins `min_length` caractères."""
    return [line for line in transcriptions if len(line.strip()) >= min_length]
def merge_consecutive_speakers(diarization):
    """Fusionne les segments consécutifs pour un même locuteur."""
    merged_segments = []
    previous_speaker = None
    current_start = None
    current_end = None

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker != previous_speaker:
            if previous_speaker is not None:
                merged_segments.append((current_start, current_end, previous_speaker))
            current_start = turn.start
            current_end = turn.end
            previous_speaker = speaker
        else:
            current_end = turn.end

    if previous_speaker is not None:
        merged_segments.append((current_start, current_end, previous_speaker))

    return merged_segments


def process_audio(audio_path, diarization_enabled, token=None, model_name="small"):
    """Traite un fichier audio avec ou sans diarisation.

    Args:
        audio_path (str): Chemin vers le fichier audio.
        diarization_enabled (bool): Si la diarisation est activée.
        token (str, optional): Jeton pour authentification si nécessaire.
        model_name (str, optional): Nom du modèle Whisper à utiliser.

    Returns:
        list: Liste des transcriptions ou segments.
        dict: Dictionnaire des fichiers audio par locuteur (si diarisation activée).
    """
    try:
        if not diarization_enabled:
            # Transcription sans diarisation
            print("Diarisation désactivée. Utilisation de WhisperX.")
            transcription = transcribe_with_whisperx(audio_path, model_name=model_name)
            return transcription, {}

        else:
            # Configuration pour le pipeline de diarisation
            CONFIG_PATH = r"D:/fasterwhisper/Fasterwhisper/models/pyannote_diarization_config.yaml"  # Chemin vers votre config
            pipeline = load_pipeline_from_pretrained(CONFIG_PATH)

            # Déplacer vers GPU si disponible
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
            else:
                print("CUDA non disponible. Utilisation du CPU.")

            # Charger l'audio dans un buffer
            with open(audio_path, 'rb') as f:
                audio_data = io.BytesIO(f.read())

            # Effectuer la diarisation
            diarization = pipeline(audio_data)
            merged_diarization = merge_consecutive_speakers(diarization)

            # Charger l'audio entier avec pydub
            audio = AudioSegment.from_file(audio_path)
            speaker_files = {}
            transcriptions = []

            # Créer un répertoire temporaire
            temp_dir = "temp"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            # Traiter chaque segment identifié
            for start, end, speaker in merged_diarization:
                start_ms = int(start * 1000)  # Conversion en millisecondes
                end_ms = int(end * 1000)

                # Extraire le segment audio
                segment_audio = audio[start_ms:end_ms]
                temp_file = os.path.join(temp_dir, f"temp_{speaker}_{start_ms}_{end_ms}.wav")
                segment_audio.export(temp_file, format="wav")

                # Ajouter le fichier au dictionnaire des locuteurs
                if speaker not in speaker_files:
                    speaker_files[speaker] = []
                speaker_files[speaker].append(temp_file)

                # Transcrire chaque segment
                result = transcribe_with_whisperx(temp_file, model_name=model_name)
                transcriptions.append((start, end, speaker, result))

            print("Traitement terminé.")
            return transcriptions, speaker_files

    except Exception as e:
        print(f"Erreur pendant le traitement de l'audio : {str(e)}")
        return None, None



def clean_temp_files():
    """Supprime les fichiers audio temporaires."""
    temp_dir = "temp"
    if os.path.exists(temp_dir):
        for file_name in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Erreur lors de la suppression de {file_path} : {e}")
