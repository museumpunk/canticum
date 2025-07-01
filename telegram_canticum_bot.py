import base64
import aiohttp
import asyncio
import uuid
import time
import logging
import ssl
import os
import sqlite3
import re
import tempfile
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from io import BytesIO
from pydub import AudioSegment
from fpdf import FPDF
try:
    import ffmpeg
except ImportError:
    ffmpeg = None

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Путь к сертификату (замените на ваш путь)
CERT_PATH = "путь/к/russian_trusted_root_ca.cer"

# API-ключ для DeepSeek
DEEPSEEK_API_KEY = "ВАШ_КЛЮЧ"

# Класс для работы с SaluteSpeech API
class SaluteSpeechAuth:
    def __init__(self, client_id, client_secret, scope="SALUTE_SPEECH_PERS"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.access_token = None
        self.expires_at = 0
        self.token_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        self.auth_key = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

        # Проверка наличия сертификата
        if not os.path.exists(CERT_PATH):
            logger.error(f"Сертификат не найден: {CERT_PATH}")
            raise FileNotFoundError(f"Сертификат не найден: {CERT_PATH}")

        # SSL-контекст для HTTP
        self.ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self.ssl_context.verify_mode = ssl.CERT_REQUIRED
        self.ssl_context.load_verify_locations(cafile=CERT_PATH)
        logger.info(f"Сертификат загружен для HTTP: {CERT_PATH}")

    async def get_token(self):
        if self.access_token and self.expires_at > time.time() + 60:
            logger.info("Используется существующий токен")
            return self.access_token

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {self.auth_key}",
        }
        payload = {"scope": self.scope}

        logger.info(f"Запрос токена: {self.token_url}")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.token_url,
                    headers=headers,
                    data=payload,
                    ssl=self.ssl_context,
                    timeout=30
                ) as response:
                    response_text = await response.text()
                    logger.info(f"Ответ токена: {response.status} - {response_text}")
                    if response.status != 200:
                        logger.error(f"Ошибка получения токена: {response_text}")
                        raise Exception(f"Не удалось получить токен: {response.status}")
                    data = await response.json()
                    self.access_token = data["access_token"]
                    self.expires_at = data["expires_at"] / 1000
                    logger.info("Токен успешно получен")
                    return self.access_token
            except Exception as e:
                logger.error(f"Ошибка при запросе токена: {str(e)}")
                raise

    async def recognize_audio(self, audio_data: BytesIO, file_format: str, hints=None, retries=3):
        token = await self.get_token()
        url = "https://smartspeech.sber.ru/rest/v1/speech:recognize"
        audio_data.seek(0)
        content_type = "audio/ogg;codecs=opus" if file_format == "ogg" else "audio/x-pcm;bit=16;rate=8000"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": content_type,
            "Accept": "application/json",
        }
        params = {
            "topic": "general",
            "language": "ru-RU",
            "audio_encoding": "OGG_OPUS" if file_format == "ogg" else "PCM",
            "enable_profanity_filter": "false",
            "enable_diarization": "true",
            "insight_models": ["call_features"],
        }
        if hints:
            params["hints"] = ",".join(hints)

        for attempt in range(retries):
            logger.info(f"Отправка HTTP-запроса (попытка {attempt + 1}): size={audio_data.getbuffer().nbytes} bytes")
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        url,
                        headers=headers,
                        params=params,
                        data=audio_data,
                        ssl=self.ssl_context,
                        timeout=60
                    ) as response:
                        response_text = await response.text()
                        logger.info(f"HTTP ответ: {response.status} - {response_text}")
                        if response.status != 200:
                            logger.error(f"Ошибка распознавания HTTP: {response_text}")
                            return {"error": f"Ошибка распознавания: {response.status}"}
                        data = await response.json()
                        logger.info(f"Структура ответа HTTP: {data}")
                        results = []
                        for i, text in enumerate(data.get("result", ["Текст не распознан"])):
                            if text.strip():
                                result = {
                                    "result": [text],
                                    "emotions": data.get("emotions", [{}])[i % len(data.get("emotions", [{}]))],
                                    "speaker_id": data.get("speaker_id", -1),
                                    "call_features": {
                                        "interruptions": data.get("insight", {}).get("call_features", {}).get("interruptions", 0),
                                        "emo_score": data.get("insight", {}).get("call_features", {}).get("emo_score", 0),
                                    },
                                    "person_identity": data.get("person_identity", {"age": "age_none", "gender": "gender_none", "age_score": 0, "gender_score": 0})
                                }
                                if result["speaker_id"] == -1:
                                    logger.warning(f"Диаризация не сработала, speaker_id: -1")
                                if not data.get("insight", {}).get("call_features", {}) or result["call_features"]["interruptions"] == 0:
                                    logger.warning("call_features пустые или не содержат данных (возможно, монолог или Insights не активированы)")
                                if result["person_identity"].get("age") == "age_none" and result["person_identity"].get("gender") == "gender_none":
                                    logger.warning("person_identity не определён (age_none, gender_none)")
                                results.append(result)
                        return results
                except aiohttp.ClientError as e:
                    logger.error(f"Ошибка HTTP на попытке {attempt + 1}: {str(e)}")
                    if attempt < retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        audio_data.seek(0)
                        continue
                    return {"error": f"Ошибка при распознавании после {retries} попыток: {str(e)}"}
                except Exception as e:
                    logger.error(f"Общая ошибка при распознавании HTTP на попытке {attempt + 1}: {str(e)}")
                    return {"error": f"Ошибка при распознавании: {str(e)}"}

# Класс для работы с базой данных
class Database:
    def __init__(self, db_path="recognitions.db"):
        self.db_path = db_path
        try:
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
            self.create_table()
        except Exception as e:
            logger.error(f"Ошибка инициализации БД: {str(e)}")
            raise

    def create_table(self):
        try:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS recognitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    date TEXT,
                    recognized_text TEXT,
                    summary_text TEXT
                )
            ''')
            self.conn.commit()
            logger.info("Таблица recognitions создана или уже существует")
        except Exception as e:
            logger.error(f"Ошибка создания таблицы: {str(e)}")
            raise

    def insert_recognition(self, username, date, recognized_text, summary_text):
        try:
            self.cursor.execute('''
                INSERT INTO recognitions (username, date, recognized_text, summary_text)
                VALUES (?, ?, ?, ?)
            ''', (username, date, recognized_text, summary_text))
            self.conn.commit()
            logger.info(f"Запись добавлена в БД для пользователя {username} на дату {date}")
        except Exception as e:
            logger.error(f"Ошибка при записи в БД: {str(e)}")
            raise

    def fetch_all_records(self):
        try:
            self.cursor.execute('''
                SELECT date, summary_text, recognized_text FROM recognitions ORDER BY date DESC
            ''')
            records = self.cursor.fetchall()
            logger.info(f"Извлечено {len(records)} записей из БД")
            return records
        except Exception as e:
            logger.error(f"Ошибка при извлечении записей из БД: {str(e)}")
            return []

    def close(self):
        try:
            self.conn.close()
            logger.info("Соединение с БД закрыто")
        except Exception as e:
            logger.error(f"Ошибка при закрытии БД: {str(e)}")

# Функция для генерации PDF
def generate_pdf(text, filename="recognized_text.pdf"):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=12)
        pdf.multi_cell(0, 5, text, align="L")
        pdf.output(filename)
        logger.info(f"PDF сгенерирован: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Ошибка при генерации PDF: {str(e)}")
        return None

# Локальная саммаризация (запасной вариант)
def local_summarize_text(text, max_sentences=5):
    try:
        sentences = text.split(". ")
        key_sentences = sentences[:min(max_sentences, len(sentences))]
        summary = ". ".join(s.strip() for s in key_sentences if s.strip())
        if summary:
            summary += "."
        logger.info("Локальное саммари сгенерировано")
        return f"<blockquote>{summary}</blockquote>"
    except Exception as e:
        logger.error(f"Ошибка локальной саммаризации: {str(e)}")
        return f"<blockquote>Ошибка локальной саммаризации: {str(e)}</blockquote>"

# Асинхронная функция для саммаризации через DeepSeek
async def summarize_text(text):
    try:
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "Ты секретарь, который кратко резюмирует результаты совещаний на русском языке. Создай краткое саммари текста (до 150 слов), выделяя ключевые моменты, без заголовков и форматирования Markdown."},
                {"role": "user", "content": text}
            ],
            "max_tokens": 500
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                response_text = await response.text()
                logger.info(f"DeepSeek ответ: {response.status} - {response_text}")
                if response.status != 200:
                    logger.error(f"Ошибка DeepSeek: {response_text}")
                    return local_summarize_text(text)
                data = await response.json()
                summary = data.get("choices", [{}])[0].get("message", {}).get("content", "Не удалось сгенерировать саммари")
                summary = re.sub(r'[\*\*_]', '', summary)
                summary = f"<blockquote>{summary}</blockquote>"
                logger.info("Саммари успешно сгенерировано")
                return summary
    except Exception as e:
        logger.error(f"Ошибка при саммаризации через DeepSeek: {str(e)}")
        return local_summarize_text(text)

# Асинхронная функция для поиска по базе данных через DeepSeek
async def search_db(query, db):
    try:
        records = db.fetch_all_records()
        if not records:
            return "<blockquote>База данных пуста или произошла ошибка при извлечении записей.</blockquote>"

        context = "История совещаний:\n"
        for date, summary, text in records:
            context += f"Дата: {date}\nСаммари: {summary}\nПолный текст: {text}\n\n"

        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "Ты секретарь, который отвечает на вопросы о прошлых совещаниях на русском языке, основываясь на предоставленной базе данных. Отвечай кратко и точно, без Markdown-форматирования."},
                {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}"}
            ],
            "max_tokens": 500
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                response_text = await response.text()
                logger.info(f"DeepSeek поиск ответ: {response.status} - {response_text}")
                if response.status != 200:
                    logger.error(f"Ошибка DeepSeek поиска: {response_text}")
                    return "<blockquote>Ошибка поиска: не удалось обработать запрос.</blockquote>"
                data = await response.json()
                answer = data.get("choices", [{}])[0].get("message", {}).get("content", "Не удалось обработать запрос")
                answer = re.sub(r'[\*\*_]', '', answer)
                return f"<blockquote>{answer}</blockquote>"
    except Exception as e:
        logger.error(f"Ошибка при поиске через DeepSeek: {str(e)}")
        return f"<blockquote>Ошибка поиска: {str(e)}</blockquote>"

# Класс Telegram-бота
class TelegramBot:
    def __init__(self, telegram_token, salute_auth, db):
        self.app = Application.builder().token(telegram_token).build()
        self.salute_auth = salute_auth
        self.db = db
        self.hints = []

    async def set_hints(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if context.args:
                self.hints = context.args
                await update.message.reply_text(f"Хинты установлены: {', '.join(self.hints)}")
                logger.info(f"Хинты установлены: {self.hints}")
            else:
                self.hints = []
                await update.message.reply_text("Хинты сброшены.")
                logger.info("Хинты сброшены")
        except Exception as e:
            logger.error(f"Ошибка при установке хинтов: {str(e)}")
            await update.message.reply_text(f"Ошибка при установке хинтов: {str(e)}")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            await update.message.reply_text(
                "Отправьте мне запись совещания (до 1 часа), и я распознаю текст, сгенерирую краткое саммари и отправлю PDF с полным текстом. "
                "Или отправьте вопрос по прошлым совещаниям."
            )
            logger.info("Команда /start выполнена")
        except Exception as e:
            logger.error(f"Ошибка команды /start: {str(e)}")
            await update.message.reply_text(f"Ошибка: {str(e)}")

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            query = update.message.text
            if not query:
                await update.message.reply_text("Пожалуйста, укажите вопрос по прошлым совещаниям.")
                logger.warning("Пустой текстовый запрос")
                return

            answer = await search_db(query, self.db)
            TELEGRAM_MAX_MESSAGE_LENGTH = 4096
            response = f"Результат поиска:\n{answer}"
            if len(response) <= TELEGRAM_MAX_MESSAGE_LENGTH:
                await update.message.reply_text(response, parse_mode="HTML")
            else:
                parts = []
                current_part = ""
                for line in response.split("\n"):
                    if len(current_part) + len(line) + 1 <= TELEGRAM_MAX_MESSAGE_LENGTH:
                        current_part += line + "\n"
                    else:
                        parts.append(current_part)
                        current_part = line + "\n"
                if current_part:
                    parts.append(current_part)
                for part in parts:
                    await update.message.reply_text(part, parse_mode="HTML")
                    await asyncio.sleep(0.5)
            logger.info(f"Поиск выполнен для запроса: {query}")
        except Exception as e:
            logger.error(f"Ошибка обработки текстового запроса: {str(e)}")
            error_msg = f"Ошибка поиска: {str(e)}"
            if len(error_msg) > 4096:
                error_msg = error_msg[:4093] + "..."
            await update.message.reply_text(error_msg)

    async def handle_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        recognized_texts = []
        all_emotions = []
        temp_file_mp3 = None
        temp_file_m4a = None
        try:
            is_voice = False
            file_format = "ogg"
            if update.message.voice:
                file = await context.bot.get_file(update.message.voice.file_id)
                is_voice = True
                logger.info("Получено голосовое сообщение")
            elif update.message.audio:
                file = await context.bot.get_file(update.message.audio.file_id)
                file_format = update.message.audio.mime_type.split("/")[-1]
                logger.info(f"Получен аудиофайл, формат: {file_format}, file_id: {file.file_id}, file_path: {file.file_path}")
            else:
                await update.message.reply_text("Пожалуйста, отправь голосовое сообщение или аудиофайл.")
                logger.warning("Получен неподдерживаемый тип сообщения")
                return

            progress_message = await update.message.reply_text(
                "Ваше аудио обрабатывается. Прогресс: [          ] 0%"
            )

            audio_data = None
            retries = 3
            for attempt in range(retries):
                async with aiohttp.ClientSession() as session:
                    try:
                        logger.info(f"Попытка загрузки файла {file.file_path} (попытка {attempt + 1})")
                        async with session.get(file.file_path, timeout=180) as response:
                            if response.status != 200:
                                error_msg = f"Ошибка загрузки файла: HTTP {response.status}. "
                                if response.status == 413:
                                    error_msg += "Файл слишком большой (>50 МБ). Разбейте аудио на части до 3 минут."
                                elif response.status == 400:
                                    error_msg += "Попробуйте перекодировать файл в M4A или WAV с помощью ffmpeg или онлайн-конвертера (например, Zamzar или Convertio)."
                                logger.error(f"Ошибка загрузки файла: HTTP {response.status}")
                                if attempt == retries - 1:
                                    await update.message.reply_text(error_msg)
                                    return
                                continue
                            audio_data = BytesIO(await response.read())
                            logger.info(f"Аудиофайл загружен: {audio_data.getbuffer().nbytes} bytes")
                            if audio_data.getbuffer().nbytes > 50 * 1024 * 1024:
                                await update.message.reply_text(
                                    "Файл слишком большой (>50 МБ). Разбейте аудио на части до 3 минут или перекодируйте в M4A или WAV с помощью ffmpeg или онлайн-конвертера (например, Zamzar или Convertio)."
                                )
                                logger.error("Файл превышает 50 МБ")
                                return
                            break
                    except aiohttp.ClientError as e:
                        logger.error(f"Ошибка загрузки аудиофайла на попытке {attempt + 1}: {str(e)}")
                        if attempt == retries - 1:
                            error_msg = f"Ошибка загрузки аудиофайла после {retries} попыток: {str(e)}. Попробуйте перекодировать файл в M4A или WAV с помощью ffmpeg или онлайн-конвертера (например, Zamzar или Convertio)."
                            if len(error_msg) > 4096:
                                error_msg = error_msg[:4093] + "..."
                            await update.message.reply_text(error_msg)
                            return
                        await asyncio.sleep(2 ** attempt)
            if not audio_data:
                logger.error("Не удалось загрузить аудиофайл после всех попыток")
                return

            if file_format == "mpeg":
                if not ffmpeg:
                    error_msg = "Модуль ffmpeg-python не установлен. Установите его (pip install ffmpeg-python) или перекодируйте файл в M4A (ffmpeg -i input.mp3 -c:a aac -b:a 128k output.m4a) или WAV (ffmpeg -i input.mp3 -c:a pcm_s16le -ar 8000 -ac 1 output.wav)."
                    await update.message.reply_text(error_msg)
                    logger.error("Модуль ffmpeg-python не установлен")
                    return
                try:
                    logger.info("Конвертация MP3 в M4A с помощью ffmpeg")
                    temp_file_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    audio_data.seek(0)
                    temp_file_mp3.write(audio_data.read())
                    temp_file_mp3.close()
                    temp_file_m4a = tempfile.NamedTemporaryFile(delete=False, suffix=".m4a")
                    temp_file_m4a.close()
                    stream = ffmpeg.input(temp_file_mp3.name)
                    stream = ffmpeg.output(stream, temp_file_m4a.name, format="mp4", acodec="aac", ab="128k")
                    stream = ffmpeg.overwrite_output(stream)
                    ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
                    logger.info(f"Конвертация MP3 в M4A завершена: {temp_file_m4a.name}")
                    try:
                        os.remove(temp_file_mp3.name)
                        logger.info(f"Временный файл удалён: {temp_file_mp3.name}")
                    except Exception as e:
                        logger.warning(f"Ошибка удаления временного файла {temp_file_mp3.name}: {str(e)}")

                    logger.info("Конвертация M4A в WAV с помощью ffmpeg")
                    temp_file_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    temp_file_wav.close()
                    stream = ffmpeg.input(temp_file_m4a.name)
                    stream = ffmpeg.output(stream, temp_file_wav.name, format="wav", ac=1, ar=8000, acodec="pcm_s16le")
                    stream = ffmpeg.overwrite_output(stream)
                    ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
                    with open(temp_file_wav.name, "rb") as f:
                        audio_data = BytesIO(f.read())
                    file_format = "wav"
                    logger.info(f"Конвертация M4A в WAV завершена: {temp_file_wav.name}")
                    try:
                        os.remove(temp_file_m4a.name)
                        logger.info(f"Временный файл удалён: {temp_file_m4a.name}")
                    except Exception as e:
                        logger.warning(f"Ошибка удаления временного файла {temp_file_m4a.name}: {str(e)}")
                    try:
                        os.remove(temp_file_wav.name)
                        logger.info(f"Временный файл удалён: {temp_file_wav.name}")
                    except Exception as e:
                        logger.warning(f"Ошибка удаления временного файла {temp_file_wav.name}: {str(e)}")
                except Exception as e:
                    error_msg = f"Ошибка конвертации MP3 в M4A или WAV: {str(e)}. Попробуйте перекодировать файл в M4A (ffmpeg -i input.mp3 -c:a aac -b:a 128k output.m4a) или WAV (ffmpeg -i input.mp3 -c:a pcm_s16le -ar 8000 -ac 1 output.wav) с помощью ffmpeg или онлайн-конвертера (например, Zamzar или Convertio)."
                    if len(error_msg) > 4096:
                        error_msg = error_msg[:4093] + "..."
                    await update.message.reply_text(error_msg)
                    logger.error(f"Ошибка конвертации MP3/M4A: {str(e)}")
                    for temp_file in [temp_file_mp3, temp_file_m4a, temp_file_wav]:
                        if temp_file and os.path.exists(temp_file.name):
                            try:
                                os.remove(temp_file.name)
                                logger.info(f"Временный файл удалён: {temp_file.name}")
                            except Exception as e:
                                logger.warning(f"Ошибка удаления временного файла {temp_file.name}: {str(e)}")
                    return
            elif file_format == "m4a":
                if not ffmpeg:
                    error_msg = "Модуль ffmpeg-python не установлен. Установите его (pip install ffmpeg-python) или перекодируйте файл в WAV (ffmpeg -i input.m4a -c:a pcm_s16le -ar 8000 -ac 1 output.wav)."
                    await update.message.reply_text(error_msg)
                    logger.error("Модуль ffmpeg-python не установлен")
                    return
                try:
                    logger.info("Конвертация M4A в WAV с помощью ffmpeg")
                    temp_file_m4a = tempfile.NamedTemporaryFile(delete=False, suffix=".m4a")
                    audio_data.seek(0)
                    temp_file_m4a.write(audio_data.read())
                    temp_file_m4a.close()
                    temp_file_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    temp_file_wav.close()
                    stream = ffmpeg.input(temp_file_m4a.name)
                    stream = ffmpeg.output(stream, temp_file_wav.name, format="wav", ac=1, ar=8000, acodec="pcm_s16le")
                    stream = ffmpeg.overwrite_output(stream)
                    ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
                    with open(temp_file_wav.name, "rb") as f:
                        audio_data = BytesIO(f.read())
                    file_format = "wav"
                    logger.info(f"Конвертация M4A в WAV завершена: {temp_file_wav.name}")
                    try:
                        os.remove(temp_file_m4a.name)
                        logger.info(f"Временный файл удалён: {temp_file_m4a.name}")
                    except Exception as e:
                        logger.warning(f"Ошибка удаления временного файла {temp_file_m4a.name}: {str(e)}")
                    try:
                        os.remove(temp_file_wav.name)
                        logger.info(f"Временный файл удалён: {temp_file_wav.name}")
                    except Exception as e:
                        logger.warning(f"Ошибка удаления временного файла {temp_file_wav.name}: {str(e)}")
                except Exception as e:
                    error_msg = f"Ошибка конвертации M4A в WAV: {str(e)}. Попробуйте перекодировать файл в WAV (ffmpeg -i input.m4a -c:a pcm_s16le -ar 8000 -ac 1 output.wav) или онлайн-конвертера (например, Zamzar или Convertio)."
                    if len(error_msg) > 4096:
                        error_msg = error_msg[:4093] + "..."
                    await update.message.reply_text(error_msg)
                    logger.error(f"Ошибка конвертации M4A: {str(e)}")
                    if temp_file_m4a and os.path.exists(temp_file_m4a.name):
                        try:
                            os.remove(temp_file_m4a.name)
                            logger.info(f"Временный файл удалён: {temp_file_m4a.name}")
                        except Exception as e:
                            logger.warning(f"Ошибка удаления временного файла {temp_file_m4a.name}: {str(e)}")
                    return

            try:
                audio = AudioSegment.from_file(audio_data, format=file_format)
                audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
                duration_seconds = len(audio) / 1000
                logger.info(f"Длительность аудио (pydub): {duration_seconds} секунд")
                if duration_seconds > 3600:
                    await update.message.reply_text("Ошибка: аудио превышает 1 час.")
                    logger.error("Аудио превышает 1 час")
                    return
            except Exception as e:
                error_msg = f"Ошибка проверки длительности аудио: {str(e)}. Попробуйте перекодировать файл в WAV (ffmpeg -i input.mp3 -c:a pcm_s16le -ar 8000 -ac 1 output.wav) или онлайн-конвертера (например, Zamzar или Convertio)."
                if len(error_msg) > 4096:
                    error_msg = error_msg[:4093] + "..."
                await update.message.reply_text(error_msg)
                logger.error(f"Ошибка обработки аудио в pydub: {str(e)}")
                return

            segment_duration = 55 * 1000
            segments = [audio[i:i + segment_duration] for i in range(0, len(audio), segment_duration)]
            logger.info(f"Разбито на {len(segments)} фрагментов")

            total_segments = len(segments)
            for i, segment in enumerate(segments):
                seg_duration = len(segment) / 1000
                logger.info(f"Обработка фрагмента {i + 1}, длительность: {seg_duration} секунд")
                if seg_duration > 55:
                    logger.warning(f"Фрагмент {i + 1} обрезан до 55 секунд")
                    segment = segment[:55 * 1000]

                segment_buffer = BytesIO()
                try:
                    segment = segment.set_frame_rate(8000).set_channels(1).set_sample_width(2)
                    segment.export(segment_buffer, format="wav", parameters=["-f", "s16le"])
                    logger.info(f"Фрагмент {i + 1} экспортирован в PCM 8 кГц")
                except Exception as e:
                    error_msg = f"Ошибка экспорта фрагмента {i + 1}: {str(e)}"
                    if len(error_msg) > 4096:
                        error_msg = error_msg[:4093] + "..."
                    await update.message.reply_text(error_msg)
                    logger.error(f"Ошибка экспорта фрагмента {i + 1}: {str(e)}")
                    return
                segment_buffer.seek(0)

                results = await self.salute_auth.recognize_audio(segment_buffer, "wav", hints=self.hints)
                if isinstance(results, dict) and "error" in results:
                    logger.error(f"Ошибка в фрагменте {i + 1}: {results['error']}")
                    error_msg = f"Ошибка в фрагменте {i + 1}: {results['error']}"
                    if len(error_msg) > 4096:
                        error_msg = error_msg[:4093] + "..."
                    await update.message.reply_text(error_msg)
                    if recognized_texts:
                        username = update.message.from_user.username or "Unknown"
                        date = time.strftime("%Y-%m-%d %H:%M:%S")
                        partial_text = "\n".join(f"Спикер {sid}: {text}" for sid, text in recognized_texts)
                        self.db.insert_recognition(username, date, partial_text, "Обработка прервана")
                    return

                for result in results:
                    if result["result"]:
                        text = " ".join(result["result"]) if isinstance(result["result"], list) else result["result"]
                        speaker_id = result["speaker_id"] if result["speaker_id"] != -1 else 1
                        recognized_texts.append((speaker_id, text))
                        logger.info(f"Фрагмент {i + 1} распознан: {text}")
                    if result["emotions"]:
                        all_emotions.append(result["emotions"])
                        logger.info(f"Эмоции фрагмента {i + 1}: {result['emotions']}")

                progress = int((i + 1) / total_segments * 100)
                filled = "█" * (progress // 10)
                empty = " " * (10 - progress // 10)
                await progress_message.edit_text(
                    f"Ваше аудио обрабатывается. Прогресс: [{filled}{empty}] {progress}%"
                )

            speaker_texts = {}
            for speaker_id, text in recognized_texts:
                if speaker_id not in speaker_texts:
                    speaker_texts[speaker_id] = []
                speaker_texts[speaker_id].append(text)
            
            full_text = ""
            for speaker_id, texts in speaker_texts.items():
                speaker_label = f"Спикер {speaker_id}"
                full_text += f"{speaker_label}:\n{' '.join(texts)}\n\n"

            summary = await summarize_text(full_text)
            if summary.startswith(("<blockquote>Не удалось сгенерировать саммари", "<blockquote>Ошибка")):
                error_msg = f"Ошибка генерации саммари: {summary.replace('<blockquote>', '').replace('</blockquote>', '')}"
                if len(error_msg) > 4096:
                    error_msg = error_msg[:4093] + "..."
                await update.message.reply_text(error_msg)

            pdf_filename = generate_pdf(full_text)
            if not pdf_filename:
                logger.error("PDF не сгенерирован")
                await update.message.reply_text("Ошибка при генерации PDF.")
                username = update.message.from_user.username or "Unknown"
                date = time.strftime("%Y-%m-%d %H:%M:%S")
                self.db.insert_recognition(username, date, full_text, summary.replace('<blockquote>', '').replace('</blockquote>', ''))
                return

            username = update.message.from_user.username or "Unknown"
            date = time.strftime("%Y-%m-%d %H:%M:%S")
            try:
                self.db.insert_recognition(username, date, full_text, summary.replace('<blockquote>', '').replace('</blockquote>', ''))
            except Exception as e:
                logger.error(f"Ошибка при сохранении в БД: {str(e)}")
                error_msg = f"Ошибка сохранения в БД: {str(e)}"
                if len(error_msg) > 4096:
                    error_msg = error_msg[:4093] + "..."
                await update.message.reply_text(error_msg)

            avg_emotions = {
                "positive": sum(e.get("positive", 0) for e in all_emotions) / len(all_emotions) if all_emotions else 0,
                "neutral": sum(e.get("neutral", 0) for e in all_emotions) / len(all_emotions) if all_emotions else 0,
                "negative": sum(e.get("negative", 0) for e in all_emotions) / len(all_emotions) if all_emotions else 0,
            }

            response = (
                f"Краткое саммари совещания:\n{summary}\n\n"
                f"Средние эмоции:\n"
                f"  Позитивная: {avg_emotions['positive']:.2%}\n"
                f"  Нейтральная: {avg_emotions['neutral']:.2%}\n"
                f"  Негативная: {avg_emotions['negative']:.2%}\n"
            )

            TELEGRAM_MAX_MESSAGE_LENGTH = 4096
            if len(response) <= TELEGRAM_MAX_MESSAGE_LENGTH:
                await update.message.reply_text(response, parse_mode="HTML")
            else:
                parts = []
                current_part = ""
                for line in response.split("\n"):
                    if len(current_part) + len(line) + 1 <= TELEGRAM_MAX_MESSAGE_LENGTH:
                        current_part += line + "\n"
                    else:
                        parts.append(current_part)
                        current_part = line + "\n"
                if current_part:
                    parts.append(current_part)
                for part in parts:
                    await update.message.reply_text(part, parse_mode="HTML")
                    await asyncio.sleep(0.5)

            with open(pdf_filename, "rb") as pdf_file:
                await update.message.reply_document(pdf_file, caption="Полный распознанный текст")

            try:
                os.remove(pdf_filename)
                logger.info(f"Временный PDF удалён: {pdf_filename}")
            except Exception as e:
                logger.error(f"Ошибка удаления PDF: {str(e)}")

        except Exception as e:
            logger.error(f"Общая ошибка в handle_audio: {str(e)}")
            error_msg = f"Произошла ошибка: {str(e)}"
            if len(error_msg) > 4096:
                error_msg = error_msg[:4093] + "..."
            await update.message.reply_text(error_msg)
            if recognized_texts:
                username = update.message.from_user.username or "Unknown"
                date = time.strftime("%Y-%m-%d %H:%M:%S")
                partial_text = "\n".join(f"Спикер {sid}: {text}" for sid, text in recognized_texts)
                self.db.insert_recognition(username, date, partial_text, "Обработка прервана")
            for temp_file in [temp_file_mp3, temp_file_m4a, temp_file_wav]:
                if temp_file and os.path.exists(temp_file.name):
                    try:
                        os.remove(temp_file.name)
                        logger.info(f"Временный файл удалён: {temp_file.name}")
                    except Exception as e:
                        logger.warning(f"Ошибка удаления временного файла {temp_file.name}: {str(e)}")

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.error(f"Update {update} caused error {context.error}")
        if update and update.message:
            error_msg = "Произошла ошибка. Попробуйте еще раз."
            if len(error_msg) > 4096:
                error_msg = error_msg[:4093] + "..."
            await update.message.reply_text(error_msg)

    def run(self):
        try:
            self.app.add_handler(CommandHandler("start", self.start))
            self.app.add_handler(CommandHandler("sethints", self.set_hints))
            self.app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self.handle_audio))
            self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
            self.app.add_error_handler(self.error_handler)
            self.app.run_polling()
        except Exception as e:
            logger.error(f"Ошибка при запуске polling: {str(e)}")
            raise

# Запуск бота
if __name__ == "__main__":
    logger.info("Запуск бота...")
    try:
        db = Database()
        salute_auth = SaluteSpeechAuth(
            client_id="ВАШ_CLIENT_ID",
            client_secret="ВАШ_CLIENT_SECRET",
        )
        logger.info("Инициализирован SaluteSpeech API")
        bot = TelegramBot(
            telegram_token="ВАШ_TELEGRAM_TOKEN",
            salute_auth=salute_auth,
            db=db,
        )
        logger.info("Бот инициализирован, запуск polling...")
        bot.run()
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {str(e)}")
        raise
