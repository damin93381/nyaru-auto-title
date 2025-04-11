import os
import time
import shutil
import whisper
import logging
import yaml
from pathlib import Path
from tqdm import tqdm

# read config.yaml
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

WATCH_DIR = Path(config["watch_dir"])
WORK_DIR = Path(config["work_dir"])
KEYWORD = config["keyword"]
CHECK_INTERVAL = config.get("check_interval", 60)
WHISPER_MODEL = config.get("whisper_model", "base")
WHISPER_LANGUAGE = config.get("whisper_language", "ja")
DELETE_PROCESSED_VIDEO = config.get("delete_processed_video", True)

# log setup
LOG_PATH = WORK_DIR / "whisper_subtitle.log"
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# create work_dir
WORK_DIR.mkdir(parents=True, exist_ok=True)

# load whisper model
try:
    logging.info(f"加载 Whisper 模型: {WHISPER_MODEL}")
    model = whisper.load_model(WHISPER_MODEL)
    logging.info(f"Whisper 模型 {WHISPER_MODEL} 加载完成")
except Exception as e:
    logging.error(f"加载 Whisper 模型失败: {e}")
    raise SystemExit(1)

# record translated files
processed = set()

def is_target_file(filename):
    return filename.endswith(".flv") and KEYWORD in filename

def copy_to_work_dir(file_path):
    dest_path = WORK_DIR / file_path.name
    shutil.copy2(str(file_path), dest_path)
    logging.info(f"已复制文件到处理目录: {file_path.name}")
    return dest_path

def generate_subtitles(file_path):
    try:
        logging.info(f"开始生成字幕: {file_path.name}（语言：{WHISPER_LANGUAGE}）")
        result = model.transcribe(str(file_path), language=WHISPER_LANGUAGE)
        srt_path = file_path.with_suffix(".srt")

        with open(srt_path, "w", encoding="utf-8") as f:
            for segment in tqdm(result["segments"], desc=f"📝 生成字幕: {file_path.name}", unit="段"):
                f.write(f"{segment['id'] + 1}\n")
                f.write(format_time(segment['start']) + " --> " + format_time(segment['end']) + "\n")
                f.write(segment['text'].strip() + "\n\n")

        shutil.move(str(srt_path), str(WATCH_DIR / srt_path.name))
        logging.info(f"字幕生成成功: {srt_path.name}")

        if DELETE_PROCESSED_VIDEO:
            file_path.unlink()
            logging.info(f"已删除处理后的视频: {file_path.name}")
        else:
            logging.info(f"保留处理后的视频文件: {file_path.name}")

    except Exception as e:
        logging.error(f"字幕生成失败: {file_path.name}，错误信息: {e}")

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")

# main cycle
logging.info("开始监听目录...")

while True:
    for file in WATCH_DIR.iterdir():
        if file.is_file() and is_target_file(file.name) and file not in processed:
            try:
                target_file = copy_to_work_dir(file)
                generate_subtitles(target_file)
                processed.add(file)
            except Exception as e:
                logging.error(f"处理文件 {file.name} 时出错: {e}")
    time.sleep(CHECK_INTERVAL)
