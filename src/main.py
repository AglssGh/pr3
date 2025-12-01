import matplotlib.pyplot as plt
import argparse
import base64
import os
from huggingface_hub.errors import HfHubHTTPError
from pathlib import Path
from typing import List, Tuple, Dict
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# --- Настройки проекта ---

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Папки с изображениями и целевые метки
CLASS_DIRS = [
    ("cats", "cat"),
    ("dogs", "dog"),
]

PROMPT_STRICT = (
    "You are an image classifier. Look at the image and answer with ONLY one word: "
    "'cat' or 'dog'. If you are unsure, choose the most probable option. "
    "Answer in English with just that one word."
)

PROMPT_LENIENT = (
    "You are an image classifier. Look at the image and try to decide whether it is a "
    "'cat' or a 'dog'. If you are uncertain or the animal is not clearly visible, "
    "DEFAULT to 'cat'. Answer with EXACTLY one word: 'cat' or 'dog'."
)


MODEL_CONFIGS = [
    {
        "id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "name": "Qwen-7B-strict",
        "prompt": PROMPT_STRICT,
    },
    {
        "id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "name": "Qwen-7B-lenient",
        "prompt": PROMPT_LENIENT,
    },
]



# --- Вспомогательные функции работы с данными ---

def load_dataset(max_per_class: int | None = None) -> List[Tuple[Path, str]]:
    """Собираем (путь_к_фото, истинная_метка) из папок data/cats и data/dogs."""
    dataset: List[Tuple[Path, str]] = []

    for folder_name, label in CLASS_DIRS:
        class_dir = DATA_DIR / folder_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Папка с данными не найдена: {class_dir}")

        count = 0
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".jfif"}:
                continue
            dataset.append((img_path, label))
            count += 1
            if max_per_class is not None and count >= max_per_class:
                break

    if not dataset:
        raise RuntimeError("Датасет пуст. Положите изображения в data/cats и data/dogs.")
    return dataset


def image_to_data_url(path: Path) -> str:
    """Кодируем картинку в base64 data URL, который понимает InferenceClient."""
    with path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    ext = path.suffix.lower()
    if ext == ".png":
        mime = "image/png"
    else:
        mime = "image/jpeg"

    return f"data:{mime};base64,{b64}"


# --- Работа с моделями ---

def build_client(model_id: str, hf_token: str) -> InferenceClient:
    """Создаём клиента для конкретной модели."""
    return InferenceClient(model=model_id, token=hf_token)


def ask_model(client: InferenceClient, image_data_url: str, prompt: str) -> str:
    """Отправляем картинку + промпт и получаем текстовый ответ."""
    output = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        max_tokens=32,
    )

    message_content = output.choices[0].message.content
    if isinstance(message_content, str):
        text = message_content
    elif isinstance(message_content, list):
        text = "".join(
            part.get("text", "")
            for part in message_content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    else:
        text = str(message_content)

    return text.strip()


def normalize_prediction(raw_text: str) -> str:
    """Приводим произвольный текст модели к 'cat' или 'dog'."""
    if raw_text is None:
        return "cat"

    t = str(raw_text).lower()
    t = t.replace(".", " ").replace(",", " ").strip()

    # Если вообще ничего нет – дефолтимся к 'cat'
    if not t:
        return "cat"

    # Простые правила
    if "cat" in t and "dog" not in t:
        return "cat"
    if "dog" in t and "cat" not in t:
        return "dog"

    parts = t.split()
    if not parts:
        return "cat"

    first = parts[0]
    if "dog" in first:
        return "dog"
    return "cat"



# --- Основная логика эксперимента ---

def evaluate_models(
        dataset: List[Tuple[Path, str]],
        hf_token: str,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Прогоняем датасет через каждую модель-конфигурацию.
    Возвращаем:
    {
      model_name: {
        "y_true": [...],
        "y_pred": [...]
      },
      ...
    }
    """
    results: Dict[str, Dict[str, List[str]]] = {}

    encoded_images = {path: image_to_data_url(path) for path, _ in dataset}

    for cfg in MODEL_CONFIGS:
        model_id = cfg["id"]
        model_name = cfg["name"]
        prompt = cfg["prompt"]

        print(f"\n=== Модель: {model_name} ({model_id}) ===")
        client = build_client(model_id, hf_token)

        y_true: List[str] = []
        y_pred: List[str] = []

        try:
            for path, true_label in dataset:
                img_data_url = encoded_images[path]
                raw_answer = ask_model(client, img_data_url, prompt)
                pred_label = normalize_prediction(raw_answer)

                y_true.append(true_label)
                y_pred.append(pred_label)

                print(f"{path.name:30s}  true={true_label:3s}  pred={pred_label:3s}  raw='{raw_answer}'")

        except HfHubHTTPError as e:
            print(f"!!! Модель {model_name} недоступна, пропускаю.")
            print(f"    Детали от API: {e}")
            continue

        results[model_name] = {"y_true": y_true, "y_pred": y_pred}

    return results


def print_metrics(results: Dict[str, Dict[str, List[str]]]) -> None:
    print("\n\n================ ОЦЕНКА МОДЕЛЕЙ ================\n")

    for model_id, data in results.items():
        y_true = data["y_true"]
        y_pred = data["y_pred"]

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        print(f"Модель: {model_id}")
        print(f"  Accuracy : {acc:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall   : {recall:.3f}")
        print(f"  F1-score : {f1:.3f}")
        print("\nПодробный отчёт по классам:")
        print(classification_report(y_true, y_pred, digits=3))
        print("-" * 60)

def print_metrics(results: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, float]]:
    print("\n\n================ ОЦЕНКА МОДЕЛЕЙ ================\n")

    metrics: Dict[str, Dict[str, float]] = {}

    for model_id, data in results.items():
        y_true = data["y_true"]
        y_pred = data["y_pred"]

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        metrics[model_id] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        print(f"Модель: {model_id}")
        print(f"  Accuracy : {acc:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall   : {recall:.3f}")
        print(f"  F1-score : {f1:.3f}")
        print("\nПодробный отчёт по классам:")
        print(classification_report(y_true, y_pred, digits=3))
        print("-" * 60)

    return metrics

def plot_metrics(metrics: Dict[str, Dict[str, float]], output_path: str = "metrics.png") -> None:
    if not metrics:
        print("Нет метрик для построения графика.")
        return

    models = list(metrics.keys())
    acc = [metrics[m]["accuracy"] for m in models]
    prec = [metrics[m]["precision"] for m in models]
    rec = [metrics[m]["recall"] for m in models]
    f1 = [metrics[m]["f1"] for m in models]

    x = list(range(len(models)))
    width = 0.2

    plt.figure()
    plt.bar([i - 1.5 * width for i in x], acc, width, label="accuracy")
    plt.bar([i - 0.5 * width for i in x], prec, width, label="precision")
    plt.bar([i + 0.5 * width for i in x], rec, width, label="recall")
    plt.bar([i + 1.5 * width for i in x], f1, width, label="f1")

    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("score")
    plt.title("Сравнение метрик моделей")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"График метрик сохранён в {output_path}")


def main():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise RuntimeError(
            "Не найден HF_TOKEN. Создайте файл .env в корне проекта и добавьте строку:\n"
            "HF_TOKEN=hf_xxx_ваш_токен"
        )

    parser = argparse.ArgumentParser(description="Cats vs Dogs LLM experiment (Hugging Face).")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=15,
        help="Максимум изображений на класс (по умолчанию 15).",
    )
    args = parser.parse_args()

    dataset = load_dataset(max_per_class=args.max_per_class)
    print(f"Всего изображений: {len(dataset)}")

    results = evaluate_models(dataset, hf_token)
    metrics = print_metrics(results)
    plot_metrics(metrics)



if __name__ == "__main__":
    main()
