# Запуск комплекта

Самостоятельный инференс `prostate-classifier`: замороженный Phikon-v2
(DINOv2 ViT-L/16, CLS 1024) и обученная MLP 1024→256→2, epoch 4,
`run_20260928_135445/best.pt`. Выбор следует действующей записи
`configs/model_registry.yaml` и текущему маршруту простаты исходного проекта.
Все необходимые веса уже включены. Qwen, API, базы данных, исходные датасеты,
QuPath, WSI-reader и необязательная ROI-MIL для этого запуска не нужны.

## Требования

- Windows x64, Python **3.11.9 x64** (проверенная версия).
- NVIDIA GPU с аппаратной поддержкой BF16 и драйвером для CUDA 13.0.
  Проверено: Windows, RTX 5070 Ti 16 ГБ, драйвер 610.88.
  Минимальный объём VRAM на других GPU не измерялся; batch всегда 1.
- Место под комплект — около 1,22 ГБ; дополнительно несколько гигабайт
  под PyTorch, окружение и установочный кэш. Рекомендуется 12 ГБ свободного места.
- Интернет нужен для первой установки Python, драйвера и пакетов из PyPI/PyTorch.
  Веса скачивать не требуется. После установки инференс работает локально,
  с `local_files_only=True`, `HF_HUB_OFFLINE=1`, без токена Hugging Face.
  Wheel-файлы для полностью автономной установки в архив не включены.

CPU-инференс исходной модели не поддерживается: зафиксирован CUDA BF16 autocast
для энкодера и FP32 для головы/softmax. Параметры обеих частей остаются FP32.
Переключение на CPU или FP32-энкодер изменило бы численный режим; такой вариант
в этом комплекте не добавлен. `--device cpu` завершается понятной ошибкой.

## 1. Компоненты Windows PowerShell

Если Python 3.11.9 x64 уже установлен, установку Python пропустите.
Официальный источник: [Python 3.11.9](https://www.python.org/downloads/release/python-3119/).

```powershell
$installer = Join-Path $env:TEMP 'python-3.11.9-amd64.exe'
Invoke-WebRequest 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile $installer
Start-Process -FilePath $installer -ArgumentList '/passive','InstallAllUsers=0','Include_launcher=1','Include_test=0' -Wait -WindowStyle Hidden
py -3.11 --version
```

Должно быть `Python 3.11.9`. Если `py` не найден, заново откройте PowerShell.
Установите подходящий драйвер с [официальной страницы NVIDIA](https://www.nvidia.com/en-us/drivers/),
затем проверьте `nvidia-smi`. Отдельный CUDA Toolkit для этого wheel PyTorch не нужен.
Версии torch/torchvision и CUDA-вариант соответствуют
[официальным командам PyTorch 2.12.1](https://pytorch.org/get-started/previous-versions/#v2-12-1).

## 2. Распаковка и чистое окружение

Команды ниже предполагают, что `model_release.zip` лежит в текущей папке.
Архив содержит каталог `model_release/`.

```powershell
Expand-Archive -LiteralPath .\model_release.zip -DestinationPath .\unpacked
Set-Location .\unpacked\model_release
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r .\requirements.lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -B .\verify_package.py
```

Активация окружения и изменение ExecutionPolicy не требуются. Не используйте
`--system-site-packages`. Если папка уже распакована, перейдите в неё и начните
с создания окружения. `requirements.lock.txt` содержит шесть непосредственных
зависимостей (torch, torchvision, transformers, numpy, Pillow, safetensors) и
проверенное замыкание их обязательных зависимостей; это не список рабочего окружения.
Версии всех 37 пакетов закреплены. PyTorch использует индекс `cu130`.

## 3. Веса и целостность

Файлы уже находятся на нужных местах:

| Файл | Размер, байт | Назначение |
|---|---:|---|
| `weights/phikon-v2/model.safetensors` | 1 213 455 560 | полный замороженный энкодер |
| `weights/head.safetensors` | 1 060 304 | все шесть тензоров обученной головы |

Рядом с энкодером обязательны `config.json` и `preprocessor_config.json`.
Последний сохранён для происхождения нормализации: runtime использует исходные
`ToTensor` и `Normalize`, а не автоматический процессор с resize/crop.
Токенизатор для изображений не используется; пакет tokenizers является зависимостью Transformers.
Базовая ревизия: `2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd`.
Её [проверенный источник](https://huggingface.co/owkin/phikon-v2/tree/2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd)
и исходные SHA-256 записаны в `PROVENANCE.json`. Каждый файл комплекта, кроме
самого манифеста, перечислен в `MANIFEST.json` с размером и SHA-256.
Инференс проверяет этот список перед загрузкой модели. Не редактируйте файлы
выпуска без осознанной сборки нового манифеста.

## 4. Первый запуск на синтетическом примере

```powershell
.\.venv\Scripts\python.exe -B .\inference.py --image .\examples\synthetic.png --output .\outputs\synthetic_result.json --device cuda:0
Get-Content -LiteralPath .\outputs\synthetic_result.json
```

Перед загрузкой напечатается `Actual device: cuda:0 / ...; encoder BF16 autocast, head FP32`.
После обработки — путь результата и выбранный класс. Этот геометрический пример
проверяет только запуск; его классификация не имеет медицинского смысла.
Эталон технической проверки — `validation/synthetic_result.json`.
Точные числа могут различаться на другой GPU, драйвере или версии библиотек.

Синтетический файл можно воспроизвести без внешних данных:

```powershell
.\.venv\Scripts\python.exe -B .\make_example.py --output .\outputs\synthetic.png
.\.venv\Scripts\python.exe -B .\inference.py --image .\outputs\synthetic.png --output .\outputs\synthetic_result.csv --device cuda:0
```

## 5. Собственный вход и устройство

Допустим один **PNG, RGB, uint8, ровно 224×224**. Grayscale, RGBA, JPEG,
другой размер и целый скан не принимаются. Вход не ресайзится, не обрезается,
не перекрашивается и не нормализуется по окраске. Исходный контракт относится
к нативным полям гистологии простаты, приблизительно 0,5119 мкм/пиксель при 20×.
Физический масштаб нельзя вывести из одного PNG: его должен обеспечить источник.
Не сжимайте целый препарат до 224×224.

```powershell
.\.venv\Scripts\python.exe -B .\inference.py --image 'C:\input\patch224.png' --output 'C:\results\patch224.json' --device cuda:0
# Если имеется вторая совместимая NVIDIA GPU:
.\.venv\Scripts\python.exe -B .\inference.py --image 'C:\input\patch224.png' --output 'C:\results\patch224.csv' --device cuda:1
```

Пути входа/выхода могут быть абсолютными или относительно текущей папки.
Веса и конфигурации всегда ищутся относительно `inference.py`, независимо от cwd.
Результат создаётся по `--output`; родительский каталог создаётся автоматически.
Существующий результат по этому пути заменяется. Защищённые файлы комплекта не заменяются.
Запрошенная недоступная GPU вызывает ошибку и код выхода 2; подмена CPU не выполняется.

JSON UTF-8 содержит оба logits, оба softmax score, индекс и имя класса, SHA-256
изображения, технический QC и фактическое устройство. CSV UTF-8 с BOM содержит
одну строку, оба logits/score, класс, устройство и QC-флаги. Порядок классов:
`0=adenoma_hyperplasia`, `1=tumor`; гиперплазия не названа нормой.
Порог исходного адаптера: `score_tumor > 0.5`, при равенстве — класс 0.
Параметры нормализации: mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`,
после преобразования uint8 в FP32 / 255. Калибратора нет.

`qc_pixels.py` и `qc_config.json` сохранены без изменений. QC не изменяет пиксели
и не отменяет forward этого одноизображенческого CLI. Показатели и причины
технического отклонения сохраняются отдельно; ручные отметки старого датасета
не переносятся. CLI не выполняет отбор регионов/агрегацию полного препарата.
Успешный запуск, высокий score и отсутствие QC-флагов не подтверждают качество модели.

## 6. Публикация на GitHub — пока не выполнена

Для репозитория загрузите **всё содержимое `model_release/`**, включая скрытые
`.gitignore`, `.gitattributes`, `.python-version`, исходники, конфигурации,
`requirements.lock.txt`, `RUN.md`, `NOTICE.md`, `PROVENANCE.json`, `MANIFEST.json`,
`licenses/`, `examples/`, `validation/`, `weights/`.
Не загружайте созданные `.venv/`, `outputs/` и собственные входные данные.

Большой энкодер нельзя коммитить обычным Git: GitHub блокирует файлы свыше
100 MiB. Здесь подготовлен Git LFS через `.gitattributes`. Перед первым `git add`
установите Git с Git LFS и выполните в каталоге комплекта:

```powershell
git init
git lfs install
git add .
git lfs ls-files
```

В последней команде должен быть `weights/phikon-v2/model.safetensors`.
Команды публикации/добавления удалённого репозитория намеренно не выполнялись.
Владелец учётной записи проверяет квоту LFS. Получатель после клонирования выполняет
`git lfs pull`, затем `verify_package.py`. Маленький LFS-pointer не является весами.
`.gitattributes` сохраняет байты текстовых файлов, чтобы Git CRLF-конверсия не ломала SHA-256.

Другой способ передачи: приложить готовый **полный `model_release.zip` к GitHub Release**;
он содержит настоящие веса. ZIP не нужно коммитить в Git. Автоматический Source code ZIP
GitHub может не содержать LFS-объекты. Отдельный архив весов не нужен.
Правила: [размеры файлов](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github),
[Git LFS и лимиты](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).
Сохраните `NOTICE.md` и обе лицензии/атрибуцию в `licenses/`; ограничения Owkin
описаны там. Лицензия собственного кода не назначалась автоматически.

## 7. Типичные ошибки и границы проверки

| Сообщение | Действие |
|---|---|
| `Python ... 3.11.9` | Создайте новое окружение нужным интерпретатором, проверьте `--version`. |
| `No module named ...` | Установите lock-файл именно через `.venv\Scripts\python.exe -m pip`. |
| `CUDA is unavailable` | Проверьте NVIDIA GPU, `nvidia-smi`, драйвер и установку варианта `+cu130`. |
| `lacks native BF16` | Требуется другая GPU; автоматической замены точности нет. |
| `CPU inference is not supported` | Используйте CUDA с BF16. CPU-режим этой модели не экспортирован. |
| `CUDA out of memory` | Освободите VRAM от других задач или выберите другую GPU; batch уже 1. |
| `Expected PNG RGB ...` | Подготовьте исходное нативное поле нужного формата/размера. |
| `Missing file` / `SHA-256 mismatch` | Распакуйте полный архив повторно либо выполните `git lfs pull`; проверьте манифест. |
| `DLL load failed` | Проверьте x64 Python, драйвер и восстановите установку пакетов в новом окружении. |
| `Access denied` при сохранении | Укажите доступный для записи каталог через `--output`. |

Фактические проверки записаны в `validation/VALIDATION.json`: чистая установка,
загрузка обеих частей и полный CUDA forward; запуск с пустыми модельными кэшами,
запрещёнными Python-сетевыми соединениями и доступом к исходному проекту;
побитовое совпадение preprocessing и обеих вероятностей с исходным CLI на одном
синтетическом входе. CPU forward, другая физическая машина/GPU, Linux,
полностью офлайн-установка, WSI и клиническое качество не проверялись.
Технические проверки не являются оценкой диагностической точности.
