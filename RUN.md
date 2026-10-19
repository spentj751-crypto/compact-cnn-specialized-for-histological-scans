# Публичный комплект без весов

В этом каталоге есть код, конфигурация алгоритма и синтетический пример.
**Веса Phikon-v2 и обученной головы отсутствуют намеренно. Без обоих компонентов
полный инференс не запускается.** Данные пациентов, готовые предсказания,
обучающие датасеты и полный локальный архив сюда не входят.

## Проверка файлов без установки ML-зависимостей

После распаковки `model_release_public.zip` перейдите в `model_release_public/`:

```powershell
py -3.11 -B .\verify_package.py
py -3.11 -B .\inference.py --help
```

Первая команда проверяет размеры и SHA-256 только публичных файлов из
`MANIFEST.json`; отсутствующие приватные веса не считаются повреждением архива.
Второй режим проверяет также все отдельно размещённые компоненты модели:

```powershell
py -3.11 -B .\verify_package.py --with-weights
```

Без весов он завершится сообщением `This public package contains no model weights`
и перечислит недостающие файлы. Такая же проверка выполняется в `inference.py`
до импорта ML-библиотек и выделения памяти GPU. Пропуска проверки хешей нет.

## Установка для последующего локального запуска

Проверенная среда исходного комплекта: Windows x64, Python **3.11.9**,
torch **2.12.1+cu130**, torchvision **0.27.1+cu130**, transformers **5.12.1**.
Сохранён lock-файл 37 runtime-зависимостей; исходные параметры вычислений не менялись.
Нужна NVIDIA GPU с аппаратным BF16 и драйвером для CUDA 13.0.
Полный локальный комплект проверен на RTX 5070 Ti 16 ГБ, драйвер 610.88.
CPU forward не поддерживается исходным численным контрактом CUDA/BF16.

Если нужный Python ещё не установлен:

```powershell
$installer = Join-Path $env:TEMP 'python-3.11.9-amd64.exe'
Invoke-WebRequest 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile $installer
Start-Process -FilePath $installer -ArgumentList '/passive','InstallAllUsers=0','Include_launcher=1','Include_test=0' -Wait -WindowStyle Hidden
py -3.11 --version
```

Источник: [Python 3.11.9](https://www.python.org/downloads/release/python-3119/).
При необходимости установите драйвер с [сайта NVIDIA](https://www.nvidia.com/en-us/drivers/),
затем выполните `nvidia-smi`. Отдельный CUDA Toolkit для готового wheel не нужен.
Для установки зависимостей требуется интернет; установочные wheel-файлы в ZIP не входят.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r .\requirements.lock.txt
.\.venv\Scripts\python.exe -m pip check
```

Активация venv и изменение ExecutionPolicy не нужны. Не добавляйте
`--system-site-packages`. Если нужен только генератор синтетического изображения,
достаточно установить в окружение `Pillow==12.2.0` вместо всего lock-файла.

## Отдельное размещение локальных компонентов

Получите файлы только при наличии необходимых прав. Источник базовой модели:
[owkin/phikon-v2, закреплённая ревизия](https://huggingface.co/owkin/phikon-v2/tree/2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd).
Обученная голова конкретного проекта на этой странице отсутствует; публичная
ссылка на неё не предоставляется. Случайная голова или одни базовые веса её не заменят.
Условия внешней модели приведены в `NOTICE.md` и сохранённой лицензии.

Необходима структура относительно `inference.py`:

```text
weights/
  head.safetensors                      1 060 304 байта
  phikon-v2/
    model.safetensors                  1 213 455 560 байт
    config.json                        1 214 байт
    preprocessor_config.json           750 байт
```

Точные SHA-256 каждого файла находятся в `WEIGHTS_REQUIRED.json`.
Голова соответствует epoch 4, `run_20260928_135445`, шести FP32-тензорам
MLP 1024→256→2. Исходный обучающий `best.pt` нельзя просто переименовать
в `head.safetensors`.

Если у вас уже есть полный локальный комплект `model_release/`, скопируйте
его каталог `weights/` в вашу локальную рабочую копию публичного комплекта.
При расположении двух каталогов рядом команда из `model_release_public/`:

```powershell
Copy-Item -LiteralPath ..\model_release\weights -Destination .\weights -Recurse
.\.venv\Scripts\python.exe -B .\verify_package.py --with-weights
```

Копируйте в ещё не созданный каталог `weights/`. После проверки все файлы
модели локальны; ни исходный проект, ни HF-кэш, ни интернет для forward не нужны.
Веса занимают суммарно 1 214 515 864 байта; это не размер публичного ZIP.

## Запуск после размещения весов

```powershell
.\.venv\Scripts\python.exe -B .\inference.py --image .\examples\synthetic.png --output .\outputs\synthetic_result.json --device cuda:0
# Воспроизведение синтетического входа:
.\.venv\Scripts\python.exe -B .\make_example.py --output .\outputs\synthetic.png
# Собственный вход:
.\.venv\Scripts\python.exe -B .\inference.py --image 'C:\input\patch224.png' --output 'C:\results\patch224.csv' --device cuda:0
```

Вход — строго PNG RGB uint8 224×224, нативное поле гистологии простаты,
приблизительно 0,5119 мкм/пиксель. Физический масштаб обеспечивает источник:
из одного PNG он не проверяется. Resize/crop/перекрашивания нет. Целый скан,
JPEG, RGBA, grayscale и другие размеры не поддерживаются этим CLI.
ToTensor делит uint8 на 255; Normalize использует mean `[0.485,0.456,0.406]`
и std `[0.229,0.224,0.225]`. Энкодер — BF16 autocast на CUDA, параметры и
голова — FP32, выход — FP32 softmax. Классы `adenoma_hyperplasia=0`, `tumor=1`;
выбор tumor при score строго больше 0,5. Калибратора нет.

`--device cuda:1` выбирает вторую GPU, если она есть. Фактическое устройство
печатается в консоли и сохраняется в результате. CPU, недоступная GPU или
отсутствие BF16 вызывают понятную ошибку; изменения точности автоматически нет.
Недостающие веса проверяются раньше доступности GPU.

JSON UTF-8 содержит logits, softmax scores, класс, SHA-256 входа, пиксельный QC
и фактическое устройство; CSV UTF-8 с BOM — одну строку с обоими logits/score,
классом, устройством и QC-флагами. Файл задаёт `--output`, каталог создаётся
автоматически, существующий результат заменяется. Файлы выпуска защищены от перезаписи.
Веса ищутся относительно приложения, вход/выход — относительно cwd либо по полному пути.
QC не меняет изображение и не отменяет forward этого CLI.

Синтетический пример содержит только геометрические цвета. Он проверяет запуск
и не доказывает качество модели. Результат отдельного патча не является диагнозом препарата.

## Публикация этой версии

Публикуйте содержимое **только `model_release_public/`**. Не добавляйте полный
локальный `model_release/`, `model_release.zip`, собственные входы и результаты.
`.gitignore` исключает весь `weights/`, распространённые форматы весов, окружения,
данные, `.env`, кэши и результаты. Git LFS для этой версии не требуется.
Прежде чем создавать коммит в отдельной публичной копии:

```powershell
git init
git add .
git diff --cached --name-only
git ls-files -- weights
```

Последняя команда должна быть пустой. Не используйте `git add -f` для весов.
`.gitignore` не убирает ранее отслеживаемые файлы, поэтому начинайте с новой
папки/репозитория. Сохраните `NOTICE.md`, лицензию и сведения о происхождении.
Архив `model_release_public.zip` также не содержит весов.

## Типичные ошибки и проверки

- `contains no model weights`: разместите все четыре файла по `WEIGHTS_REQUIRED.json`.
- `Size/SHA-256 mismatch`: выбран другой файл/ревизия либо файл повреждён; восстановите
  нужную версию. LFS-pointer вместо настоящих весов тоже не пройдёт проверку.
- `No module named ...`: установите lock-файл через Python именно из `.venv`.
- `CUDA is unavailable` / `lacks native BF16`: проверьте GPU, драйвер и CUDA wheel.
- `CPU inference is not supported`: нужен совместимый CUDA GPU.
- `Expected PNG RGB ...`: нужен исходный PNG RGB 224×224, без скрытого преобразования.
- `CUDA out of memory`: освободите VRAM или выберите другую GPU; batch уже 1.

В публичном ZIP полный forward невозможен по составу. Проверки именно этой сборки
перечислены в `PUBLIC_VALIDATION.json`; в ней нет предсказаний модели.
