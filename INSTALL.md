# Установка и настройка окружения

Инструкция для Linux (Ubuntu 22.04/24.04) и Windows через WSL2. macOS годится
только для подготовки и анализа — CHAPERONg рассчитан на Linux, а GROMACS без
CUDA на Mac считает такие системы неприемлемо долго.

Итоговый стек:

```
GROMACS 2023+ (CUDA)  ──┐
CHAPERONg (bash)      ──┼── cg-us (этот пакет, Python 3.10+)
charmm36 force field  ──┘
```

---

## 0. Что понадобится по железу

| Ресурс | Минимум | Комментарий |
|---|---|---|
| CPU | 8 ядер | одно окно US = один mdrun |
| GPU | 1× NVIDIA (8 ГБ) | без GPU 20 нс × 26 окон × 3 реплики нереальны |
| Диск | ~40 ГБ на систему | 26 окон × 20 нс, `nstxout-compressed = 25000` |
| RAM | 16 ГБ | |

Оценка времени для bench6 при настройках по умолчанию (20 нс на окно,
~26 окон, 3 реплики): **≈ 1600 нс на систему**, то есть порядка 2–5 GPU-суток
на систему на RTX 4090 для комплексов ~60–110 тыс. атомов. Планируйте очередь.

---

## 1. Системные пакеты

```bash
sudo apt update
sudo apt install -y build-essential cmake git wget curl \
                    libfftw3-dev libopenmpi-dev openmpi-bin \
                    grace pymol imagemagick ghostscript-x ffmpeg \
                    python3-pip python3-venv dos2unix
```

`grace` (gracebat), `pymol` и `imagemagick` нужны только CHAPERONg — он рисует
ими промежуточные картинки и «фильм» траектории SMD. Подробности по стадии 13 —
в разделе 4.1.

WSL2: то же самое внутри дистрибутива, плюс в Windows поставьте драйвер NVIDIA
с поддержкой WSL — тогда `nvidia-smi` работает внутри WSL и GROMACS видит GPU.

---

## 2. GROMACS

### 2.1. Вариант «быстро» — conda

```bash
conda create -n md -c conda-forge gromacs=2024 -y
conda activate md
gmx --version
```

Годится для отладки пайплайна. Сборка из conda-forge обычно без CUDA — считать
продакшн на ней не стоит.

### 2.2. Вариант «правильно» — сборка с CUDA

```bash
wget https://ftp.gromacs.org/gromacs/gromacs-2024.3.tar.gz
tar xf gromacs-2024.3.tar.gz && cd gromacs-2024.3
mkdir build && cd build

cmake .. \
  -DGMX_BUILD_OWN_FFTW=ON \
  -DGMX_GPU=CUDA \
  -DCMAKE_INSTALL_PREFIX=/opt/gromacs-2024.3 \
  -DGMX_DOUBLE=OFF \
  -DGMX_SIMD=AVX2_256          # AVX_512 для Xeon Scalable

make -j"$(nproc)"
make check                      # ~20 минут, стоит того
sudo make install
echo 'source /opt/gromacs-2024.3/bin/GMXRC' >> ~/.bashrc
source ~/.bashrc
```

Проверка:

```bash
gmx --version | grep -E "GROMACS version|GPU support"
```

**Требования к версии.** Пакет генерирует `.mdp` с `pcoupl = C-rescale` — это
GROMACS ≥ 2021. Автоматическое замыкание циклических пептидов в `pdb2gmx`
(пороги `-sb`/`-lb`) появилось в 2021.1. Если у вас 2020.x, замените в
`protocol.yaml` баростат вручную (см. раздел 7 «Старый GROMACS»); циклопептиды
на такой версии придётся замыкать патчем топологии. `gmx wham` с
`-nBootstrap`/`-bs-method b-hist` есть во всех версиях начиная с 5.x.

---

## 3. Силовое поле CHARMM36

GROMACS не содержит CHARMM36 в комплекте — его нужно положить рядом.

```bash
mkdir -p ~/ff && cd ~/ff
wget http://mackerell.umaryland.edu/download.php?filename=CHARMM_ff_params_files/charmm36-jul2022.ff.tgz \
     -O charmm36-jul2022.ff.tgz
tar xf charmm36-jul2022.ff.tgz
ls charmm36-jul2022.ff/forcefield.doc
```

Если ссылка изменилась — берите актуальный «GROMACS port» со страницы
MacKerell Lab (раздел *CHARMM force field files → GROMACS*).

Дальше есть два способа отдать поле в pdb2gmx:

* **рекомендуемый** — передать каталог в `cg-us prep --ff-dir ~/ff/charmm36-jul2022.ff`;
  он будет симлинкнут в каждую рабочую папку реплики, и `pdb2gmx -ff charmm36-jul2022`
  подхватит его из рабочей директории;
* глобально — скопировать `charmm36-jul2022.ff` в `$GMXDATA/top`
  (`/opt/gromacs-2024.3/share/gromacs/top`).

В `protocol.yaml` имя поля должно совпадать с именем каталога без `.ff`:

```yaml
prep:
  force_field: charmm36-jul2022
  water: tip3p
```

Для AMBER ничего скачивать не надо: `amber99sb-ildn` идёт в комплекте, тогда
`force_field: amber99sb-ildn` и `ff_in_workdir: false`.

---

## 4. CHAPERONg

```bash
cd ~/soft
git clone https://github.com/abeebyekeen/CHAPERONg.git
cd CHAPERONg
dos2unix CHAP_modules/*.sh CHAP_utilities/*.py install_CHAPERONg.sh   # на всякий случай
chmod +x install_CHAPERONg.sh
./install_CHAPERONg.sh          # отвечать y
source ~/.bashrc
```

Инсталлятор дописывает в `~/.bashrc` переменную `CHAPERONg_PATH` и добавляет
`$CHAPERONg_PATH/CHAP_modules` в `PATH`. Проверка:

```bash
echo "$CHAPERONg_PATH"
run_CHAPERONg.sh -v
run_CHAPERONg.sh -H | head -30
```

Если `run_CHAPERONg.sh` не находится — добавьте руками:

```bash
export CHAPERONg_PATH="$HOME/soft/CHAPERONg"
export PATH="$PATH:$CHAPERONg_PATH/CHAP_modules"
```

Зависимости CHAPERONg (опциональные, но лучше поставить):

```bash
conda activate base
cd ~/soft/CHAPERONg && chmod +x conda_env_setup.sh && ./conda_env_setup.sh
conda activate chaperong
```

> `conda_env_setup.sh` создаёт окружение `chaperong` с numpy/scipy/matplotlib/
> pandas/networkx/pymol. Если оно ставится долго или конфликтует — можно обойтись
> системными пакетами из шага 1 плюс `pip install numpy scipy matplotlib pandas`.

### 4.1. Зависимости стадии 13 («фильм» SMD-траектории)

Стадия 13 рендерит ролик разъединения комплекса. К PMF она отношения не имеет,
но в CHAPERONg встроена в цепочку: любая точка входа с номером ≤ 13 её
выполняет. Что она вызывает и что для этого нужно:

| Пакет | Что делает | Обязателен |
|---|---|---|
| `pymol` | `mpng` с `ray_trace_frames=1` — рендерит 200–300 PNG | да, без него ролика нет |
| `imagemagick` (`convert`) | склейка PNG → `dynamics_movie.gif` и `.mp4` | да |
| `ffmpeg` | делегат ImageMagick для mp4; также нужен PyMOL `movie.produce` в запасном пути | для mp4 |
| `xvfb` | виртуальный X-сервер, если PyMOL всё же требует дисплей | обычно нет |
| `grace` (`gracebat`) | графики xvg → png на других стадиях | желательно |

```bash
sudo apt install -y pymol imagemagick ffmpeg xvfb ghostscript-x
# или в conda-окружении chaperong:
conda install -c conda-forge pymol-open-source imagemagick ffmpeg
```

Три вещи, о которых стоит знать заранее:

1. **PyMOL запускается в GUI-режиме.** CHAPERONg вызывает `pymol script.pml`
   без `-c`, и на headless-узле по SSH это падает с ошибкой дисплея. cg-us
   кладёт в `<реплика>/bin/pymol` шим, который подставляет `pymol -cq`
   (и `xvfb-run -a`, если он установлен), и добавляет этот каталог в начало PATH
   дочернего процесса. Отключается через `run.pymol_headless: false`.
2. **ImageMagick 7** ставит бинарник `magick`, а `convert` оставляет как legacy —
   в некоторых сборках его нет вообще. Тогда сделайте симлинк
   `ln -s "$(which magick)" ~/bin/convert` или ставьте `imagemagick-6.q16`.
   Кроме того, `/etc/ImageMagick-6/policy.xml` часто режет память и диск —
   при склейке сотен PNG это вылезает как «cache resources exhausted»;
   поднимите `memory`, `map` и `disk` или выключите ролик.
3. **Ray tracing — это дорого.** 300 кадров крупного комплекса считаются
   десятками минут на CPU и идут *последовательно*, пока GPU простаивает.
   Варианты в `protocol.yaml`:

```yaml
run:
  skip_movie: false    # true — полностью обойти стадию 13
  movie_frames: 60     # 0 = дефолт CHAPERONg (200-300); 60 быстрее в ~5 раз
  pymol_headless: true
```

`movie_frames` заодно убирает интерактивный вопрос про длину ролика.

**Как cg-us обходит стадию 13 при `skip_movie: true`.** Цепочку CHAPERONg
нельзя прервать на 12-й стадии, поэтому работа режется на две сессии: первая
останавливается ровно на промпте `Enter 1 or 2 here` (steered MD к этому моменту
уже записан на диск, наличие `pull.gro` проверяется), вторая заходит заново на
стадию 14, которая начинается с извлечения кадров. Логи при этом лежат в
`chaperong_stage0.log` и `chaperong_stage14.log`.

**Важно про интерактивность.** CHAPERONg целиком построен на `read -p`. Пакет
`cg-us` отвечает на эти вопросы через `pexpect` по шаблонам промптов. Если вы
обновите CHAPERONg и формулировки вопросов изменятся, драйвер упадёт с понятной
ошибкой вида `CHAPERONg ended before prompt '<label>'` — тогда либо поправьте
шаблоны в `src/cg_us/backends/chaperong.py::script`, либо переключитесь на
`backend: direct`.

---

## 5. Пакет cg-us

```bash
python3 -m venv ~/venv/cgus
source ~/venv/cgus/bin/activate
cd ~/soft/cg-us            # каталог с pyproject.toml
pip install -e .
cg-us --help
```

Если вы работаете внутри conda-окружения `chaperong`, ставьте туда же:

```bash
conda activate chaperong
pip install -e ~/soft/cg-us
```

Зависимости: numpy, pandas, scipy, matplotlib, pyyaml, pexpect. Ничего
компилируемого.

---

## 6. Проверка окружения

### 6.1. Манифест и структуры

```bash
cg-us validate --manifest ~/data/bench6/manifest.csv
```

Ожидаемый вывод — таблица из шести систем, пять с экспериментальной ΔG и один
контроль без неё.

### 6.2. Подготовка дерева расчётов

```bash
cg-us prep \
  --manifest ~/data/bench6/manifest.csv \
  --root ~/runs/bench6 \
  --protocol protocol.yaml \
  --ff-dir ~/ff/charmm36-jul2022.ff
```

Проверьте предупреждения: для `trypsin_sfti` и `trypsin_mcoti` должно появиться
сообщение про head-to-tail замыкание (N–C ≈ 1.33 Å) и про дисульфиды.

### 6.3. Короткий дымовой прогон

Не запускайте сразу продакшн. Сделайте отдельный протокол на минуту счёта:

```bash
cp protocol.yaml smoke.yaml
python - <<'PY'
import yaml
p = yaml.safe_load(open("smoke.yaml"))
p["replicas"] = 1
p["smd"]["time_ns"] = 0.02
p["umbrella"]["time_ns"] = 0.05
p["umbrella"]["equil_ns"] = 0.01
p["umbrella"]["discard_ns"] = 0.0
p["umbrella"]["window_spacing"] = 0.3
p["umbrella"]["max_distance"] = 1.0
p["wham"]["bootstraps"] = 20
p["run"]["backend"] = "direct"
yaml.safe_dump(p, open("smoke.yaml", "w"))
PY

cg-us prep --manifest ~/data/bench6/manifest.csv --root ~/runs/smoke \
           --protocol smoke.yaml --ff-dir ~/ff/charmm36-jul2022.ff
cg-us run  --root ~/runs/smoke --system mdm2_p53
cg-us analyze --root ~/runs/smoke --no-convergence
```

На выходе должны появиться `~/runs/smoke/report.html` и PNG в
`~/runs/smoke/analysis/`. Числа физического смысла иметь не будут — проверяется
только проходимость пайплайна.

### 6.4. Тесты пакета

```bash
pip install pytest
pytest -q          # 7 тестов, GROMACS не нужен
```

---

## 7. Частые проблемы

**`Fatal error: Atom N in residue GLY 1 was not found in rtp entry`**
Обычно нестандартные остатки или отсутствующие атомы. Проверьте, что в манифесте
указаны правильные цепи, и что вы не тянете в расчёт HETATM-лиганды.

**`Fatal error: number of coordinates in coordinate file does not match topology`**
Топология и `.gro` рассинхронизировались — почти всегда из-за повторного запуска
поверх старой рабочей папки. Удалите папку реплики и запустите заново
(`cg-us run --force`).

**`Group Target not found in index file`**
`index.ndx` не был создан или создан не в тот момент. В бэкенде `chaperong`
индекс пишется хуком на промпте *«Do you need to make custom index for pulling
groups?»* — если вы отвечали на этот вопрос вручную, индекса не будет. Проверьте
`ls <workdir>/index.ndx` и наличие в нём групп `[ Target ]` / `[ Binder ]`.

**`1-4 interaction between X and Y at distance > cut-off` при grompp**
Обычно это следствие того, что pdb2gmx не замкнул циклический пептид, а
структура при этом циклическая. Посмотрите `topol_Protein_chain_<байндер>.itp` —
должна быть связь между N первого и C последнего остатка. Порог замыкания
задаётся `prep.cyclic_lb` (по умолчанию 0.4 нм вместо штатных 0.25).

**Циклический пептид замкнулся там, где не надо**
Проверьте `cyclic_lb_nm` в `systems/<name>/prep.json` и колонку `cyclic_type` в
манифесте: для не-head-to-tail связок cg-us сам опускает `-lb` до 0.25 нм. Если
манифест молчит о типе цикла, порог остаётся 0.4 нм — впишите `cyclic_type`.

**`cg-us: this gmx build rejects -sb 0.05 -lb 0.4` в логе**
`-sb`/`-lb` — скрытые опции `pdb2gmx`; на вашей сборке их нет. Запуск
продолжится с дефолтами (0.05–0.25 нм), но длинные замыкания (> 2.5 Å) не
распознаются — для таких систем нужен GROMACS ≥ 2021.1.

**Старый GROMACS (2020.x):** замените баростат в сгенерированных `.mdp`

```bash
sed -i 's/pcoupl                   = C-rescale/pcoupl                   = Berendsen/' \
    ~/runs/bench6/systems/*/rep*/npt*.mdp
```

или отредактируйте `src/cg_us/mdp.py` (одна строка в `_npt`/`_npt_umbrella`).

**`ExceptionPexpect: The command was not found or was not executable: run_CHAPERONg.sh`**
`pexpect` ищет команду в PATH своего процесса, а не в том окружении, которое ему
передают, поэтому экспортированного `CHAPERONg_PATH` мало — нужен ещё
`CHAP_modules` в PATH. Начиная с 0.2.1 cg-us разрешает путь сам (PATH, затем
`$CHAPERONg_PATH/CHAP_modules`, затем `$CHAPERONg_PATH`) и проверяет установку
до старта расчётов. Если авто-поиск не срабатывает, укажите путь явно:

```bash
cg-us run --root ... --chaperong "$CHAPERONg_PATH/CHAP_modules/run_CHAPERONg.sh"
```

или в `protocol.yaml`:

```yaml
run:
  chaperong: /home/orrls/soft/CHAPERONg/CHAP_modules/run_CHAPERONg.sh
```

**`pexpect.exceptions.TIMEOUT` в бэкенде chaperong**
mdrun идёт дольше, чем `run.timeout_s`. Увеличьте значение в `protocol.yaml`
(по умолчанию 48 часов) или переходите на `direct` + Slurm.

**PyMOL падает на стадии 13**
Это стадия «фильм SMD-траектории», для расчёта она не нужна. CHAPERONg её
переживает, лог будет с ошибками. Если мешает — используйте `backend: direct`.

---

## 8. Запуск на кластере (Slurm)

Бэкенд `direct` разделяет пайплайн на стадии, поэтому его удобно резать на джобы.
Подготовка (быстрая, CPU):

```bash
cg-us run --root ~/runs/bench6 --backend direct --stop index
```

Продакшн (GPU-джоб на систему):

```bash
#!/bin/bash
#SBATCH -J us-%x
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 48:00:00

source /opt/gromacs-2024.3/bin/GMXRC
source ~/venv/cgus/bin/activate

cg-us run --root ~/runs/bench6 --backend direct \
          --system "$SYSTEM" --replica "$REP" --start nvt
```

```bash
for s in mdm2_p53 mdm2_pmi keap1_nrf2 trypsin_sfti trypsin_mcoti actrIIb_bimagrumab_cdr3; do
  for r in 1 2 3; do
    sbatch --export=SYSTEM=$s,REP=$r --job-name="$s-r$r" us.sbatch
  done
done
```

Если на узле несколько GPU, поднимите `run.window_workers` (окна пойдут
параллельно внутри одной реплики) и задайте `run.gpu_id`.

Анализ — CPU-джоб, `gmx wham` быстрый:

```bash
cg-us analyze --root ~/runs/bench6
```
