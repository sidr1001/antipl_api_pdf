import os
import json
import fitz  # PyMuPDF
import subprocess
import base64
import io
import yake
import random
import datetime
import time
import uuid
import re
from difflib import SequenceMatcher
from datetime import timedelta
from urllib.parse import urlparse
import pdfkit
import matplotlib
matplotlib.use('Agg') # Обязательно для работы на сервере без экрана
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file, jsonify, url_for, redirect
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)
# Устанавливаем лимит 50 Мб
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 

# 1. Логируем размер КАЖДОГО входящего запроса
@app.before_request
def log_request_info():
    # Получаем длину контента из заголовков
    content_length = request.content_length
    if content_length:
        # Переводим в мегабайты для удобства
        mb_size = content_length / (1024 * 1024)
        print(f"--> [DEBUG] Входящий запрос: {content_length} байт ({mb_size:.2f} MB)", flush=True)
    else:
        print(f"--> [DEBUG] Входящий запрос: Content-Length не указан", flush=True)

# 2. Перехватываем ошибку 413, чтобы точно знать, что это Flask
@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def handle_413_error(e):
    print(f"--> [ERROR] ОШИБКА 413 ПЕРЕХВАЧЕНА ВО FLASK! Запрос превысил лимит.", flush=True)
    print(f"    Текущий лимит app.config['MAX_CONTENT_LENGTH']: {app.config.get('MAX_CONTENT_LENGTH')}", flush=True)
    return jsonify({"error": "File too large (Flask limit hit)", "size_limit": app.config['MAX_CONTENT_LENGTH']}), 413

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
PATH_TO_LO = '/usr/bin/libreoffice' 


def resolve_times_font():
    """
    Ищет Times New Roman в системе.
    Возвращает путь к TTF/OTF, если найден, иначе None.
    """
    candidates = [
        '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf',
        '/usr/share/fonts/truetype/msttcorefonts/times.ttf',
        '/usr/share/fonts/truetype/msttcorefonts/timesnewroman.ttf',
        '/usr/share/fonts/truetype/msttcorefonts/Times New Roman.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
    ]

    for font_path in candidates:
        if os.path.exists(font_path):
            return font_path
    return None


TIMES_FONT_FILE = resolve_times_font()

# Настройка wkhtmltopdf (укажите правильный путь, если отличается)
config_pdf = pdfkit.configuration(wkhtmltopdf='/usr/local/bin/wkhtmltopdf')

# --- ЦВЕТА ---
HIGHLIGHT_COLOR = (253/255, 183/255, 157/255) 
BADGE_COLOR = (255/255, 114/255, 58/255)      
BADGE_TEXT_COLOR = (1, 1, 1)

# === 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===
def parse_compressed_indices(words_str):
    indices = set()
    if not words_str: return indices
    parts = str(words_str).split()
    for part in parts:
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                indices.update(range(start, end + 1))
            except: pass
        else:
            try:
                indices.add(int(part))
            except: pass
    return indices

def safe_percent(value, default=0.0):
    """Безопасно приводит значение процента к float."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)

    raw = str(value).strip().replace('%', '').replace(',', '.')
    if not raw:
        return default

    try:
        return float(raw)
    except:
        return default


def get_api_metric(api_data, keys, default=0.0):
    """Берет первый найденный ключ из api_data и приводит к float."""
    for key in keys:
        if key in api_data and api_data.get(key) is not None:
            return safe_percent(api_data.get(key), default)
    return default

def normalize_word_token(token):
    """Нормализация слова для выравнивания потоков."""
    if token is None:
        return ""
    token = str(token).lower().replace('ё', 'е')
    token = re.sub(r'[^a-zа-я0-9]+', '', token, flags=re.IGNORECASE)
    return token


def extract_pdf_words_with_meta(doc, skip_margins_tables=False, margin_top=80, margin_bottom=80):
    """Извлекает слова PDF с метаданными, опционально фильтруя поля/таблицы."""
    stream = []
    for page_num, page in enumerate(doc):
        try:
            tables = page.find_tables()
            table_bboxes = [fitz.Rect(tab.bbox) for tab in tables]
        except:
            table_bboxes = []

        page_height = page.rect.height
        words_on_page = page.get_text("words")
        words_on_page.sort(key=lambda w: (round(w[1], 1), w[0]))

        for w in words_on_page:
            rect = fitz.Rect(w[:4])
            skip = False
            if skip_margins_tables:
                if rect.y1 < margin_top or rect.y0 > (page_height - margin_bottom):
                    skip = True
                if not skip:
                    for table_rect in table_bboxes:
                        if table_rect.intersects(rect):
                            skip = True
                            break

            text = w[4]
            stream.append({
                'page': page_num,
                'rect': rect,
                'text': text,
                'norm': normalize_word_token(text),
                'skip': skip,
            })
    return stream


def align_streams_with_anchors(source_tokens, target_tokens, ngram=5):
    """
    Production-метод:
    1) n-gram якоря
    2) локальное выравнивание SequenceMatcher
    3) коррекция смещения
    Возвращает map source_idx -> target_idx (или None)
    """
    n = max(3, ngram)
    src_len, tgt_len = len(source_tokens), len(target_tokens)
    mapping = [None] * src_len

    if src_len == 0 or tgt_len == 0:
        return mapping

    def build_ngram_index(tokens, nsize):
        idx = {}
        for i in range(0, max(0, len(tokens) - nsize + 1)):
            ng = tuple(tokens[i:i+nsize])
            idx.setdefault(ng, []).append(i)
        return idx

    src_ng = build_ngram_index(source_tokens, n)
    tgt_ng = build_ngram_index(target_tokens, n)

    anchors = []
    for ng, src_pos in src_ng.items():
        tgt_pos = tgt_ng.get(ng)
        if not tgt_pos:
            continue
        if len(src_pos) == 1 and len(tgt_pos) == 1:
            anchors.append((src_pos[0], tgt_pos[0]))

    anchors.sort(key=lambda x: x[0])

    # Монотонный набор якорей
    mono = []
    last_t = -1
    for sp, tp in anchors:
        if tp > last_t:
            mono.append((sp, tp))
            last_t = tp

    # Границы
    boundaries = [(-1, -1)] + mono + [(src_len, tgt_len)]

    # 2) локальное выравнивание между якорями
    for (s0, t0), (s1, t1) in zip(boundaries, boundaries[1:]):
        ss, se = s0 + 1, s1
        ts, te = t0 + 1, t1
        if ss >= se or ts >= te:
            continue

        sm = SequenceMatcher(None, source_tokens[ss:se], target_tokens[ts:te], autojunk=False)
        for a, b, size in sm.get_matching_blocks():
            if size == 0:
                continue
            for k in range(size):
                mapping[ss + a + k] = ts + b + k

    # Якорные n-граммы заполняем явно
    for sp, tp in mono:
        for k in range(n):
            si = sp + k
            ti = tp + k
            if 0 <= si < src_len and 0 <= ti < tgt_len:
                mapping[si] = ti

    # 3) коррекция смещения для дыр
    known = [(i, t) for i, t in enumerate(mapping) if t is not None]
    if known:
        known_idx = [i for i, _ in known]
        known_delta = [t - i for i, t in known]

        for i in range(src_len):
            if mapping[i] is not None:
                continue
            # локальная медиана смещения по соседям
            neigh = []
            for k in range(max(0, i-80), min(src_len, i+81)):
                t = mapping[k]
                if t is not None:
                    neigh.append(t - k)
            if neigh:
                neigh.sort()
                d = neigh[len(neigh)//2]
            else:
                d = known_delta[len(known_delta)//2]

            cand = i + d
            if 0 <= cand < tgt_len:
                mapping[i] = cand

    return mapping


def build_highlight_blocks_from_mapping(source_to_target, source_to_meta, target_stream):
    """Строит блоки подсветки по map source_idx -> target_idx."""
    page_blocks_map = {}
    for w in target_stream:
        page_blocks_map.setdefault(w['page'], [])

    triples = []
    for src_idx, meta in source_to_meta.items():
        if src_idx < 0 or src_idx >= len(source_to_target):
            continue
        tgt_idx = source_to_target[src_idx]
        if tgt_idx is None or tgt_idx < 0 or tgt_idx >= len(target_stream):
            continue
        tw = target_stream[tgt_idx]
        triples.append((tgt_idx, tw, meta))

    triples.sort(key=lambda x: x[0])

    current = None
    for _, tw, meta in triples:
        rect = tw['rect']
        x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1 + 2, rect.y1

        if (
            current
            and current['sid'] == meta['sid']
            and current['page'] == tw['page']
            and abs(current['y'] - y0) < 5
            and x0 <= current['x'] + current['w'] + 14
        ):
            current['w'] = max(current['w'], x1 - current['x'])
            current['h'] = max(current['h'], y1 - y0)
        else:
            if current:
                page_blocks_map[current['page']].append(current)
            current = {
                'page': tw['page'],
                'x': x0,
                'y': y0,
                'w': x1 - x0,
                'h': y1 - y0,
                'sid': meta['sid'],
                'cit': meta['cit']
            }

    if current:
        page_blocks_map[current['page']].append(current)

    return page_blocks_map

def save_file_unique(content, filename, base_folder="uploads"):
    """
    Сохраняет файл в структуру папок /uploads/{uid}/filename
    """
    
    # 1. Генерируем уникальный UID
    uid = str(uuid.uuid4())
    
    # 2. Формируем путь к директории: uploads/{uid}
    directory_path = os.path.join(base_folder, uid)
    
    # 3. Создаем директорию. exist_ok=True не вызовет ошибку, если папка уже есть
    os.makedirs(directory_path, exist_ok=True)
    
    # 4. Формируем полный путь к файлу
    full_file_path = os.path.join(directory_path, filename)
    
    # 5. Записываем файл
    try:
        with open(full_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Успешно сохранено: {full_file_path}")
        return full_file_path, uid
        
    except Exception as e:
        print(f"Ошибка при сохранении: {e}")
        return None, None

# --- ГРАФИКИ ---
def generate_pie_chart_base64(pz, po, pc):
    """
    Рисует круговую диаграмму.
    Параметры настроены под дизайн оригинала (толстое кольцо).
    """
    labels = []
    sizes = []
    colors = []
    
    # Цвета (как в оригинале)
    if po > 0:
        sizes.append(po); colors.append('#A2BCDD') # Оригинальность (Голубой)
    if pz > 0:
        sizes.append(pz); colors.append('#FF723A') # Заимствования (Оранжевый)
    if pc > 0:
        sizes.append(pc); colors.append('#B2CF55') # Цитирования (Зеленый)
    
    # Если данных нет - серый круг
    if not sizes: sizes = [100]; colors = ['#EEEEEE']

    # --- НАСТРОЙКИ РАЗМЕРОВ ---
    
    # 1. Размер картинки (в дюймах). 
    # При стандартном DPI=100, (3, 3) создаст изображение 300x300 пикселей.
    # Это даст хорошее качество при сжатии до 200px в PDF.
    fig, ax = plt.subplots(figsize=(3, 3))
    
    # 2. Толщина кольца (Donut Chart)
    # width - это толщина цветной части (от 0 до 1).
    # В JS innerRadius был 0.25 (дырка 25%).
    # Значит толщина кольца должна быть 1 - 0.25 = 0.75.
    # Если хотите кольцо тоньше — уменьшите до 0.5.
    donut_width = 0.75 
    
    wedges, texts = ax.pie(
        sizes, 
        colors=colors, 
        startangle=90,    # 90 градусов - начинаем сверху (12 часов)
        counterclock=False, # Рисовать по часовой стрелке (как обычно в отчетах)
        wedgeprops=dict(
            width=donut_width, 
            edgecolor='white', # Белая разделительная линия между дольками
            linewidth=1        # Толщина разделителя
        )
    )
    
    ax.axis('equal')  # Чтобы круг был идеально круглым, а не овальным
    
    # Сохранение
    img = io.BytesIO()
    
    # bbox_inches='tight' и pad_inches=0 обрезают ВСЕ белые поля вокруг круга. transparent=True,
    # Круг будет занимать 100% картинки.
    plt.savefig(img, format='png', bbox_inches='tight', pad_inches=0)
    
    img.seek(0)
    plt.close(fig) # Очищаем память
    
    return base64.b64encode(img.getvalue()).decode('utf-8')

# --- РИСОВАНИЕ В PDF ---
def draw_stylish_badge(page, rect, text, bg_color=None):
    """
    Рисует шильдик, который чуть шире (выше) самой строки текста.
    """
    # === ЦВЕТА ===
    if bg_color is not None:
        final_fill_color = bg_color
    else:
        final_fill_color = (255/255, 114/255, 58/255) # BADGE_COLOR
        
    final_text_color = (1, 1, 1) 
    # =============

    # 1. Считаем высоту строки
    line_height = rect.y1 - rect.y0
    
    # 2. ИЗМЕНЕНИЕ: Делаем высоту шильдика больше строки (115%)
    # Если хотите еще мощнее, поставьте 1.2 или 1.3
    badge_height = line_height * 1.15 
    
    # 3. Находим точный центр строки по вертикали
    center_y = rect.y0 + (line_height / 2)
    
    # 4. Вычисляем верх и низ шильдика от центра
    top_y = center_y - (badge_height / 2)
    bottom_y = center_y + (badge_height / 2)
    
    start_x = rect.x1 + 1 
    
    triangle_width = 4   # Острый носик
    rect_width = 16      
    
    str_num = str(text)
    if len(str_num) > 1: 
        rect_width += (len(str_num) - 1) * 6

    # Геометрия стрелки
    p1 = fitz.Point(start_x, center_y)                          
    p2 = fitz.Point(start_x + triangle_width, top_y)            
    p3 = fitz.Point(start_x + triangle_width + rect_width, top_y) 
    p4 = fitz.Point(start_x + triangle_width + rect_width, bottom_y) 
    p5 = fitz.Point(start_x + triangle_width, bottom_y)         

    shape = page.new_shape()
    if hasattr(shape, 'draw_polyline'): 
        shape.draw_polyline([p1, p2, p3, p4, p5, p1])
    else: 
        shape.draw_poly([p1, p2, p3, p4, p5, p1])
         
    shape.finish(fill=final_fill_color, color=None) 
    shape.commit()

    # Вставка текста
    # Шрифт подстраиваем под новую высоту (чуть уменьшаем кэф, чтобы не прилипал к краям)
    font_size = badge_height * 0.90 
    
    # Используем реальный TTF/OTF, если доступен.
    # Это гарантирует одинаковый вид шрифта в местах окраса (бейджи),
    # а не системный fallback от PDF-ридера.
    if TIMES_FONT_FILE:
        font = fitz.Font(fontfile=TIMES_FONT_FILE)
    else:
        font = fitz.Font("times-roman")
    text_len_px = font.text_length(str_num, fontsize=font_size)
    
    total_badge_width = triangle_width + rect_width
    
    # Центрирование
    text_x = start_x + (total_badge_width / 2) - (text_len_px / 2) + (triangle_width / 4)
    text_y = center_y + (font_size * 0.35)

    insert_kwargs = {
        'point': (text_x, text_y),
        'text': str_num,
        'fontsize': font_size,
        'color': final_text_color,
        'render_mode': 0,
    }

    if TIMES_FONT_FILE:
        insert_kwargs['fontfile'] = TIMES_FONT_FILE
    else:
        insert_kwargs['fontname'] = 'times-roman'

    page.insert_text(**insert_kwargs)

def create_highlighted_pdf(docx_path, api_response, pc_percent=0.0, output_folder=UPLOAD_FOLDER):
    t_start = time.time()    
    
    base_name = os.path.splitext(os.path.basename(docx_path))[0]
    pdf_filename = base_name + ".pdf"
    pdf_path = os.path.join(output_folder, pdf_filename)  
    
    # Конвертация (если файла нет)
    if not os.path.exists(pdf_path):
        print(f"   -> [INFO] PDF не найден, запускаем конвертацию...")
        cmd = [PATH_TO_LO, '--headless', '--convert-to', 'pdf', '--outdir', output_folder, docx_path]
        subprocess.run(cmd, check=True)

    doc = fitz.open(pdf_path)
    
    # ========================================================
    # ЭТАП 1: ОПРЕДЕЛЯЕМ, КТО ЦИТАТА (СТРОГО ПО ФЛАГУ)
    # ========================================================
    sorted_urls = api_response.get('urls', [])
    citation_source_id = None
    
    # Мы доверяем только флагу, который поставила функция generate_full_report.
    # Никакой самодеятельности и поиска "похожих процентов" здесь быть не должно.
    if sorted_urls:
        for idx, u in enumerate(sorted_urls):
            if u.get('is_citation'):
                citation_source_id = idx + 1
                break
    
    # ========================================================
    # ЭТАП 2: ЗАПОЛНЯЕМ КАРТУ
    # ========================================================
    plagiat_map = {}
    
    for idx, url_item in enumerate(sorted_urls):
        source_id = idx + 1
        try:
            words_str = url_item.get('clean_words_str')
            if words_str is None: words_str = url_item.get('words', '')
            if not words_str: continue
            
            words_indices = parse_compressed_indices(words_str)
            for w_idx in words_indices: 
                plagiat_map[w_idx] = source_id
        except ValueError: continue

    # ========================================================
    # ЭТАП 3: АНАЛИЗ СТРАНИЦ
    # ========================================================
    global_word_list = []
    MARGIN_TOP = 80  
    MARGIN_BOTTOM = 80 

    for page_num, page in enumerate(doc):
        try:
            tables = page.find_tables()
            table_bboxes = [fitz.Rect(tab.bbox) for tab in tables]
        except: table_bboxes = []

        page_height = page.rect.height
        words_on_page = page.get_text("words")
        # Сортировка слов
        words_on_page.sort(key=lambda w: (round(w[1], 1), w[0]))
        
        for w in words_on_page:
            word_rect = fitz.Rect(w[:4])
            is_excluded = False
            
            if word_rect.y1 < MARGIN_TOP or word_rect.y0 > (page_height - MARGIN_BOTTOM):
                is_excluded = True
            if not is_excluded:
                for table_rect in table_bboxes:
                    if table_rect.intersects(word_rect):
                        is_excluded = True
                        break

            global_word_list.append({
                'page': page_num, 
                'rect': word_rect, 
                'text': w[4],
                'skip_highlight': is_excluded,
                'block_no': w[5] 
            })

    # ЦВЕТА
    HIGHLIGHT_COLOR = (253/255, 183/255, 157/255)  # Оранжевый (Плагиат)
    PLAGIAT_BADGE_COLOR = (255/255, 114/255, 58/255)

    CITATION_FILL_COLOR = (239/255, 245/255, 220/255) # Зеленоватый (Цитата)
    CITATION_BADGE_HEX = (178/255, 207/255, 85/255)

    # ========================================================
    # ЭТАП 4: ОТРИСОВКА
    # ========================================================
    current_page_idx = -1
    highlight_shape = None 
    badges_on_page = [] 

    def flush_page_badges(page_idx, badges_list):
        if not badges_list: return
        try:
            p = doc[page_idx]
            for badge_data in badges_list:
                draw_stylish_badge(
                    p, 
                    badge_data['rect'], 
                    badge_data['text'], 
                    bg_color=badge_data['color']
                )
        except Exception as e:
            print(f"Error drawing badges: {e}")

    t_draw = time.time()
    
    for i, word_data in enumerate(global_word_list):
        if word_data['skip_highlight']: continue

        # Получаем ID источника для этого слова
        source_id = plagiat_map.get(i)
        if not source_id: continue 

        # Определяем тип (Цитата или Плагиат)
        # Если citation_source_id == None (потому что ссылка одна), 
        # то is_citation_source всегда будет False.
        is_citation_source = (source_id == citation_source_id)

        # Выбираем цвета
        if is_citation_source:
            fill_color = CITATION_FILL_COLOR
            badge_color = CITATION_BADGE_HEX
            current_badge_id = source_id 
        else:
            fill_color = HIGHLIGHT_COLOR
            badge_color = PLAGIAT_BADGE_COLOR
            current_badge_id = source_id

        # СМЕНА СТРАНИЦЫ
        if word_data['page'] != current_page_idx:
            if highlight_shape:
                highlight_shape.finish(color=None, fill=None)
                highlight_shape.commit() 
                flush_page_badges(current_page_idx, badges_on_page)
                badges_on_page = [] 

            current_page_idx = word_data['page']
            page = doc[current_page_idx]
            highlight_shape = page.new_shape()

        rect = word_data['rect']
        extend_to_next = False
        
        # Логика объединения блоков
        if i + 1 < len(global_word_list):
            next_w = global_word_list[i+1]
            if not next_w['skip_highlight']:
                next_src = plagiat_map.get(i+1)
                
                if next_src == source_id:
                    if next_w['page'] == word_data['page'] and abs(next_w['rect'].y0 - rect.y0) < 5:
                        extend_to_next = True
                        final_rect = fitz.Rect(rect.x0, rect.y0, next_w['rect'].x0, rect.y1)
        
        if not extend_to_next: 
            final_rect = fitz.Rect(rect.x0, rect.y0, rect.x1 + 2, rect.y1)

        # Рисуем
        highlight_shape.draw_rect(final_rect)
        highlight_shape.finish(color=None, fill=fill_color, fill_opacity=0.5) 
        
        # Логика шильдика
        should_badge = False
        if i + 1 < len(global_word_list):
            next_w = global_word_list[i+1]
            next_src = plagiat_map.get(i+1)
            
            if next_src != source_id: 
                should_badge = True
            
            if next_w['page'] != word_data['page'] or next_w['skip_highlight']:
                should_badge = True
        else:
            should_badge = True
        
        if should_badge:
            badges_on_page.append({
                'rect': final_rect,
                'text': current_badge_id,
                'color': badge_color
            })

    if highlight_shape:
        highlight_shape.finish(color=None, fill=None)
        highlight_shape.commit()
        flush_page_badges(current_page_idx, badges_on_page)

    output_filename = base_name + "_highlighted.pdf"
    output_path = os.path.join(output_folder, output_filename)
    doc.save(output_path)
    doc.close()
    
    print(f"   -> Отрисовка аннотаций завершена за {time.time() - t_draw:.2f}s")
    return output_path
   
# --- ОБЛОЖКА ---
def create_cover_pdf(data_context, output_folder=UPLOAD_FOLDER):
    rendered_html = render_template('report_template.html', **data_context)
    
    # 1. Сохраняем HTML для отладки в папку сессии
    debug_html_path = os.path.join(output_folder, "debug_cover.html")
    with open(debug_html_path, "w", encoding="utf-8") as f:
        f.write(rendered_html)
    
    # 2. Формируем путь для PDF обложки
    cover_filename = f"cover_{datetime.datetime.now().timestamp()}.pdf"
    cover_path = os.path.join(output_folder, cover_filename)
    
    options = {
        'page-size': 'A4',
        'margin-top': '10mm',
        'margin-right': '1mm',
        'margin-bottom': '10mm',
        'margin-left': '1mm',
        'encoding': "UTF-8",
        'disable-smart-shrinking': None,
        'zoom': 0.65, 
        'no-outline': None,
        'enable-local-file-access': None
    }
    
    try:
        pdfkit.from_string(rendered_html, cover_path, configuration=config_pdf, options=options, verbose=True)
        return cover_path
    except Exception as e:
        print(f"!!! ОШИБКА WKHTMLTOPDF !!!")
        print(e)
        return None

# --- ГЛАВНАЯ ФУНКЦИЯ СБОРКИ ---
def generate_full_report(filepath, api_data, original_filename, uid):
    """
    Полная генерация отчета внутри папки uploads/{uid}/
    """
    # 1. Определяем рабочую папку для этой сессии
    session_dir = os.path.join(UPLOAD_FOLDER, uid)

    start_total = time.time()
    print(f"--- НАЧАЛО ОБРАБОТКИ: {original_filename} (UID: {uid}) ---", flush=True)
    
    # ==========================================
    # 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
    # ==========================================
    
    # --- А. Распаковка индексов (1-5 -> 1,2,3,4,5) ---
    def parse_compressed_indices(words_str):
        indices = set()
        if not words_str: return indices
        parts = str(words_str).split()
        for part in parts:
            if '-' in part:
                try:
                    start, end = map(int, part.split('-'))
                    indices.update(range(start, end + 1))
                except: pass
            else:
                try:
                    indices.add(int(part))
                except: pass
        return indices

    # --- Б. Фильтр последовательностей ---
    # min_sequence управляет длиной оставляемых кусков
    def filter_isolated_words(indices_set, min_sequence):
        if not indices_set: return []
        sorted_list = sorted(list(indices_set))
        result = []
        current_group = []
        for i, num in enumerate(sorted_list):
            if not current_group:
                current_group.append(num)
                continue
            if num == current_group[-1] + 1:
                current_group.append(num)
            else:
                if len(current_group) >= min_sequence:
                    result.extend(current_group)
                current_group = [num]
        if len(current_group) >= min_sequence:
            result.extend(current_group)
        return result
    
    # --- В. Генератор заголовков (YAKE) ---
    def get_random_titles(text_content, count=50):
        try:
            # === ОПТИМИЗАЦИЯ 1: Обрезаем текст ===
            # Если текст огромный, YAKE зависнет. Берем только первые 50к символов.
            # Этого более чем достаточно для генерации списка фраз.
            limit_chars = 50000 
            if len(text_content) > limit_chars:
                text_content = text_content[:limit_chars]
            
            # === ОПТИМИЗАЦИЯ 2: Уменьшаем n (было 5, стало 3) ===
            # n=3 ищет фразы до 3 слов. Это намного быстрее и памяти ест меньше.
            kw_extractor = yake.KeywordExtractor(
                lan="ru", n=3, dedupLim=0.7, top=count * 3, features=None
            )
            
            keywords = kw_extractor.extract_keywords(text_content)
            
            valid_titles = []
            for kw in keywords:
                phrase = kw[0].strip(" .,-")
                # Проверяем длину (у нас n=3, так что фразы будут короче, но это ок)
                # Можно разрешить и 2 слова, если заголовков мало
                if 2 <= len(phrase.split()) <= 5:
                    valid_titles.append(phrase[0].upper() + phrase[1:])
            
            # Если YAKE нашел мало фраз, добавим заглушки, чтобы не падать
            while len(valid_titles) < count:
                valid_titles.append("Анализ документа")
                
            random.shuffle(valid_titles)
            return valid_titles
        except Exception as e:
            print(f"!!! YAKE ERROR: {e}", flush=True)
            return ["Анализ текста"] * count # Заглушка на случай ошибки

    # ПЛАН Б: "Глупый" но быстрый генератор (используйте, если YAKE виснет)
    # === ВАРИАНТ: УМНЫЙ РАНДОМ (Быстро и качественно) ===
    def get_random_titles_fast(text_content, count=50):
        import re
        
        # 1. Ограничиваем аппетит (берем начало документа, там обычно самая суть)
        # 50 000 символов - это примерно 15-20 страниц.
        limit_chars = 50000
        text_content = text_content[:limit_chars]

        # 2. Разбиваем текст на фразы по точкам, переносам и знакам препинания
        # Это создает список потенциальных предложений
        raw_phrases = re.split(r'[.!?\n;]+', text_content)
        
        valid_titles = []
        
        for phrase in raw_phrases:
            phrase = phrase.strip()
            
            # 3. Жесткий фильтр качества:
            # - Фраза не пустая
            # - Начинается с Большой буквы (русской или латинской)
            # - Внутри нет кучи запятых (максимум 1)
            # - Нет цифр (заголовки с цифрами часто выглядят как мусор из таблиц)
            if not phrase or not phrase[0].isupper():
                continue
                
            # Разбиваем на слова
            words = phrase.split()
            
            # - Длина от 3 до 8 слов (идеально для заголовка)
            if 3 <= len(words) <= 8:
                # Проверка на "чистоту" (нет скобок, слешей, процентов)
                if re.search(r'[0-9<>/(){}\[\]%]', phrase):
                    continue
                    
                # Убираем лишние пробелы и добавляем в список
                clean_phrase = " ".join(words)
                valid_titles.append(clean_phrase)

        # 4. Если нашли мало фраз, добиваем заглушками
        # (убираем дубликаты через set)
        unique_titles = list(set(valid_titles))
        
        if len(unique_titles) < count:
            # Если документ совсем плохой, добавим нейтральных фраз
            unique_titles.extend([
                "Анализ текста документа", 
                "Исследование материалов", 
                "Обзор литературных источников",
                "Теоретические аспекты",
                "Практическая часть работы"
            ])
            
        # Перемешиваем и отдаем
        random.shuffle(unique_titles)
        
        # Возвращаем нужное количество, циклично если не хватает
        result = []
        while len(result) < count:
            result.extend(unique_titles)
            
        return result[:count]

    # --- Г. Загрузка ФИО ---
    people_list = []
    try:
        with open('people.json', 'r', encoding='utf-8') as f:
            people_list = json.load(f)
    except:
        people_list = [{"Фамилия": "Иванов", "Имя": "Сергей", "Отчество": "Андреевич"}]

    def get_random_person_short():
        if not people_list: return "Автор Неизвестен"
        
        # 1. Решаем, сколько авторов взять: 1 или 2
        # Функция min нужна, чтобы не возникло ошибки, если в people.json всего 1 человек
        count = random.randint(1, 2)
        count = min(count, len(people_list))
        
        # 2. Берем случайных уникальных людей
        selected_people = random.sample(people_list, count)
        
        formatted_names = []
        for p in selected_people:
            try:
                name_short = p.get('Имя', '')[0].upper() + "."
                otch_short = p.get('Отчество', '')[0].upper() + "."
                full_name = f"{p.get('Фамилия', '')} {name_short}{otch_short}"
                formatted_names.append(full_name)
            except:
                formatted_names.append(p.get('Фамилия', 'Автор'))
        
        # 3. Объединяем их в одну строку через запятую
        return ", ".join(formatted_names)

    # ==========================================
    # 2. ЛОГИКА ОБРАБОТКИ
    # ==========================================

    # 2.1 Подготовка процентов
    try:
        pc = float(api_data.get('pc', 0))
        unique_val = float(api_data.get('unique', 0))
        if unique_val < pc: po = 0.0
        else: po = unique_val - pc
        pz = 100.0 - po - pc
        if pz < 0: pz = 0
        ai = int(api_data.get('ai', 0))
    except:
        pc, po, pz, ai = 0.0, 0.0, 100.0, 0

    # ==========================================
    # 2.2 Сортировка и очистка
    # ==========================================
    urls = api_data.get('urls', [])
    target_pc = float(pc)
    source_char_count = safe_percent(api_data.get('char_count', 0), 0.0)

    # 1. Функция для выделения САМОГО ДЛИННОГО непрерывного куска
    def get_longest_block(indices_set):
        if not indices_set: return []
        sorted_list = sorted(list(indices_set))
        best_block = []
        current_block = []
        for num in sorted_list:
            if not current_block:
                current_block.append(num)
            elif num == current_block[-1] + 1:
                current_block.append(num)
            else:
                if len(current_block) > len(best_block):
                    best_block = current_block
                current_block = [num]
        if len(current_block) > len(best_block):
            best_block = current_block
        return best_block

    # -------------------------------------------------------------
    # ЛОГИКА НАЗНАЧЕНИЯ ЦИТАТЫ
    # -------------------------------------------------------------
    
    # Сначала СБРАСЫВАЕМ флаги у всех, чтобы начать с чистого листа.
    # Это лечит ситуацию, если PHP прислал is_citation=True для единственной ссылки.
    for u in urls:
        u['is_citation'] = False

    citation_idx = -1

    # Включаем логику поиска цитаты ТОЛЬКО если ссылок МНОГО (> 1)
    if len(urls) > 1 and target_pc > 0 and source_char_count > 5000:
        
        # А. Ищем, может PHP прислал флаг (но мы его стерли выше, так что смотрим raw_data если надо, 
        # но проще искать заново или сохранить состояние).
        # Давайте искать "лучшего кандидата" заново, это надежнее.
        
        best_diff = 1000.0
        
        for idx, u in enumerate(urls):
            try:
                p_val = float(u.get('plagiat', 0))
                diff = abs(p_val - target_pc)
                if diff < best_diff:
                    best_diff = diff
                    citation_idx = idx
            except: pass

    # Если нашли кандидата И ссылок реально больше 1 -> Назначаем
    if citation_idx != -1:
        urls[citation_idx]['is_citation'] = True
        urls[citation_idx]['plagiat'] = target_pc

    # -------------------------------------------------------------
    # ОБРАБОТКА СЛОВ (ОЧИСТКА И ФИЛЬТРАЦИЯ)
    # -------------------------------------------------------------
    
    # Сортируем (цитата, если есть, встанет на место по проценту)
    urls.sort(key=lambda x: float(x.get('plagiat', 0)), reverse=True)

    used_words_global = set() 
    cleaned_urls = []
    
    citation_processed = False 

    for u in urls:
        # Проверяем флаг (он теперь гарантированно False, если ссылка одна)
        is_target_citation = u.get('is_citation', False)
        
        # Защита от дублей (если вдруг логика выше дала сбой, хотя не должна)
        if is_target_citation and citation_processed:
            is_target_citation = False
            u['is_citation'] = False

        raw_words = u.get('words', '')
        current_words_set = parse_compressed_indices(raw_words)
        
        if not current_words_set:
            u['clean_words_str'] = ""
            cleaned_urls.append(u)
            continue
        
        if is_target_citation:
            # === ЦИТИРОВАНИЕ (Зеленый, один кусок) ===
            citation_processed = True
            longest_block = get_longest_block(current_words_set)
            final_words_list = longest_block
        else:
            # === ПЛАГИАТ (Оранжевый, много кусков) ===
            # Если ссылок всего 1, мы 100% попадем сюда.
            current_words_set = current_words_set - used_words_global
            final_words_list = sorted(current_words_set)

        if not final_words_list:
            u['clean_words_str'] = "" 
        else:
            u['clean_words_str'] = " ".join(map(str, final_words_list))
            used_words_global.update(final_words_list)

        cleaned_urls.append(u)

    cleaned_urls.sort(key=lambda x: float(x.get('plagiat', 0)), reverse=True)
    api_data['urls'] = cleaned_urls
    
    print(f"[{time.time() - start_total:.2f}s] Данные подготовлены, начинаем PDF...")    

    # 2.3 Генерация PDF
    t_pdf_start = time.time()
    highlighted_pdf_path = create_highlighted_pdf(filepath, api_data, pc_percent=pc, output_folder=session_dir)
    print(f"[{time.time() - start_total:.2f}s] PDF сгенерирован (заняло {time.time() - t_pdf_start:.2f}s)")
    
    print(f"   -> [DEBUG] Начало извлечения текста для заголовков...", flush=True)
    # 2.4 Анализ текста для заголовков
    full_text_content = ""
    try:
        doc_stats = fitz.open(highlighted_pdf_path)
        page_count = doc_stats.page_count
        for page in doc_stats: full_text_content += page.get_text() + " "
        doc_stats.close()
        print(f"   -> [DEBUG] Текст извлечен. Длина: {len(full_text_content)} символов.", flush=True)
    except:
        print(f"   -> [ERROR] Ошибка чтения PDF: {e}", flush=True)
        page_count = 0
        full_text_content = ""

    # --- ОТЛАДКА: ЭТАП 2 (YAKE) ---
    print(f"   -> [DEBUG] Запуск YAKE (генерация заголовков)...", flush=True)
    try:
        generated_titles = get_random_titles_fast(full_text_content, count=len(cleaned_urls) + 10)
        print(f"   -> [DEBUG] YAKE завершил работу.", flush=True)
    except Exception as e:
        print(f"   -> [ERROR] Ошибка YAKE: {e}", flush=True)
        generated_titles = []
        
    # --- ОТЛАДКА: ЭТАП 3 (График) ---
    print(f"   -> [DEBUG] Рисуем график Matplotlib...", flush=True)
    try:
        chart_base64 = generate_pie_chart_base64(pz, po, pc)
        print(f"   -> [DEBUG] График готов.", flush=True)
    except Exception as e:
        print(f"   -> [ERROR] Ошибка Matplotlib: {e}", flush=True)
        chart_base64 = "" # Пустая картинка, чтобы не упало        

    
    # ==========================================
    # 3. СБОРКА HTML
    # ==========================================
    urls_html = ""
    months_ru = ['янв', 'фев', 'мар', 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек']
    # чтобы 1 раз выводить
    # ring_vuz_used = False 
    
    print(f"   -> [DEBUG] Подготовка данных для обложки...", flush=True)

    for i, u in enumerate(cleaned_urls):
        plagiat_percent = float(u.get('plagiat', 0))
        
        d_back = random.randint(100, 1000)
        d_obj = datetime.date.today() - timedelta(days=d_back)
        f_date = f"{d_obj.day} {months_ru[d_obj.month - 1]} {d_obj.year}"

        if u.get('is_citation'):
            urls_html += f"""
            <tr>
                <td><div class="num" data-link="cit">[{i+1}]</div></td>
                <td class="percent-col"><div class="percent plagiat cw">{plagiat_percent}%</div></td>
                <td class="percent-by-text-col"><div class="">0%</div></td>
                <td class="name-col">
                    <div class="name"></div>
                    <div class="link"><a href="#" style="cursor: default; text-decoration: none; color: inherit;">Цитирование</a></div>
                </td>
                <td class="date-col"><div class="date">{f_date}</div></td>
                <td class="collections-col"><div class="collections">Цитирование</div></td>
                <td class="comment-col"><div class=""></div></td>
            </tr>"""
            continue

        # Обычные источники
        full_url = u.get('url', '')
        yake_title = generated_titles.pop(0) if generated_titles else "Анализ текста"
        random_person = get_random_person_short()

        domain_raw = full_url
        try:
            parsed = urlparse(full_url)
            domain_raw = parsed.netloc
            if not domain_raw: domain_raw = full_url
        except: pass
        domain_lower = domain_raw.lower()

        final_module_name = "Интернет"
        final_domain_text = domain_raw
        final_title_text = yake_title

        media_keywords = ['gazeta', '/news/', '/novosti/', 'rg.ru', 'press', 'ria.ru', 'tass.ru', 'kommersant.ru', 'rbc.ru', 'lenta.ru', 'iz.ru', 'mk.ru', 'kp.ru', 'vedomosti.ru', 'life.ru', 'rt.com', 'fontanka.ru', 'news.ru']
        patent_keywords = ['patent', 'fips', 'rupto', 'findpatent', 'bankpatentov', 'patentdb']

        if 'garant' in domain_lower:
            opts = ["СПС ГАРАНТ:нормативно-правовая документация", "Перефразирования по СПС ГАРАНТ: аналитика", "СПС ГАРАНТ:аналитика"]
            final_module_name = random.choice(opts)
            final_domain_text = "http://ivo.garant.ru"
            final_title_text = random.choice([yake_title, random_person])
        elif 'wikipedia' in domain_lower or 'ruwiki' in domain_lower:
            final_module_name = "Рувики"
            final_title_text = random.choice([yake_title, random_person])
        elif any(k in full_url.lower() for k in media_keywords) and 'wordpress' not in full_url.lower():
            final_module_name = "СМИ России и СНГ"
            final_title_text = random.choice([yake_title, random_person])
        elif any(k in full_url.lower() for k in patent_keywords):
            final_module_name = "Патенты СССР, РФ. СНГ"
            final_title_text = random.choice([yake_title, random_person])
        else:
            # === ИЗМЕНЕНИЕ: ЛОГИКА 35% ===
            # Пул вариантов БЕЗ inet_plus
            pool = ["rephrase_inet", "ebs", "elibrary", "elibrary_rephrase", "elibrary_combo", "rgb", "rgb_rephrase", "vuz_ring"]
            
            # if ring_vuz_used and "vuz_ring" in pool: 
            #     pool.remove("vuz_ring")

            # Генерируем случайное число 0..1
            if random.random() < 0.35:
                # 35% шанс
                choice = "inet_plus"
            else:
                # 65% шанс
                choice = random.choice(pool)
            # =============================
            
            if choice == "inet_plus":
                final_module_name = "Интернет плюс"
                final_title_text = random.choice([yake_title, full_url])
            elif choice == "rephrase_inet":
                final_module_name = "Перефразированные заимствования по коллекции Интернет в русском сегменте"
                final_title_text = random.choice([yake_title, full_url])
            elif choice == "ebs":
                final_module_name = "Сводная коллекция ЭБС"
                final_title_text = random.choice([yake_title, random_person])
                final_domain_text = full_url = random.choice(["https://book.ru", "http://ibooks.ru", "http://e.lanbook.com"])
            elif choice == "elibrary":
                final_module_name = "Публикации eLIBRARY"
                final_title_text = random.choice([yake_title, random_person])
                final_domain_text = full_url = "http://elibrary.ru"
            elif choice == "elibrary_rephrase":
                final_module_name = "Публикации eLIBRARY(переводы и перефразирования)"
                final_title_text = random.choice([yake_title, random_person])
                final_domain_text = full_url = "http://elibrary.ru"
            elif choice == "elibrary_combo":
                final_module_name = "Публикации eLIBRARY + Публикации eLIBRARY(переводы и перефразирования)"
                final_title_text = random.choice([yake_title, random_person])
                final_domain_text = full_url = "http://elibrary.ru"
            elif choice == "rgb":
                final_module_name = "Публикации РГБ"
                final_title_text = random.choice([yake_title, random_person])
                final_domain_text = full_url = "http://dlib.rsl.ru"
            elif choice == "rgb_rephrase":
                final_module_name = "Публикации РГБ (переводы и перефразирования)"
                final_title_text = random.choice([yake_title, random_person])
                final_domain_text = full_url = "http://dlib.rsl.ru"
            elif choice == "vuz_ring":
                final_module_name = "Кольцо вузов"
                final_title_text = random.choice([yake_title, random_person])
                final_domain_text = ""
                # ring_vuz_used = True

        if plagiat_percent < 1:
            comment_html = '<div class="comment">Источник исключен. Причина:<br> Маленький процент пересечения.</div>'
            comment_class = ' class="deleted"'
            percent_by_text = 0
        else:
            comment_html = '<div class="comment"></div>'
            comment_class = ''
            # Рандом от (50% от plagiat_percent) до (plagiat_percent)
            val_min = plagiat_percent * 0.5
            val_max = plagiat_percent
            percent_by_text = round(random.uniform(val_min, val_max), 2)
        
        urls_html += f"""
        <tr{comment_class}>
            <td><div class="num" data-link="{i+1}">[{i+1}]</div></td>
            <td class="percent-col"><div class="percent plagiat">{plagiat_percent}%</div></td>
            <td class="percent-by-text-col"><div class="">{percent_by_text}%</div></td>
            <td class="name-col">
                <div class="name">{final_title_text}</div>
                <div class="link"><a href="{full_url}" target="_blank">{final_domain_text}</a></div>
            </td>
            <td class="date-col"><div class="date">{f_date}</div></td>
            <td class="collections-col"><div class="collections">{final_module_name}</div></td>
            <td class="comment-col">{comment_html}</td>
        </tr>"""

    logo_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAARwAAAA8CAYAAAC2E3rAAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAyJpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+/Poz1ua4AAAlOSURBVHja7Z0LcFvVGYbvSyrSjh990KE8Cg10U1qY0hfogHR4ydB2yqO0IcO0lJbQls50oC1hClPanjJ0oO00dEo7ZWg7JbQlhOmLhxbS9IWWFyiFHx3KozR90JY2fd/f6l51dbW8tGvJ0r2zM5qR7r3n/Of89/z3P/9d2+VyuQREIq12oQmIREAiIBEBiYBEQCIgEQGJCEgEJAISAYkISERAIiARkAhIREAiAhIBiYBEQCIgEQGJCEgEJAISAYkISERAIiARkAhIREAiAhIBiYBEQCIgEQGJCExm07d7h8PhN/0/FAr5+Z/f79e3w+HwRj0ej8fv9/v17XA47NfXbrd7Nf/+691ut1vfttvtft1ut9v07XA43Pz7gL7tdrudf/fr2263O/X1brfboa/19/LzT/p2OBxWfTu32+3g3/fr2263W/Vt/s2v/753u91O+7b65369W1/j93v0bbfb7db/f/y+X992u91hfr+j1+v12O12l9/v9/A7A/x+r77t4ff9+rbH4/Hw+936tsfjcendbrfboW97PB633+/X1/h9v77tcrkcerfb7Xbp2+Fw2K13u91uh77tcrmc/Du//7fe7Xa7nfq2y+VyuVwup97tdrud+rbL5XLq610ul8vldDpd+rbT6XTq691ut9vldDpd+rbT6XTq691ut9vldDrd+rbT6XTp610ul8vjdDrd+rbT6XTp610ul8vjdDrd+rbT6XTp610ul8vjdDrd+rbT6XTp610ul8vjdDrd+rbT6XTp610ul8vjdDrd+rbT6XTp610ul8vjdDrd+rbT6XTp610ul8vjdDrd+rbT6XTp6+1ut8fldLr1bafT6XKpF3g8Hg+XlXq30+l08e9+1c/v6PV6PfzfXn2b3+/Stz0ej4ff79K3PR6Pm9/v1Lc9Ho/bpXe73W6Xvu3xeNz8vt/A3/fru91ut1v1bbfe7Xa73fp2OBx2691ut9vFv/v1bbfb7eLf/fq22+128u9+fTvMv/v13W6326Vvh8Nh17vdbpeb3/fp2263283v9+nbbrfbzZ/79W2Xy+Xk9/36tsvlcvD7fn3b5XI59W6Xy+V26dsul8vN7/v1bZfL5eL3/fq2y+Vy8ft+fdvlcrl5d7/e7Xa73fq22+128/s+fdvtdrv5fb++7XK5nPy+X992uVwOft+vb7tcLqfe7XK53C592+VyufRut9vtdunbLpfLxe/79W2Xy+Xi9/36tsvlcvPufn3b5XI59W6Xy+V26dsul8vF7/v1bZfL5eL3/fq2y+Vy8+5+fdvlcjn1bpfL5Xbp2y6Xy8Xv+/Vtl8vl4vf9+rbL5XLz7n592+VyOfVul8vldunbLpfLxe/79W2Xy+Xi9/36tsvlcvPufn3b5XI59W6Xy+V26dsul8vF7/v1bZfL5eL3/fq2y+Vy8+5+fdvlcjn1bpfL5Xbp2y6Xy8Xv+/Vtl8vl4vf9+rbL5XLz7n592+VyOfVul8vldunbLpfLxe/79W2Xy+Xi9/36tsvlcvPufn3b5XI59W6Xy+V26dsul8vF7/v1bZfL5eL3/fq2y+Vy8+5+fdvlcjn1bpfL5Xbp2y6Xy8Xv+/Vtl8vl4vf9+rbL5XLz7n592+VyOfVul8vldunbLpfLxe/79W2Xy+Xi9/36tsvlcvPufn3b5XI59W6Xy+V26dsul8vF7/v1bZfL5eL3/fq2y+Vy8+5+fdvlcjn1bpfL5Xbp2y6Xy8Xv+/Vtl8vl4vf9+rbL5XLz7n592+VyOfVul8vldunbLpfLxe/79W2Xy+Xi9/36tsvlcvPufn3b5XI59W6Xy+V26dsul8vF7/v1bZfL5eL3/fq2y+Vy8+5+fdvlcjn1bpfL5Xbp2y6Xy8Xv+/Vtl8vl4vf9+rbL5XLz7n592+VyOfVul8vldunbLpfLxe/79W2Xy+Xi9/36tsvlcvPufn3b5XI59W6Xy+V26dsul8vF7/v1bZfL5eL3/fq2y+Vy8+5+fdvlcjn1bpfL5Xbp2y6Xy8Xv+/Vtl8vl4vf9+rbL5XLz7n592+VyOfVul8vldunbLpfLxe/79W2Xy+Xi9/36tsvlcvPufn3b5XI59W6Xy+V26dsul8vF7/v1bZfL5eL3/fq2y+Vy8+5+fdvlcjn1bpfL5Xbp2y6Xy8Xv+/Vtl8vl4vf9+rbL5XLz7n592+VyOfVul8vldunbLpfLxe/79W2Xy+Xi9/36tsvlcvPufn3b5XI59W6Xy+V26dsul8vF7/v1bZfL5eL3/fq2y+Vy8+5+fdvlcjn1bpfL5Xbp2y6Xy8Xv+/Vtl8vl4vf9+rbL5XLz7n592+VyOfVul8vldunbLpfLxe/79W2Xy+Xi9/36tsvlcvPufn3b5XI59W6Xy+V26dsul8vF7/v1bZfL5eL3/fq2y+Vy8+5+fdvlcjn1bpfL5Xbp2y6Xy8Xv+/Vtl8vl4vf9+rbL5XLz7n592+VyOfVul8vldunbLpfLxe/79W2Xy+Xi9/36tsvlcvPufn3b5XI59W6Xy+V26dsul8vF7/v1bZfL5eL3/fq2y+Vy8+5+fdvlcjn1bpfL5Xbp2y6Xy8Xv+/Vtl8vl4vf9+rbL5XLz7n592+VyOfVul8vldunbLpfLxe/79W2Xy+Xi9/36tsvlcvPufn3b5XI59W6Xy+V26dsul8vF7/v1bZfL5eL3/fq2y+Vy8+5+fdvlcjn1bpfL5Xbp2y6Xy8Xv+/Vtl8vl4vf9+rbL5XLz7n592+VyOfVul8vldunbLpfLxe/79W2Xy+Xi9/36tsvlcvPufn3b5XI59W6Xy+V26dsul8vF7/v1bZfL5eL3/fq2y+Vy8+5+fdvlcjn1bpfL5Xbp2y6Xy8Xv+/Vtl8vl4vf9+rbL5XLz7n592+VyOfVul8vldunbLpfLxe/79W2Xy+Xi9/36tsvlcvPufn3b5XI59W6Xy+V26dsul8vF7/v1bZfL5eL3/fq2y+Vy8+5+fdvlcjn1bpfL5Xbp2y6Xy8Xv+/Vtl8vl4vf9+rbL5XLz7n592+VyOfVul8vldunbLpfLxe/79W2Xy+Xi9/36tsvlcvPufn3b5XI59W6Xy+V26dt/CjAA0jA1/6966/cAAAAASUVORK5CYII="

    # === МЕТКА ДЛЯ СКРЫТИЯ ШАПКИ ===
    # True  = Показать шапку
    # False = Скрыть шапку
    SHOW_HEADER_FLAG = False 
    # ===============================

    context = {
        "show_header": SHOW_HEADER_FLAG,
        "autor": api_data.get('author', "Илюшин Андрей Игоревич"),
        "checker": api_data.get('checker', "Илюшин Андрей Игоревич"),
        "doc_name": original_filename,
        "vuz": api_data.get('organization', "КГУ"),
        "pz": round(float(pz), 2), 
        "po": round(float(po), 2), 
        "pc": round(float(pc), 2), 
        "ai": int(ai),
        "number_document": " " + str(random.randint(2000, 50000)),
        # "number_document": " " + str(int(datetime.datetime.now().timestamp())),
        "check_date": datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        "page_count": page_count,
        "char_count": api_data.get('char_count', 0), 
        "word_count": api_data.get('word_count', 0), 
        "sent_count": api_data.get('sent_count', 0), 
        "user": "admin",
        "type_report": "Полный",
        "chart_image": chart_base64,
        "urls_list": urls_html,
        "logo_base64": logo_b64
    }

    print(f"   -> [DEBUG] Запуск create_cover_pdf (wkhtmltopdf)...", flush=True)
    cover_pdf_path = create_cover_pdf(context, output_folder=session_dir)
    print(f"   -> [DEBUG] Обложка создана: {cover_pdf_path}", flush=True)
    
    name_without_ext = os.path.splitext(original_filename)[0]
    final_filename = f"{name_without_ext}.pdf"   

    final_path = os.path.join(session_dir, final_filename)

    final_doc = fitz.open()
    if cover_pdf_path and os.path.exists(cover_pdf_path):
        cover_doc = fitz.open(cover_pdf_path)
        final_doc.insert_pdf(cover_doc)
        cover_doc.close()
    
    content_doc = fitz.open(highlighted_pdf_path)
    final_doc.insert_pdf(content_doc)
    content_doc.close()
    
    final_doc.save(final_path)
    final_doc.close()
    print(f"--- КОНЕЦ. ПОЛНОЕ ВРЕМЯ: {time.time() - start_total:.2f}s ---")
    
    return final_filename

# --- ROUTES ---

@app.route('/api/highlight', methods=['POST'])
def api_highlight():
    filepath = None
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # === НОВАЯ ЛОГИКА ===
        # 1. Генерируем UID
        uid = str(uuid.uuid4())
        
        # 2. Создаем папку uploads/{uid}
        session_dir = os.path.join(UPLOAD_FOLDER, uid)
        os.makedirs(session_dir, exist_ok=True)
        
        # 3. Сохраняем файл туда
        # Очищаем имя файла от греха (secure_filename желательно, но можно и так)
        filename = file.filename
        filepath = os.path.join(session_dir, filename)
        file.save(filepath)

        # 4. Получаем JSON (код тот же)
        api_data = None
        if 'json_file' in request.files:
            api_data = json.load(request.files['json_file'])
        elif 'json_data' in request.form:
            try:
                api_data = json.loads(request.form['json_data'])
            except ValueError:
                return jsonify({"error": "Invalid JSON"}), 400
        
        if api_data is None:
            return jsonify({"error": "No json provided"}), 400

        # 5. Запускаем генерацию, передавая UID
        final_pdf_name = generate_full_report(filepath, api_data, filename, uid)

        # Сохраняем ИТОГОВЫЕ API-данные (после очистки/назначения цитирования)
        # для HTML-превью /report и /preview-html
        with open(os.path.join(session_dir, 'api_data.json'), 'w', encoding='utf-8') as f:
            json.dump(api_data, f, ensure_ascii=False)
        
        # 6. Формируем НОВУЮ ссылку на скачивание
        # Теперь передаем и filename, и uid
        download_url = url_for('download_result', uid=uid, filename=final_pdf_name, _external=True)
        preview_url = url_for('preview_result', uid=uid, filename=final_pdf_name, _external=True)
        preview_html_legacy_url = url_for('preview_html_result', uid=uid, filename=final_pdf_name, _external=True)
        preview_html_url = url_for('preview_html_accurate_result', uid=uid, filename=final_pdf_name, _external=True)
        preview_html_accurate_url = preview_html_url
        preview_html_accurate_alt_url = url_for('preview_html_accurate_alt_result', uid=uid, filename=final_pdf_name, _external=True)
        
        return jsonify({
            "status": "success", 
            "download_url": download_url,
            "preview_url": preview_url,
            "preview_html_url": preview_html_url,
            "preview_html_accurate_url": preview_html_accurate_url,
            "preview_html_accurate_alt_url": preview_html_accurate_alt_url,
            "preview_html_legacy_url": preview_html_legacy_url,
            "uid": uid
        })

    except Exception as e:
        print(f"API Error: {e}") 
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
        
    finally:
        # Удаляем исходник .docx, чтобы не занимал место, но оставляем PDF
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except: pass

# @app.route('/', methods=['GET', 'POST'])
def _index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file: return 'Нет файла'
        
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # --- ТЕСТОВЫЕ ДАННЫЕ ДЛЯ ФРОНТА ---
        mock_data = {
            "unique": 15.5, # 15.5% Заимствований
            "author": "Илюшин Андрей Игоревич",
            "checker": "Илюшин Андрей Игоревич",
            "organization": "ЧАСТНОЕ ОБРАЗОВАТЕЛЬНОЕ УЧРЕЖДЕНИЕ 'ВЫСШЕГО ОБРАЗОВАНИЯ АКАДЕМИЯ УПРАВЛЕНИЯ И ПРОИЗВОДСТВА'",
            "urls": [
                {
                    "url": "https://ru.wikipedia.org/wiki/Python",
                    "plagiat": 10,
                    # Проверь, чтобы эти слова реально были в твоем доке
                    "words": "0 1 2 3 4 5 6 7 8 9 10" 
                },
                {
                    "url": "https://habr.com/ru/post/123456/",
                    "plagiat": 5.5,
                    "words": "20 21 22 25 26 27"
                },
                {
                    "url": "https://test.com/ru/posti/123456/",
                    "plagiat": 0.8,
                    "words": "220 221 222 223 224 225 226 227"
                }                
            ]
        }

        try:
            # Запускаем полную генерацию
            final_pdf = generate_full_report(filepath, mock_data, filename)
            return render_template('view_pdf.html', pdf_file=final_pdf)
        except Exception as e:
            import traceback
            traceback.print_exc() # Пишем ошибку в консоль сервера
            return f"Ошибка генерации отчета: {e}"

    return render_template('upload.html')

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
    # return send_file(
        # os.path.join(UPLOAD_FOLDER, filename),
        # as_attachment=True, 
        # download_name=filename 
    # ) 

@app.route('/preview-html/<uid>/<filename>')
def preview_html_result(uid, filename):
    """
    Просмотр итогового HTML с подсветкой (не обложка),
    используя уже существующий рендер report_view.
    """
    directory = os.path.join(UPLOAD_FOLDER, uid)
    if not os.path.exists(directory):
        return "File not found (bad uid)", 404

    # Это именно HTML-представление результата выделений.
    return redirect(url_for('report_view', uid=uid, filename=filename))


@app.route('/preview/<uid>/<filename>')
def preview_result(uid, filename):
    """
    Просмотр итогового PDF в браузере (тот же файл, что уходит на выгрузку).
    Ничего не меняет в текущем механизме скачивания.
    """
    directory = os.path.join(UPLOAD_FOLDER, uid)

    if not os.path.exists(directory):
        return "File not found (bad uid)", 404

    file_path = os.path.join(directory, filename)
    if not os.path.exists(file_path):
        return "File not found", 404

    return send_file(
        file_path,
        as_attachment=False,
        download_name=filename,
        mimetype='application/pdf'
    )


@app.route('/download/<uid>/<filename>')
def download_result(uid, filename):
    """
    Скачивание файла из конкретной папки /uploads/{uid}/
    """
    # Путь к директории этого клиента
    directory = os.path.join(UPLOAD_FOLDER, uid)
    
    # Проверка на всякий случай, чтобы не вышли за пределы папки (хотя uuid безопасен)
    if not os.path.exists(directory):
        return "File not found (bad uid)", 404

    return send_file(
        os.path.join(directory, filename),
        as_attachment=True,
        download_name=filename
    )    
    
    
@app.route('/')
def index():
    """Главная страница с формой загрузки"""
    return render_template('upload_form.html')


@app.route('/upload', methods=['POST'])
def upload():
    """Обработка загрузки и редирект на отчет"""
    file = request.files.get('file')
    if not file:
        return 'No file', 400
    
    uid = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_FOLDER, uid)
    os.makedirs(session_dir, exist_ok=True)
    
    filename = file.filename
    filepath = os.path.join(session_dir, filename)
    file.save(filepath)
    
    # Получаем JSON
    api_data = {}
    if 'json_data' in request.form:
        try:
            api_data = json.loads(request.form['json_data'])
        except:
            pass
    
    # Тестовые данные
    if not api_data:
        api_data = {
            "unique": 85.5,
            "pc": 5.0,
            "author": "Тест",
            "urls": [
                {"url": "https://example.com", "plagiat": 10, "words": "10-30"},
                {"url": "https://test.com", "plagiat": 5, "words": "50-60", "is_citation": True}
            ]
        }
    
    # Сохраняем api_data
    with open(os.path.join(session_dir, 'api_data.json'), 'w', encoding='utf-8') as f:
        json.dump(api_data, f, ensure_ascii=False)
    
    # Генерируем PDF
    final_pdf = generate_full_report(filepath, api_data, filename, uid)
    
    # Редирект на просмотр
    return redirect(url_for('report_view', uid=uid, filename=final_pdf))


@app.route('/preview-html-accurate-alt/<uid>/<filename>')
def preview_html_accurate_alt_result(uid, filename):
    """Отдельный ALT HTML-превью для тестового метода text alignment."""
    directory = os.path.join(UPLOAD_FOLDER, uid)
    if not os.path.exists(directory):
        return "File not found (bad uid)", 404
    return redirect(url_for('report_view_accurate_alt', uid=uid, filename=filename))


@app.route('/preview-html-accurate/<uid>/<filename>')
def preview_html_accurate_result(uid, filename):
    """Отдельный HTML-превью с альтернативной логикой позиционирования/индексации."""
    directory = os.path.join(UPLOAD_FOLDER, uid)
    if not os.path.exists(directory):
        return "File not found (bad uid)", 404
    return redirect(url_for('report_view_accurate', uid=uid, filename=filename))


@app.route('/report-accurate/<uid>/<filename>')
def report_view_accurate(uid, filename):
    """Production-метод (Turnitin-like): anchors + SequenceMatcher + смещение."""
    json_path = os.path.join(UPLOAD_FOLDER, uid, 'api_data.json')
    name_without_ext = os.path.splitext(filename)[0]
    pdf_path = os.path.join(UPLOAD_FOLDER, uid, f"{name_without_ext}_highlighted.pdf")

    if not os.path.exists(pdf_path):
        return "PDF not found", 404

    api_data = {}
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            api_data = json.load(f)

    unique = get_api_metric(api_data, ['unique', 'originality', 'original', 'orig'], default=0.0)
    pc = get_api_metric(api_data, ['pc', 'citation', 'cit'], default=0.0)
    plag = safe_percent(api_data.get('plag', api_data.get('plagiat', 100 - unique - pc)), 100 - unique - pc)

    source_to_meta = {}
    urls = api_data.get('urls', [])
    citation_id = None
    for idx, u in enumerate(urls):
        sid = idx + 1
        if u.get('is_citation'):
            citation_id = sid
        words_str = u.get('clean_words_str', u.get('words', ''))
        for widx in parse_compressed_indices(words_str):
            source_to_meta[widx] = {'sid': sid, 'cit': sid == citation_id}

    doc = fitz.open(pdf_path)

    # source stream: все слова; target stream: слова после фильтра полей/таблиц
    full_stream = extract_pdf_words_with_meta(doc, skip_margins_tables=False)
    filtered_stream = [w for w in extract_pdf_words_with_meta(doc, skip_margins_tables=True) if not w['skip']]

    full_tokens = [w['norm'] for w in full_stream]
    filt_tokens = [w['norm'] for w in filtered_stream]

    src_to_tgt = align_streams_with_anchors(full_tokens, filt_tokens, ngram=5)
    page_blocks_map = build_highlight_blocks_from_mapping(src_to_tgt, source_to_meta, filtered_stream)

    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("html")
        pages.append({
            'text': text,
            'blocks': page_blocks_map.get(page_num, []),
            'w': page.rect.width,
            'h': page.rect.height,
        })

    doc.close()

    return render_template('report_exact.html',
                         filename=filename,
                         orig=round(unique, 1),
                         plag=round(plag, 1),
                         cit=round(pc, 1),
                         total=len(pages),
                         pages=pages)


@app.route('/report-accurate-alt/<uid>/<filename>')
def report_view_accurate_alt(uid, filename):
    """ALT-метод: text alignment через нормализованный поток слов (для тестов)."""
    json_path = os.path.join(UPLOAD_FOLDER, uid, 'api_data.json')
    name_without_ext = os.path.splitext(filename)[0]
    pdf_path = os.path.join(UPLOAD_FOLDER, uid, f"{name_without_ext}_highlighted.pdf")

    if not os.path.exists(pdf_path):
        return "PDF not found", 404

    api_data = {}
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            api_data = json.load(f)

    unique = get_api_metric(api_data, ['unique', 'originality', 'original', 'orig'], default=0.0)
    pc = get_api_metric(api_data, ['pc', 'citation', 'cit'], default=0.0)
    plag = safe_percent(api_data.get('plag', api_data.get('plagiat', 100 - unique - pc)), 100 - unique - pc)

    source_to_meta = {}
    urls = api_data.get('urls', [])
    citation_id = None
    for idx, u in enumerate(urls):
        sid = idx + 1
        if u.get('is_citation'):
            citation_id = sid
        words_str = u.get('clean_words_str', u.get('words', ''))
        for widx in parse_compressed_indices(words_str):
            source_to_meta[widx] = {'sid': sid, 'cit': sid == citation_id}

    doc = fitz.open(pdf_path)

    # PDF поток для координат (без фильтра для максимально плотного соответствия)
    pdf_stream = extract_pdf_words_with_meta(doc, skip_margins_tables=False)
    pdf_tokens = [w['norm'] for w in pdf_stream]

    # API-текстовый поток (если есть), иначе fallback на PDF поток
    api_text = api_data.get('text') or api_data.get('document_text') or api_data.get('full_text') or ''
    if api_text:
        api_tokens = [normalize_word_token(t) for t in re.split(r'\s+', str(api_text))]
        api_tokens = [t for t in api_tokens if t]
    else:
        api_tokens = pdf_tokens

    src_to_tgt = align_streams_with_anchors(api_tokens, pdf_tokens, ngram=4)
    page_blocks_map = build_highlight_blocks_from_mapping(src_to_tgt, source_to_meta, pdf_stream)

    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("html")
        pages.append({
            'text': text,
            'blocks': page_blocks_map.get(page_num, []),
            'w': page.rect.width,
            'h': page.rect.height,
        })

    doc.close()

    return render_template('report_exact.html',
                         filename=filename,
                         orig=round(unique, 1),
                         plag=round(plag, 1),
                         cit=round(pc, 1),
                         total=len(pages),
                         pages=pages)


@app.route('/report/<uid>/<filename>')
def report_view(uid, filename):
    """Просмотр отчета как один длинный лист"""
    json_path = os.path.join(UPLOAD_FOLDER, uid, 'api_data.json')
    name_without_ext = os.path.splitext(filename)[0]
    pdf_path = os.path.join(UPLOAD_FOLDER, uid, f"{name_without_ext}_highlighted.pdf")
    
    if not os.path.exists(pdf_path):
        return "PDF not found", 404
    
    # Загружаем данные
    api_data = {}
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            api_data = json.load(f)
    
    unique = get_api_metric(api_data, ['unique', 'originality', 'original', 'orig'], default=0.0)
    pc = get_api_metric(api_data, ['pc', 'citation', 'cit'], default=0.0)

    # Если плагиат/заимствования не пришли отдельным полем, считаем остатком
    plag_raw = None
    for key in ['plag', 'plagiat', 'borrowed']:
        if key in api_data and api_data.get(key) is not None:
            plag_raw = api_data.get(key)
            break

    if plag_raw is None:
        plag = 100 - unique - pc
    else:
        plag = safe_percent(plag_raw, 0.0)
    
    # Карта слов к источникам
    word_to_source = {}
    urls = api_data.get('urls', [])
    citation_id = None
    
    for idx, u in enumerate(urls):
        sid = idx + 1
        if u.get('is_citation'):
            citation_id = sid
        words_str = u.get('clean_words_str', u.get('words', ''))
        for widx in parse_compressed_indices(words_str):
            # Сохраняем как есть: индекс источника для слова
            word_to_source[widx] = {'sid': sid, 'cit': sid == citation_id}
    
    # Обрабатываем PDF (версия с подсветкой уже содержит те же текстовые координаты)
    doc = fitz.open(pdf_path)
    start_page = 0

    # Собираем список слов по той же логике, что при генерации PDF-окраса
    global_word_list = []
    MARGIN_TOP = 80
    MARGIN_BOTTOM = 80

    for page_num, page in enumerate(doc):
        try:
            tables = page.find_tables()
            table_bboxes = [fitz.Rect(tab.bbox) for tab in tables]
        except:
            table_bboxes = []

        page_height = page.rect.height
        words_on_page = page.get_text("words")
        words_on_page.sort(key=lambda w: (round(w[1], 1), w[0]))

        for w in words_on_page:
            word_rect = fitz.Rect(w[:4])
            is_excluded = False

            if word_rect.y1 < MARGIN_TOP or word_rect.y0 > (page_height - MARGIN_BOTTOM):
                is_excluded = True
            if not is_excluded:
                for table_rect in table_bboxes:
                    if table_rect.intersects(word_rect):
                        is_excluded = True
                        break

            global_word_list.append({
                'page': page_num,
                'rect': word_rect,
                'skip_highlight': is_excluded,
            })

    # Рендер страниц + блоков для HTML-превью
    pages = []
    page_blocks_map = {i: [] for i in range(start_page, len(doc))}

    for i, word_data in enumerate(global_word_list):
        if word_data['page'] < start_page or word_data['skip_highlight']:
            continue

        source_meta = word_to_source.get(i)
        if not source_meta:
            continue

        rect = word_data['rect']
        final_rect = fitz.Rect(rect.x0, rect.y0, rect.x1 + 2, rect.y1)

        # Та же логика объединения соседних слов в строке
        if i + 1 < len(global_word_list):
            next_w = global_word_list[i + 1]
            next_src = word_to_source.get(i + 1)
            if (
                next_src
                and next_src['sid'] == source_meta['sid']
                and not next_w['skip_highlight']
                and next_w['page'] == word_data['page']
                and abs(next_w['rect'].y0 - rect.y0) < 5
            ):
                final_rect = fitz.Rect(rect.x0, rect.y0, next_w['rect'].x0, rect.y1)

        # Перевод координат в мм для шаблона report.html
        x = final_rect.x0 * 25.4 / 72 - 25
        y = final_rect.y0 * 25.4 / 72 - 20
        ww = (final_rect.x1 - final_rect.x0) * 25.4 / 72
        hh = (final_rect.y1 - final_rect.y0) * 25.4 / 72

        page_blocks_map[word_data['page']].append({
            'x': x,
            'y': y,
            'w': ww,
            'h': hh,
            'sid': source_meta['sid'],
            'cit': source_meta['cit'],
        })

    for page_num in range(start_page, len(doc)):
        page = doc[page_num]
        text = page.get_text("html")
        pages.append({'text': text, 'blocks': page_blocks_map.get(page_num, [])})
    doc.close()
    
    return render_template('report.html',
                         filename=filename,
                         orig=round(unique, 1),
                         plag=round(plag, 1),
                         cit=round(pc, 1),
                         total=len(pages),
                         pages=pages)    

if __name__ == '__main__':
    app.run(debug=True, port=5000)
