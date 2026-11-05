# Подключаем средства работы с файлами и аргументами.
import argparse  # Читаем параметры запуска.
import math  # Вычисляем углы и тригонометрию.
import sys  # Выводим сообщения об ошибках.
import time  # Измеряем длительность кадров.
from pathlib import Path  # Находим ресурсы рядом с программой.

# Подключаем графические библиотеки.
import glfw  # Создаем окно и получаем ввод.
import moderngl  # Управляем OpenGL через Python.
import numpy as np  # Работаем с векторами и матрицами.
from PIL import Image  # Загружаем текстуры и сохраняем кадры.

# Задаем постоянные параметры сцены.
ROOT = Path(__file__).resolve().parent  # Не зависим от текущей папки.
SUN_POSITION = np.array((-0.55, 0.9, 0.65), dtype="f4")  # Задаем сторону расположения солнца.
#SUN_POSITION = np.array((0.8, 0.25, -0.5), dtype="f4") # Низкое солнце сбоку (закат).
SUN_POSITION *= 32 / np.linalg.norm(SUN_POSITION)  # Сохраняем прежнее положение солнечного шара.
SHADOW_NEAR, SHADOW_FAR = 5.0, 120.0  # Ограничиваем глубину солнечной камеры.
SKY = (0.54, 0.73, 0.87)  # Цвет неба и далекого тумана.

# Переносим вершины в мировые координаты и на экран.
VERTEX_SHADER = """
#version 330
in vec3 in_position; // Положение вершины.
in vec3 in_normal; // Нормаль поверхности.
in vec2 in_uv; // Координаты текстуры.
uniform mat4 model; // Преобразование объекта.
uniform mat3 normal_matrix; // Преобразование нормалей.
uniform mat4 view_projection; // Матрица камеры.
uniform mat4 light_matrix; // Матрица солнечной камеры.
out vec3 world_position; // Мировая позиция для освещения.
out vec3 world_normal; // Мировая нормаль.
out vec2 uv; // Координаты для фрагмента.
out vec4 light_position; // Позиция в карте теней.
void main() { // Обрабатываем одну вершину.
    vec4 p = model * vec4(in_position, 1.0); // Переносим вершину.
    world_position = p.xyz; // Запоминаем мировые координаты.
    world_normal = normal_matrix * in_normal; // Переносим нормаль.
    uv = in_uv; // Передаем координаты текстуры.
    light_position = light_matrix * p; // Проецируем со стороны солнца.
    gl_Position = view_projection * p; // Проецируем со стороны камеры.
}
"""  # Храним GLSL прямо в этом файле.

# Вычисляем текстуру, освещение и видимость солнца.
FRAGMENT_SHADER = """
#version 330
in vec3 world_position; // Мировая позиция фрагмента.
in vec3 world_normal; // Интерполированная нормаль.
in vec2 uv; // Интерполированные координаты текстуры.
in vec4 light_position; // Позиция в солнечной проекции.
uniform sampler2D surface_texture; // Цветовая текстура.
uniform sampler2D shadow_texture; // Глубина со стороны солнца.
uniform vec3 eye; // Положение наблюдателя.
uniform vec3 sun_position; // Общая позиция светящегося шара и источника света.
uniform float shadow_depth_scale; // Перевод малого смещения в перспективную глубину.
uniform vec3 tint; // Диффузный цвет материала.
uniform vec3 specular; // Цвет зеркального блика.
uniform vec2 uv_scale; // Число повторов текстуры.
uniform float shininess; // Степень блеска материала.
uniform float emission; // Собственное свечение.
uniform bool shadows; // Переключатель теней.
out vec4 fragment_color; // Итоговый цвет пикселя.
float sunlight(vec3 n, vec3 light_direction) { // Находим долю видимого света.
    if (!shadows) return 1.0; // Пропускаем отключенную тень.
    vec3 p = light_position.xyz / light_position.w; // Убираем однородность.
    p = p * 0.5 + 0.5; // Переходим к диапазону от нуля до единицы.
    if (any(lessThan(p, vec3(0.0))) || any(greaterThan(p, vec3(1.0))))
        return 1.0; // Не затеняем область за картой.
    vec3 dx = dFdx(p); // Изменение проекции по горизонтали экрана.
    vec3 dy = dFdy(p); // Изменение проекции по вертикали экрана.
    float determinant = dx.x * dy.y - dx.y * dy.x; // Определитель преобразования UV.
    vec2 slope = vec2(0.0); // Изменение глубины вдоль поверхности.
    if (abs(determinant) > 1e-12) // Избегаем деления у ребра проекции.
        slope = vec2(dy.y * dx.z - dx.y * dy.z, dx.x * dy.z - dy.x * dx.z) / determinant; // Учитываем наклон грани.
    float world_bias = max(0.008 * (1.0 - dot(n, light_direction)), 0.002); // Задаем малое смещение в единицах сцены.
    float bias = shadow_depth_scale * world_bias / (light_position.w * light_position.w); // Учитываем перспективное распределение глубины.
    vec2 texel = 1.0 / vec2(textureSize(shadow_texture, 0)); // Размер текселя.
    float visible = 0.0; // Накапливаем освещенные выборки.
    for (int x = -1; x <= 1; ++x) { // Соседние столбцы карты.
        for (int y = -1; y <= 1; ++y) { // Соседние строки карты.
            vec2 sample_uv = (floor(p.xy / texel) + vec2(x, y) + 0.5) * texel; // Берем центр текселя.
            float depth = texture(shadow_texture, sample_uv).r; // Читаем глубину.
            float receiver = p.z + dot(slope, sample_uv - p.xy); // Уточняем глубину наклонной поверхности.
            visible += (receiver - bias <= depth) ? 1.0 : 0.0; // Сравниваем расстояния.
        }
    }
    return visible / 9.0; // Сглаживаем край тени.
}
void main() { // Освещаем один фрагмент.
    vec3 base = texture(surface_texture, uv * uv_scale).rgb * tint; // Накладываем текстуру.
    if (emission >= 1.0) { // Показываем полностью светящийся материал солнца.
        fragment_color = vec4(base, 1.0); // Сохраняем яркость независимо от освещения.
        return; // Не накладываем на солнце тени и туман.
    }
    vec3 n = normalize(world_normal); // Восстанавливаем единичную нормаль.
    if (!gl_FrontFacing) n = -n; // Освещаем обратные стороны импортированных граней.
    vec3 v = normalize(eye - world_position); // Направление к наблюдателю.
    vec3 light_direction = normalize(sun_position - world_position); // Направляем свет от центра видимого солнца.
    vec3 h = normalize(light_direction + v); // Полувектор Блинна.
    float diffuse = max(dot(n, light_direction), 0.0); // Диффузное освещение Ламберта.
    float highlight = pow(max(dot(n, h), 0.0), shininess); // Зеркальный блик.
    if (diffuse <= 0.0) highlight = 0.0; // Убираем блик с теневой стороны.
    float visibility = sunlight(n, light_direction); // Проверяем перекрытие солнца.
    vec3 ambient = base * vec3(0.34, 0.37, 0.41); // Добавляем рассеянный свет неба.
    vec3 direct = base * diffuse * vec3(0.86, 0.80, 0.69); // Слегка теплый солнечный свет.
    vec3 color = ambient + visibility * (direct + specular * highlight); // Суммируем освещение.
    color += base * emission; // Подсвечиваем сигналы светофора.
    float fog = smoothstep(65.0, 150.0, length(eye - world_position)); // Ослабляем дальние объекты.
    color = mix(color, vec3(0.54, 0.73, 0.87), fog); // Смешиваем цвет с небом.
    fragment_color = vec4(color, 1.0); // Записываем непрозрачный цвет.
}
"""  # В GLSL используются короткие комментарии через две косые черты.

# Записываем ближайшие к солнцу поверхности в буфер глубины.
DEPTH_VERTEX = """
#version 330
in vec3 in_position; // Локальная позиция вершины.
uniform mat4 model; // Матрица объекта.
uniform mat4 light_matrix; // Матрица солнечной камеры.
void main() { // Вычисляем положение в карте теней.
    gl_Position = light_matrix * model * vec4(in_position, 1.0); // Проецируем вершину.
}
"""  # Цвет в первом проходе не нужен.
DEPTH_FRAGMENT = """
#version 330
void main() {} // Глубина записывается автоматически.
"""  # Используем только буфер глубины.


# Приводим произвольный вектор к единичной длине.
def normalize(vector):  # Не меняем направление вектора.
    vector = np.asarray(vector, dtype="f4")  # Используем числа одинарной точности.
    return vector / max(float(np.linalg.norm(vector)), 1e-8)  # Защищаем деление.


# Собираем матрицу положения камеры.
def look_at(eye, target):  # Направляем камеру на выбранную точку.
    forward = normalize(np.asarray(target) - eye)  # Направление взгляда.
    right = normalize(np.cross(forward, (0, 1, 0)))  # Правая ось камеры.
    up = np.cross(right, forward)  # Верхняя ось камеры.
    matrix = np.eye(4, dtype="f4")  # Начинаем с единичной матрицы.
    matrix[:3, :3] = np.array((right, up, -forward))  # Записываем базис.
    matrix[:3, 3] = -matrix[:3, :3] @ eye  # Учитываем положение камеры.
    return matrix  # Возвращаем видовую матрицу.


# Строим перспективную проекцию.
def perspective(aspect):  # Сохраняем пропорции окна.
    f = 1.0 / math.tan(math.radians(52.0) / 2)  # Задаем вертикальный угол обзора.
    near, far = 0.12, 180.0  # Ограничиваем видимую глубину.
    return np.array(((f / aspect, 0, 0, 0), (0, f, 0, 0),
                     (0, 0, (far + near) / (near - far), 2 * far * near / (near - far)),
                     (0, 0, -1, 0)), dtype="f4")  # Получаем матрицу OpenGL.


# Создаем перспективную камеру в центре видимого солнца.
def sunlight_matrix(objects):  # Охватываем поверхности городка одной картой глубины.
    view = look_at(SUN_POSITION, np.zeros(3, dtype="f4"))  # Смотрим от солнечного шара к городку.
    projected = []  # Собираем границы объектов в солнечной системе координат.
    for mesh, material, matrix, normal, uv, cast in objects:  # Учитываем и объекты, и принимающие тень покрытия.
        if material.emission >= 1:  # Исключаем сам светящийся шар.
            continue  # Не включаем источник в его собственную карту.
        low, high = mesh.bounds  # Получаем локальные габариты геометрии.
        corners = np.array([(x, y, z, 1) for x in (low[0], high[0])
                            for y in (low[1], high[1]) for z in (low[2], high[2])], dtype="f4")  # Собираем восемь углов.
        points = (view @ matrix @ corners.T).T  # Переводим углы в солнечную камеру.
        projected.extend(points[:, :2] / -points[:, 2:3])  # Находим отношения координат к глубине.
    bounds = np.asarray(projected, dtype="f4")  # Объединяем границы всей сцены.
    lower, upper = bounds.min(axis=0) - .02, bounds.max(axis=0) + .02  # Оставляем небольшой запас по краям.
    near, far = SHADOW_NEAR, SHADOW_FAR  # Используем общие границы глубины.
    left, bottom = lower * near  # Получаем нижний угол ближней плоскости.
    right, top = upper * near  # Получаем верхний угол ближней плоскости.
    projection = np.zeros((4, 4), dtype="f4")  # Готовим перспективную матрицу OpenGL.
    projection[0, 0], projection[1, 1] = 2 * near / (right - left), 2 * near / (top - bottom)  # Подгоняем масштаб под городок.
    projection[0, 2], projection[1, 2] = (right + left) / (right - left), (top + bottom) / (top - bottom)  # Учитываем смещение объема.
    projection[2, 2] = -(far + near) / (far - near)  # Переводим расстояние в глубину OpenGL.
    projection[2, 3], projection[3, 2] = -2 * far * near / (far - near), -1  # Включаем перспективное деление.
    return projection @ view  # Лучи карты исходят точно из центра солнечного шара.


# Создаем преобразование объекта с поворотом вокруг вертикали.
def transform(position=(0, 0, 0), scale=(1, 1, 1), angle=0):  # Задаем положение и размеры.
    a = math.radians(angle)  # Переводим градусы в радианы.
    c, s = math.cos(a), math.sin(a)  # Вычисляем поворот один раз.
    matrix = np.array(((c, 0, s, 0), (0, 1, 0, 0), (-s, 0, c, 0), (0, 0, 0, 1)), dtype="f4")  # Поворот вокруг Y.
    matrix[:3, :3] *= np.asarray(scale, dtype="f4")  # Масштабируем локальные оси.
    matrix[:3, 3] = position  # Записываем перенос.
    return matrix  # Получаем модельную матрицу.


# Передаем матрицу в столбцовом порядке OpenGL.
def write_matrix(uniform, matrix):  # NumPy хранит строки по умолчанию.
    uniform.write(np.asarray(matrix, dtype="f4").T.tobytes())  # Транспонируем перед отправкой.


# Добавляем плоскую грань и разбиваем ее на треугольники.
def face(vertices, points, uv=None):  # Сохраняем позиции, нормали и UV.
    points = np.asarray(points, dtype="f4")  # Подготавливаем координаты.
    normal = normalize(np.cross(points[1] - points[0], points[2] - points[0]))  # Находим нормаль грани.
    uv = uv if uv is not None else ((0, 0), (1, 0), (1, 1), (0, 1))  # Задаем обычную развертку.
    for i in range(1, len(points) - 1):  # Строим веер треугольников.
        for j in (0, i, i + 1):  # Обходим три вершины.
            vertices.append((*points[j], *normal, *uv[j]))  # Сохраняем восемь чисел.


# Строим единичный куб с отдельными нормалями граней.
def cube_vertices():  # Масштаб зададим при размещении.
    data = []  # Накапливаем вершины.
    face(data, ((-.5, -.5, .5), (.5, -.5, .5), (.5, .5, .5), (-.5, .5, .5)))  # Передняя грань.
    face(data, ((.5, -.5, -.5), (-.5, -.5, -.5), (-.5, .5, -.5), (.5, .5, -.5)))  # Задняя грань.
    face(data, ((.5, -.5, .5), (.5, -.5, -.5), (.5, .5, -.5), (.5, .5, .5)))  # Правая грань.
    face(data, ((-.5, -.5, -.5), (-.5, -.5, .5), (-.5, .5, .5), (-.5, .5, -.5)))  # Левая грань.
    face(data, ((-.5, .5, .5), (.5, .5, .5), (.5, .5, -.5), (-.5, .5, -.5)))  # Верхняя грань.
    face(data, ((-.5, -.5, -.5), (.5, -.5, -.5), (.5, -.5, .5), (-.5, -.5, .5)))  # Нижняя грань.
    return data  # Возвращаем 12 треугольников.


# Создаем крышу в форме треугольной призмы.
def roof_vertices():  # Основание находится на высоте ноль.
    data = []  # Собираем пять граней.
    face(data, ((-.5, 0, .5), (.5, 0, .5), (0, 1, .5)), ((0, 0), (1, 0), (.5, 1)))  # Передний скатный торец.
    face(data, ((.5, 0, -.5), (-.5, 0, -.5), (0, 1, -.5)), ((0, 0), (1, 0), (.5, 1)))  # Задний торец.
    face(data, ((-.5, 0, -.5), (-.5, 0, .5), (0, 1, .5), (0, 1, -.5)))  # Левый скат.
    face(data, ((.5, 0, .5), (.5, 0, -.5), (0, 1, -.5), (0, 1, .5)))  # Правый скат.
    face(data, ((-.5, 0, -.5), (.5, 0, -.5), (.5, 0, .5), (-.5, 0, .5)))  # Нижняя грань.
    return data  # Крыша остается замкнутой для теней.


# Строим цилиндр или конус с небольшим числом сегментов.
def round_vertices(top_radius, segments=12):  # Нулевой верхний радиус дает конус.
    data = []  # Собираем боковины и основания.
    for i in range(segments):  # Обходим окружность.
        a, b = 2 * math.pi * i / segments, 2 * math.pi * (i + 1) / segments  # Углы соседних ребер.
        p = (math.cos(a), 0, math.sin(a))  # Первая нижняя вершина.
        q = (math.cos(b), 0, math.sin(b))  # Вторая нижняя вершина.
        r = (q[0] * top_radius, 1, q[2] * top_radius)  # Вторая верхняя вершина.
        s = (p[0] * top_radius, 1, p[2] * top_radius)  # Первая верхняя вершина.
        face(data, (q, p, s) if top_radius == 0 else (q, p, s, r))  # Боковая поверхность.
        face(data, ((0, 0, 0), p, q))  # Нижняя крышка.
        if top_radius > 0:  # Конусу верхняя крышка не нужна.
            face(data, ((0, 1, 0), r, s))  # Верхняя крышка.
    return data  # Возвращаем готовый примитив.


# Создаем сферу для видимого солнца.
def sphere_vertices(rings=12, segments=20):  # Ограничиваем число треугольников.
    data = []  # Накапливаем вершины сферы.
    for row in range(rings):  # Обходим полосы по широте.
        for column in range(segments):  # Обходим сегменты по долготе.
            corners = []  # Собираем углы текущей ячейки.
            for v, u in ((row, column), (row + 1, column),
                         (row + 1, column + 1), (row, column + 1)):  # Задаем обход ячейки.
                latitude = math.pi * v / rings  # Отсчитываем угол от верхнего полюса.
                longitude = 2 * math.pi * u / segments  # Получаем угол вокруг вертикали.
                point = (math.sin(latitude) * math.cos(longitude), math.cos(latitude),
                         math.sin(latitude) * math.sin(longitude))  # Находим точку единичной сферы.
                corners.append((*point, *point, u / segments, v / rings))  # Радиус служит нормалью.
            indices = (() if row == 0 else (0, 3, 1)) + (() if row == rings - 1 else (1, 3, 2))  # Убираем вырожденные грани полюсов.
            data.extend(corners[index] for index in indices)  # Добавляем треугольники ячейки.
    return data  # Возвращаем сферу с гладкими нормалями.


# Храним материал поверхности.
class Material:
    def __init__(self, texture, color=(1, 1, 1), specular=(.08, .08, .08), power=24, emission=0):  # Задаем свойства.
        self.texture = texture  # Ссылка на текстуру OpenGL.
        self.color = tuple(color)  # Диффузный цвет Kd.
        self.specular = tuple(specular)  # Зеркальный цвет Ks.
        self.power = float(power)  # Показатель степени Ns.
        self.emission = float(emission)  # Сила собственного свечения.


# Загружаем текстуры только один раз.
class Textures:
    def __init__(self, ctx):  # Сохраняем графический контекст.
        self.ctx = ctx  # Контекст OpenGL.
        self.cache = {}  # Повторно используемые текстуры.

    def get(self, path=None):  # Белый пиксель заменяет отсутствующую текстуру.
        key = str(Path(path).resolve()) if path else "white"  # Получаем ключ кеша.
        if key not in self.cache:  # Избегаем повторного чтения файла.
            if path:  # Загружаем изображение с диска.
                with Image.open(path) as source:  # Закрываем файл после чтения.
                    image = source.convert("RGB").transpose(Image.Transpose.FLIP_TOP_BOTTOM)  # Совмещаем начало UV.
            else:  # Создаем нейтральную текстуру.
                image = Image.new("RGB", (1, 1), "white")  # Не меняем цвет материала.
            texture = self.ctx.texture(image.size, 3, image.tobytes(), alignment=1)  # Копируем пиксели в OpenGL.
            texture.repeat_x = texture.repeat_y = True  # Разрешаем повторение текстуры.
            texture.build_mipmaps()  # Уменьшаем мерцание вдали.
            texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)  # Включаем сглаживание текстуры.
            self.cache[key] = texture  # Сохраняем ресурс для повторного использования.
        return self.cache[key]  # Возвращаем готовую текстуру.


# Храним один набор геометрии на видеокарте.
class Mesh:
    def __init__(self, ctx, program, depth_program, vertices):  # Создаем буфер и два VAO.
        data = np.asarray(vertices, dtype="f4")  # Упаковываем восемь чисел на вершину.
        self.bounds = (data[:, :3].min(axis=0), data[:, :3].max(axis=0))  # Сохраняем габариты для солнечной проекции.
        self.buffer = ctx.buffer(data.tobytes())  # Передаем геометрию в видеопамять.
        self.vao = ctx.vertex_array(program, [(self.buffer, "3f 3f 2f", "in_position", "in_normal", "in_uv")])  # Полный формат.
        self.depth_vao = ctx.vertex_array(depth_program, [(self.buffer, "3f 20x", "in_position")])  # Пропускаем нормаль и UV.
        self.triangles = len(data) // 3  # Подсчитываем треугольники.


# Читаем материалы стандартного формата MTL.
def load_mtl(path, textures):  # Используем Kd, Ks, Ns и map_Kd.
    descriptions, current = {}, None  # Подготавливаем таблицу описаний.
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():  # Обходим строки файла.
        words = line.partition("#")[0].split()  # Убираем комментарии и пробелы.
        if not words:  # Пропускаем пустые строки.
            continue  # Переходим к следующей записи.
        name, values = words[0], words[1:]  # Разделяем ключ и значение.
        if name == "newmtl":  # Начинаем описание нового материала.
            current = {}  # Создаем отдельный набор свойств.
            descriptions[" ".join(values)] = current  # Запоминаем исходное имя.
        elif current is not None and name in ("Kd", "Ks", "Ns", "map_Kd"):  # Читаем нужные параметры.
            current[name] = values  # Оставляем преобразование до конца чтения.
    materials = {}  # Собираем готовые материалы.
    for name, values in descriptions.items():  # Обходим все описания.
        texture_path = path.parent / " ".join(values["map_Kd"]) if "map_Kd" in values else None  # Ищем изображение рядом.
        color = tuple(map(float, values.get("Kd", (1, 1, 1))))  # Читаем диффузный цвет.
        specular = tuple(map(float, values.get("Ks", (.1, .1, .1))))  # Читаем зеркальный цвет.
        power = min(256.0, max(1.0, float(values.get("Ns", (24,))[0])))  # Ограничиваем степень блеска.
        materials[name] = Material(textures.get(texture_path), color, specular, power)  # Создаем материал.
    return materials  # Возвращаем таблицу по именам.


# Загружаем OBJ и достраиваем отсутствующие нормали.
def load_obj(path, ctx, program, depth_program, textures):  # Поддерживаем грани и группы материалов.
    positions, normals, coords, triangles = [], [], [], []  # Исходные данные OBJ.
    current, smooth = "default", "off"  # Начальные материал и группа сглаживания.
    mtl_path = path.with_suffix(".mtl")  # Учитываем переименованные пары OBJ и MTL.
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():  # Читаем модель построчно.
        words = line.partition("#")[0].split()  # Убираем комментарии.
        if not words:  # Пропускаем пустую строку.
            continue  # Продолжаем чтение.
        key, values = words[0], words[1:]  # Выделяем тип записи.
        if key == "v":  # Читаем вершину.
            positions.append(tuple(map(float, values[:3])))  # Сохраняем XYZ.
        elif key == "vn":  # Читаем готовую нормаль.
            normals.append(tuple(map(float, values[:3])))  # Сохраняем направление.
        elif key == "vt":  # Читаем развертку.
            coords.append(tuple(map(float, values[:2])))  # Сохраняем UV.
        elif key == "mtllib" and not mtl_path.exists():  # Пробуем исходное имя материала.
            mtl_path = path.parent / " ".join(values)  # Разрешаем путь относительно OBJ.
        elif key == "usemtl":  # Переключаем материал.
            current = " ".join(values)  # Сохраняем полное имя.
        elif key == "s":  # Читаем группу сглаживания.
            smooth = values[0]  # Значения off и 0 означают плоские грани.
        elif key == "f":  # Читаем многоугольник.
            corners = []  # Индексы его вершин.
            for value in values:  # Разбираем ссылки v/vt/vn.
                fields = value.split("/")  # Поля могут быть пустыми.
                indices = []  # Собираем индексы трех атрибутов.
                for j, length in enumerate((len(positions), len(coords), len(normals))):  # Обрабатываем три массива.
                    index = int(fields[j]) if j < len(fields) and fields[j] else 0  # Ноль обозначает отсутствие.
                    indices.append(index - 1 if index > 0 else length + index if index < 0 else None)  # Учитываем отрицательные ссылки.
                corners.append(tuple(indices))  # Сохраняем ссылку на вершину.
            for i in range(1, len(corners) - 1):  # Триангулируем выпуклую грань веером.
                triangles.append((current, smooth, (corners[0], corners[i], corners[i + 1])))  # Сохраняем треугольник.
    points = np.asarray(positions, dtype="f4")  # Ускоряем векторные операции.
    accumulated, face_normals = {}, []  # Готовим недостающие нормали.
    for material, group, corners in triangles:  # Первый проход по треугольникам.
        a, b, c = (points[corner[0]] for corner in corners)  # Получаем три позиции.
        normal = np.cross(b - a, c - a)  # Площадь учитывает вес грани.
        face_normals.append(normalize(normal))  # Сохраняем плоскую нормаль.
        if group not in ("off", "0"):  # Усредняем только внутри группы сглаживания.
            for vi, ti, ni in corners:  # Обходим вершины треугольника.
                if ni is None:  # Не заменяем нормали из файла.
                    accumulated.setdefault((vi, group), np.zeros(3, dtype="f4"))  # Создаем накопитель.
                    accumulated[(vi, group)] += normal  # Добавляем взвешенную нормаль.
    batches = {}  # Группируем геометрию по материалу.
    for index, (material, group, corners) in enumerate(triangles):  # Второй проход по граням.
        batch = batches.setdefault(material, [])  # Получаем список вершин материала.
        for vi, ti, ni in corners:  # Восстанавливаем полный набор атрибутов.
            normal = normals[ni] if ni is not None else normalize(accumulated.get((vi, group), face_normals[index]))  # Выбираем нормаль.
            uv = coords[ti] if ti is not None else (0, 0)  # Подставляем UV при наличии.
            batch.append((*points[vi], *normal, *uv))  # Формируем вершину для OpenGL.
    materials = load_mtl(mtl_path, textures) if mtl_path.exists() else {}  # Загружаем найденные материалы.
    fallback = Material(textures.get(), (.6, .6, .6))  # Запасной нейтральный материал.
    parts = [(Mesh(ctx, program, depth_program, data), materials.get(name, fallback), name)
             for name, data in batches.items()]  # Создаем отдельный меш на материал.
    low, high = points.min(axis=0), points.max(axis=0)  # Находим габариты модели.
    return parts, low, high  # Возвращаем части и ограничивающий параллелепипед.


# Собираем и рисуем неподвижный городок.
class Town:
    def __init__(self, ctx, low_quality=False):  # Создаем ресурсы один раз.
        self.ctx = ctx  # Сохраняем контекст.
        self.program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)  # Основная программа.
        self.depth_program = ctx.program(vertex_shader=DEPTH_VERTEX, fragment_shader=DEPTH_FRAGMENT)  # Программа глубины.
        self.textures = Textures(ctx)  # Создаем кеш текстур.
        self.objects = []  # Храним экземпляры объектов.
        self.meshes = {}  # Храним общую геометрию примитивов.
        for name, data in (("cube", cube_vertices()), ("roof", roof_vertices()),
                           ("cylinder", round_vertices(1)), ("cone", round_vertices(0)),
                           ("sphere", sphere_vertices())):  # Готовим пять примитивов.
            self.meshes[name] = Mesh(ctx, self.program, self.depth_program, data)  # Создаем общие буферы.
        self.materials = {}  # Готовим материалы примитивов.
        for name in ("grass", "brick", "roof", "asphalt", "metal", "window"):  # Загружаем все пользовательские PNG.
            self.materials[name] = Material(self.textures.get(ROOT / "textures" / f"{name}.png"))  # Связываем текстуру с материалом.
        self.materials["metal"].specular = (.65, .65, .65)  # Делаем металл блестящим.
        self.materials["metal"].power = 90  # Сужаем металлический блик.
        self.materials["window"].specular = (.8, .85, .9)  # Усиливаем блики на стекле.
        self.materials["window"].power = 110  # Делаем блик стекла узким.
        for name, color in {"paving": (.62, .60, .55), "white": (.93, .92, .83),
                            "wood": (.26, .13, .065), "bark": (.29, .18, .10),
                            "green": (.19, .40, .13), "green2": (.25, .48, .16)}.items():  # Добавляем одноцветные материалы.
            self.materials[name] = Material(self.textures.get(), color)  # Используем белую текстуру.
        self.materials["sun"] = Material(self.textures.get(), (1.0, .84, .30), (0, 0, 0), 1, 1)  # Создаем желтый светящийся материал.
        self.make_scene()  # Расставляем дома и детали улиц.
        self.load_models()  # Добавляем скачанные модели.
        self.light_matrix = sunlight_matrix(self.objects)  # Ставим солнечную камеру в центр шара.
        size = 1024 if low_quality else 2048  # Выбираем разрешение карты теней.
        self.shadow = ctx.depth_texture((size, size))  # Создаем текстуру глубины.
        self.shadow.compare_func = ""  # Читаем глубину обычным sampler2D.
        self.shadow.repeat_x = self.shadow.repeat_y = False  # Не повторяем карту за краями.
        self.shadow.filter = (moderngl.NEAREST, moderngl.NEAREST)  # Сравнение и сглаживание выполняет шейдер.
        self.shadow_fbo = ctx.framebuffer(depth_attachment=self.shadow)  # Подключаем глубину к framebuffer.
        self.program["surface_texture"].value = 0  # Нулевой слот содержит цвет.
        self.program["shadow_texture"].value = 1  # Первый слот содержит карту теней.
        self.program["sun_position"].value = tuple(SUN_POSITION)  # Передаем фактическую позицию солнца.
        self.program["shadow_depth_scale"].value = SHADOW_NEAR * SHADOW_FAR / (SHADOW_FAR - SHADOW_NEAR)  # Согласуем смещение глубины с проекцией.
        write_matrix(self.program["light_matrix"], self.light_matrix)  # Передаем солнечную проекцию.
        self.bake_shadows()  # Статическая сцена требует только одного расчета теней.

    def add(self, mesh, material, matrix, uv=(1, 1), cast=True):  # Размещаем один экземпляр.
        normal = np.linalg.inv(matrix[:3, :3]).T.astype("f4")  # Нормали учитывают неравномерный масштаб.
        self.objects.append((mesh, material, matrix, normal, uv, cast))  # Сохраняем готовые параметры.

    def primitive(self, kind, material, position, scale, angle=0, uv=(1, 1), cast=True):  # Упрощаем размещение примитивов.
        self.add(self.meshes[kind], self.materials[material], transform(position, scale, angle), uv, cast)  # Создаем экземпляр.

    def house(self, x, z, angle):  # Строим один дом лицом к дороге.
        base = transform((x, .12, z), angle=angle)  # Ставим дом прямо на дорожку.
        def part(material, pos, size, uv=(1, 1), kind="cube", cast=True):  # Размещаем деталь в координатах дома.
            self.add(self.meshes[kind], self.materials[material], base @ transform(pos, size), uv, cast)  # Учитываем поворот дома.
        part("brick", (0, 2.3, 0), (5.6, 4.6, 5.0), (3, 2.6))  # Кирпичный корпус касается основания.
        part("roof", (0, 4.6, 0), (6.2, 1.7, 5.6), (3, 2), "roof")  # Добавляем двускатную крышу.
        part("brick", (1.6, 5.8, -.9), (.65, 1.7, .7), (.5, 1))  # Выводим дымоход выше ската.
        part("metal", (1.6, 6.67, -.9), (.83, .12, .88))  # Закрываем верх трубы.
        for side in (-1, 1):  # Оформляем передний и задний фасады.
            for level in (1.3, 3.4):  # Два этажа окон.
                for offset in (-1.65, 1.65):  # Два окна на каждом этаже.
                    part("paving", (offset, level, side * 2.525), (1.28, 1.42, .10))  # Создаем светлую раму.
                    part("window", (offset, level, side * 2.585), (1.1, 1.25, .025), cast=False)  # Накладываем PNG окна.
        part("wood", (0, 1.0, 2.54), (.95, 2.0, .10))  # Добавляем входную дверь.
        part("metal", (.32, .96, 2.61), (.055, .16, .04), cast=False)  # Размещаем дверную ручку.
        part("paving", (0, .08, 2.95), (1.45, .16, .8))  # Приставляем низкую ступень.
        for side in (-1, 1):  # Оформляем боковые стены.
            for level in (1.3, 3.4):  # Повторяем окна на обоих этажах.
                matrix = base @ transform((side * 2.845, level, 0), (1.2, 1.35, .025), 90)  # Поворачиваем плоскость окна.
                self.add(self.meshes["cube"], self.materials["window"], matrix, cast=False)  # Покрываем боковое окно текстурой.

    def tree(self, x, z, height=4.4):  # Создаем дерево из цилиндра и конусов.
        self.primitive("cylinder", "bark", (x, 0, z), (.18, height * .48, .18))  # Ствол уходит в землю.
        self.primitive("cone", "green", (x, height * .25, z), (1.35, height * .7, 1.35))  # Нижняя часть кроны.
        self.primitive("cone", "green2", (x, height * .51, z), (1.0, height * .55, 1.0))  # Верхняя часть кроны.

    def make_scene(self):  # Собираем улицы, дома и озеленение.
        self.primitive("sphere", "sun", tuple(SUN_POSITION), (2.1, 2.1, 2.1), cast=False)  # Совмещаем солнечный шар и источник света.
        self.primitive("cube", "grass", (0, -.16, 0), (54, .32, 48), uv=(22, 20), cast=False)  # Земля заканчивается на Y=0.
        self.primitive("cube", "asphalt", (0, .01, 0), (6.2, .02, 44), uv=(3, 22), cast=False)  # Продольная дорога.
        self.primitive("cube", "asphalt", (0, .022, -8), (50, .024, 5), uv=(25, 2.5), cast=False)  # Поперечная дорога.
        for x in (-3.75, 3.75):  # Тротуары по сторонам продольной улицы.
            for z, length in ((-16.85, 10.3), (8.85, 26.3)):  # Оставляем проезд на перекрестке.
                self.primitive("cube", "paving", (x, .06, z), (1.3, .12, length), cast=False)  # Высота тротуара равна .12.
        for z in (-11.1, -4.9):  # Тротуары поперечной улицы.
            for x in (-14.65, 14.65):  # Оставляем свободной продольную дорогу.
                self.primitive("cube", "paving", (x, .06, z), (20.5, .12, 1.2), cast=False)  # Делаем низкие бордюры.
        for z in range(-20, 22, 4):  # Наносим прерывистую осевую линию.
            if -12 < z < -4:  # Не рисуем разметку внутри перекрестка.
                continue  # Пропускаем этот штрих.
            self.primitive("cube", "white", (0, .038, z), (.11, .008, 1.8), cast=False)  # Рисуем тонкий штрих.
        for x in range(-23, 24, 4):  # Размечаем поперечную улицу.
            if abs(x) > 4:  # Оставляем середину перекрестка пустой.
                self.primitive("cube", "white", (x, .042, -8), (1.8, .008, .11), cast=False)  # Добавляем осевую разметку.
        for z in (-12.4, -3.6):  # Размещаем два пешеходных перехода.
            for x in np.arange(-2.5, 2.6, .85):  # Повторяем полосы поперек дороги.
                self.primitive("cube", "white", (x, .04, z), (.45, .008, 1.35), cast=False)  # Формируем полосу перехода.
        for x in (-10, 10):  # Создаем два ряда домов.
            for z in (-16.8, 1.5, 14.2):  # Оставляем промежутки между участками.
                self.primitive("cube", "paving", (x, .06, z), (7.3, .12, 7.6), cast=False)  # Основание под домом.
                self.primitive("cube", "paving", (math.copysign(5.5, x), .06, z), (2.3, .12, 1.4), cast=False)  # Дорожка от входа.
                self.house(x, z, 90 if x < 0 else -90)  # Поворачиваем фасад к улице.
        for x, z, height in ((-17, -19, 4.9), (17, -18, 5.2), (-17, -1, 4.6),
                             (17, 2, 4.8), (-17, 16, 5.4), (17, 16, 4.7),
                             (-8, 8, 3.9), (6, 8, 4.0), (-21, 7, 5), (21, 10, 5.1)):  # Задаем положение деревьев.
            self.tree(x, z, height)  # Создаем неподвижное дерево.
        for x in (-4.1, 4.1):  # Ставим фонари вдоль тротуара.
            for z in (5.0, 19.0):  # Оставляем окна и переходы открытыми.
                self.primitive("cylinder", "metal", (x, .12, z), (.065, 4.3, .065))  # Металлическая стойка.
                self.primitive("cube", "metal", (x, 4.4, z), (.5, .15, .7))  # Корпус уличного фонаря.
                self.primitive("cube", "white", (x, 4.31, z), (.36, .035, .53), cast=False)  # Нижний рассеиватель.

    def load_models(self):  # Загружаем все три пользовательские модели.
        def model(name):  # Находим OBJ в папке models.
            return load_obj(ROOT / "models" / f"{name}.obj", self.ctx, self.program, self.depth_program, self.textures)  # Читаем модель.
        def place(asset, pos, height=None, length=None, angle=0):  # Приводим модель к размерам сцены.
            parts, low, high = asset  # Получаем части и габариты.
            factor = height / (high[1] - low[1]) if height else length / (high[2] - low[2])  # Вычисляем единый масштаб.
            center = ((low[0] + high[0]) / 2, low[1], (low[2] + high[2]) / 2)  # Находим центр основания.
            matrix = transform(pos, (factor,) * 3, angle) @ transform(-np.asarray(center))  # Ставим низ точно на поверхность.
            for mesh, material, name in parts:  # Размещаем все материалы модели.
                self.add(mesh, material, matrix)  # Геометрия повторных экземпляров общая.
        car = model("car")  # Загружаем машину с ее MTL.
        for mesh, material, name in car[0]:  # Настраиваем кузов и стекла.
            if name == "White":  # Меняем только цвет кузова.
                material.color = (.12, .36, .63)  # Выделяем машину синим цветом.
                material.specular, material.power = (.7, .7, .7), 100  # Добавляем блеск краски.
            elif name == "Windows":  # Отдельно настраиваем стекло.
                material.color = (.075, .14, .19)  # Слегка окрашиваем окна.
        place(car, (1.65, .021, 8.5), length=3.9, angle=180)  # Машина стоит колесами на асфальте.
        place(car, (-1.65, .021, -17), length=3.9)  # Добавляем второй экземпляр.
        bench = model("wood_bank")  # Загружаем лавочку с древесной текстурой.
        place(bench, (-5.5, 0, 7.8), height=1.05, angle=90)  # Ставим первую лавочку на газон.
        place(bench, (5.5, 0, 18.8), height=1.05, angle=-90)  # Ставим вторую лавочку.
        signal = model("semaforo")  # Загружаем светофор с обеими текстурами.
        for mesh, material, name in signal[0]:  # Настраиваем сигнальные поверхности.
            if name in ("green", "red", "ambar") or "semaforo_boys" in name:  # Находим линзы и пешеходные секции.
                material.emission = .45 if name == "green" else .12  # Зеленый сигнал светится ярче.
                if "semaforo_boys" in name:  # Сохраняем читаемость пиктограмм.
                    material.color = (.85, .85, .85)  # Не затемняем картинку исходным Kd.
        place(signal, (3.85, .12, -12.2), height=4.6, angle=180)  # Ставим первый светофор на тротуар.
        place(signal, (-3.85, .12, -3.9), height=4.6)  # Размещаем встречный светофор.

    def bake_shadows(self):  # Рассчитываем глубину только при создании сцены.
        self.shadow_fbo.use()  # Выбираем буфер теней.
        self.ctx.viewport = (0, 0, *self.shadow.size)  # Совмещаем область вывода с текстурой.
        self.shadow_fbo.clear(depth=1.0)  # Начинаем с максимальной глубины.
        self.ctx.enable(moderngl.DEPTH_TEST)  # Оставляем ближайшие к солнцу поверхности.
        self.ctx.disable(moderngl.CULL_FACE)  # Сохраняем обе стороны импортированной геометрии.
        write_matrix(self.depth_program["light_matrix"], self.light_matrix)  # Передаем солнечную камеру.
        for mesh, material, matrix, normal, uv, cast in self.objects:  # Обходим объекты городка.
            if cast:  # Покрытия и разметка сами не отбрасывают тень.
                write_matrix(self.depth_program["model"], matrix)  # Передаем положение объекта.
                mesh.depth_vao.render(moderngl.TRIANGLES)  # Записываем глубину его треугольников.

    def render(self, eye, target, width, height, shadows=True):  # Рисуем сцену с текущей камеры.
        self.ctx.screen.use()  # Возвращаемся к экранному framebuffer.
        self.ctx.viewport = (0, 0, width, height)  # Учитываем размер окна.
        self.ctx.clear(*SKY, depth=1.0)  # Очищаем небо и глубину.
        view_projection = perspective(width / height) @ look_at(eye, target)  # Объединяем матрицы камеры.
        write_matrix(self.program["view_projection"], view_projection)  # Передаем проекцию объектов.
        self.program["eye"].value = tuple(eye)  # Передаем положение наблюдателя.
        self.program["shadows"].value = shadows  # Учитываем переключатель теней.
        self.shadow.use(location=1)  # Привязываем общую карту глубины.
        for mesh, material, matrix, normal, uv, cast in self.objects:  # Рисуем экземпляры сцены.
            write_matrix(self.program["model"], matrix)  # Устанавливаем матрицу объекта.
            write_matrix(self.program["normal_matrix"], normal)  # Устанавливаем матрицу нормалей.
            self.program["tint"].value = material.color  # Передаем диффузный цвет.
            self.program["specular"].value = material.specular  # Передаем силу блика.
            self.program["shininess"].value = material.power  # Передаем степень блеска.
            self.program["emission"].value = material.emission  # Передаем собственное свечение.
            self.program["uv_scale"].value = uv  # Выбираем повторение текстуры.
            material.texture.use(location=0)  # Привязываем цветовую текстуру.
            mesh.vao.render(moderngl.TRIANGLES)  # Отправляем треугольники на экран.


# Управляем свободной камерой без движения объектов сцены.
class Camera:
    def __init__(self, window):  # Запоминаем окно ввода.
        self.window = window  # Окно GLFW.
        self.dragging = False  # Обзор включается правой кнопкой мыши.
        self.last_mouse = None  # Предыдущие координаты курсора.
        self.reset()  # Устанавливаем общий вид городка.
        glfw.set_cursor_pos_callback(window, self.mouse)  # Получаем движение мыши.
        glfw.set_mouse_button_callback(window, self.button)  # Получаем нажатия кнопок.

    def reset(self):  # Возвращаем стартовый ракурс.
        self.position = np.array((35, 32, 42), dtype="f4")  # Поднимаем камеру над городком.
        direction = normalize(np.array((0, 0, 0), dtype="f4") - self.position)  # Смотрим на центр улиц.
        self.yaw = math.degrees(math.atan2(direction[2], direction[0]))  # Горизонтальный угол.
        self.pitch = math.degrees(math.asin(float(direction[1])))  # Вертикальный угол.

    def forward(self):  # Вычисляем направление взгляда.
        yaw, pitch = math.radians(self.yaw), math.radians(self.pitch)  # Переводим углы в радианы.
        return normalize((math.cos(yaw) * math.cos(pitch), math.sin(pitch), math.sin(yaw) * math.cos(pitch)))  # Строим единичный вектор.

    def button(self, window, button, action, mods):  # Включаем обзор правой кнопкой.
        if button == glfw.MOUSE_BUTTON_RIGHT:  # Остальные кнопки не меняют камеру.
            self.dragging = action == glfw.PRESS  # Запоминаем состояние кнопки.
            self.last_mouse = None  # Не допускаем скачка первого перемещения.
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED if self.dragging else glfw.CURSOR_NORMAL)  # Переключаем курсор.

    def mouse(self, window, x, y):  # Поворачиваем взгляд при движении мыши.
        if self.dragging and self.last_mouse is not None:  # Сравниваем две позиции курсора.
            self.yaw += (x - self.last_mouse[0]) * .16  # Поворачиваем по горизонтали.
            self.pitch = float(np.clip(self.pitch - (y - self.last_mouse[1]) * .16, -85, 85))  # Ограничиваем наклон.
        self.last_mouse = (x, y)  # Сохраняем положение курсора.

    def update(self, dt):  # Движение зависит от времени, а не от числа кадров.
        pressed = lambda key: glfw.get_key(self.window, key) == glfw.PRESS  # Сокращаем проверку клавиши.
        self.yaw += (pressed(glfw.KEY_RIGHT) - pressed(glfw.KEY_LEFT)) * 65 * dt  # Поворот стрелками.
        self.pitch = float(np.clip(self.pitch + (pressed(glfw.KEY_UP) - pressed(glfw.KEY_DOWN)) * 65 * dt, -85, 85))  # Наклон стрелками.
        forward = self.forward()  # Получаем направление взгляда.
        right = normalize(np.cross(forward, (0, 1, 0)))  # Получаем направление вправо.
        move = forward * (pressed(glfw.KEY_W) - pressed(glfw.KEY_S))  # Двигаемся вперед или назад.
        move += right * (pressed(glfw.KEY_D) - pressed(glfw.KEY_A))  # Двигаемся в стороны.
        move[1] += pressed(glfw.KEY_E) - pressed(glfw.KEY_Q)  # Меняем высоту камеры.
        speed = 15 if pressed(glfw.KEY_LEFT_SHIFT) else 7  # Ускоряем движение с Shift.
        if np.linalg.norm(move) > 0:  # Не нормируем нулевой вектор.
            self.position += normalize(move) * speed * dt  # Перемещаем камеру равномерно.
        self.position[1] = max(.3, self.position[1])  # Не опускаемся под землю.


# Сохраняем изображение текущего framebuffer.
def save_frame(ctx, path, width, height):  # Используем для снимка окна.
    capture = ctx.simple_framebuffer((width, height), components=3)  # Создаем обычный буфер для чтения.
    ctx.copy_framebuffer(capture, ctx.screen)  # Копируем экран и разрешаем MSAA.
    data = capture.read(components=3, alignment=1)  # Читаем RGB из собственного framebuffer.
    capture.release()  # Освобождаем временный графический буфер.
    ctx.screen.use()  # Восстанавливаем вывод в окно.
    image = Image.frombytes("RGB", (width, height), data)  # Собираем изображение.
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)  # Переворачиваем строки OpenGL.
    image.save(path)  # Записываем PNG по заданному пути.


# Открываем окно и запускаем цикл отображения.
def main():  # Все ресурсы сцены находятся рядом с main.py.
    parser = argparse.ArgumentParser(description="Статичный городок OpenGL")  # Описываем параметры запуска.
    parser.add_argument("--low", action="store_true", help="окно 960x600 и карта теней 1024x1024")  # Режим меньшей нагрузки.
    parser.add_argument("--screenshot", type=Path, help="сохранить один кадр и завершить работу")  # Проверка отрисовки.
    parser.add_argument("--no-shadows", action="store_true", help="запустить без теней")  # Позволяем сравнить освещение.
    args = parser.parse_args()  # Разбираем командную строку.
    if not glfw.init():  # Проверяем доступность оконной системы.
        raise RuntimeError("Не удалось запустить GLFW.")  # Выводим понятную причину.
    try:  # Гарантируем завершение GLFW при ошибке.
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)  # Запрашиваем OpenGL 3.
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)  # Требуем версию не ниже 3.3.
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)  # Используем современный конвейер.
        glfw.window_hint(glfw.SAMPLES, 0 if args.low else 2)  # Ограничиваем сглаживание для слабых устройств.
        if args.screenshot:  # Автоматическая проверка не требует видимого окна.
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # Создаем скрытое окно.
        size = (960, 600) if args.low else (1200, 760)  # Выбираем стартовый размер.
        window = glfw.create_window(*size, "Town | WASD QE | RMB / arrows | R reset | T shadows | F12 photo", None, None)  # Создаем окно.
        if not window:  # Обрабатываем отсутствие подходящего OpenGL.
            raise RuntimeError("Нужна поддержка OpenGL 3.3 и установленный драйвер видеокарты.")  # Сообщаем требование.
        glfw.make_context_current(window)  # Активируем контекст в текущем потоке.
        glfw.swap_interval(1)  # Включаем вертикальную синхронизацию.
        ctx = moderngl.create_context(require=330)  # Подключаем ModernGL к окну.
        ctx.enable(moderngl.DEPTH_TEST)  # Включаем правильное перекрытие объектов.
        town = Town(ctx, args.low)  # Создаем статическую сцену.
        camera = Camera(window)  # Подключаем динамическую камеру.
        shadows = not args.no_shadows  # Устанавливаем начальное состояние теней.
        previous_keys = set()  # Не повторяем переключения при удержании клавиш.
        previous_time = title_time = time.perf_counter()  # Запоминаем время запуска.
        frames = 0  # Начинаем подсчет частоты кадров.
        print("OpenGL:", ctx.info["GL_VERSION"], "|", ctx.info["GL_RENDERER"])  # Показываем выбранный драйвер.
        print("WASD: движение; Q/E: высота; ПКМ или стрелки: обзор; Shift: быстрее.")  # Выводим управление.
        print("R: начальный вид; T: тени; F12: снимок; Esc: выход.")  # Выводим дополнительные клавиши.
        while not glfw.window_should_close(window):  # Работаем до закрытия окна.
            glfw.poll_events()  # Обрабатываем ввод и изменение размера.
            now = time.perf_counter()  # Измеряем текущее время.
            dt = min(now - previous_time, .05)  # Ограничиваем скачки после паузы окна.
            previous_time = now  # Запоминаем начало кадра.
            keys = {key for key in (glfw.KEY_R, glfw.KEY_T, glfw.KEY_F12, glfw.KEY_ESCAPE)
                    if glfw.get_key(window, key) == glfw.PRESS}  # Получаем специальные клавиши.
            fresh = keys - previous_keys  # Выделяем только новые нажатия.
            previous_keys = keys  # Сохраняем состояние клавиатуры.
            if glfw.KEY_ESCAPE in fresh:  # Завершаем работу по Escape.
                glfw.set_window_should_close(window, True)  # Просим закрыть окно.
            if glfw.KEY_R in fresh:  # Возвращаем обзорный ракурс.
                camera.reset()  # Восстанавливаем положение камеры.
            if glfw.KEY_T in fresh:  # Переключаем видимость теней.
                shadows = not shadows  # Меняем только параметр шейдера.
            camera.update(dt)  # Обновляем положение наблюдателя.
            width, height = glfw.get_framebuffer_size(window)  # Учитываем реальный размер framebuffer.
            if width == 0 or height == 0:  # Окно может быть свернуто.
                glfw.wait_events_timeout(.05)  # Не нагружаем процессор пустыми кадрами.
                continue  # Ждем восстановления окна.
            town.render(camera.position, camera.position + camera.forward(), width, height, shadows)  # Рисуем неподвижный городок.
            if args.screenshot or glfw.KEY_F12 in fresh:  # Сохраняем кадр по запросу.
                destination = args.screenshot or ROOT / "screenshot.png"  # Выбираем имя снимка.
                save_frame(ctx, destination, width, height)  # Сохраняем еще не переключенный буфер.
                print("Снимок:", destination)  # Показываем путь к файлу.
                if args.screenshot:  # Однокадровый режим завершает работу.
                    break  # Выходим из цикла.
            glfw.swap_buffers(window)  # Показываем нарисованный кадр.
            frames += 1  # Учитываем показанный кадр.
            if now - title_time >= 1:  # Обновляем частоту кадров раз в секунду.
                fps = frames / (now - title_time)  # Вычисляем среднее значение.
                glfw.set_window_title(window, f"Town | {fps:.0f} FPS | WASD QE | RMB/arrows | R reset | T shadows | F12 photo")  # Выводим подсказки.
                title_time, frames = now, 0  # Начинаем новый интервал подсчета.
        ctx.release()  # Освобождаем графические ресурсы.
    finally:  # Выполняем очистку даже при исключении.
        glfw.terminate()  # Закрываем окно и освобождаем GLFW.


# Запускаем приложение только при прямом выполнении файла.
if __name__ == "__main__":  # Импорт не открывает окно.
    try:  # Показываем ошибки без потери их причины.
        main()  # Начинаем работу программы.
    except (RuntimeError, FileNotFoundError, moderngl.Error) as error:  # Перехватываем ошибки ресурсов и OpenGL.
        print(f"Ошибка: {error}", file=sys.stderr)  # Выводим короткое сообщение.
        sys.exit(1)  # Возвращаем код неудачного завершения.
