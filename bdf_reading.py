from numba import jit
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.gridspec as gridspec
from astropy.coordinates import SkyCoord
import numpy as np
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
import multiprocessing
import warnings
import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


class Counter:
    count = []

    def __init__(self, arg=None):
        if arg is not None and len(self.count) == 0:
            self.count.append(arg)
        elif arg is not None and len(self.count) != 0:
            self.count[0] = arg
            pass

    def append(self):
        self.count[0] += 1
        return self.count[0]


class UiUpdate:
    SIG = []

    def __init__(self, signals=None):
        if signals and len(self.SIG) == 0:
            self.SIG.append(signals)
        else:
            pass


def status_bar(modes_count, mode_number):
    percent = round(Counter().append() / modes_count * 100)

    UiUpdate().SIG[0].emit_update(mode_number, percent)


def main(op2_name, bdf_name,
         deform_multipler, coef_scale, dpi, swap, signals, stop):
    """
    Главная функция. Формирует файлы в формате png с графическим
     отображением собственных форм конечно-элементной модели в 4-х видах.
    Необходимы файлы BDF и OP2.

    :param  op2: объект op2 зачитанный из файла op2_name
            bdf: объект bdf зачитанный из файла bdf_name
            deform_multipler: множитель для масштабирования перемещений
            coef_scale: коэффицент для масштабирования изображения
            swap: режим поворота изображений (меняет оси местами):
            1 - XY, 2 - XZ, 3 - YZ
            signals: объект, с помощью которого организуется связь с
                                                        диалоговым окном
            stop: объект threading.event, служащий флагом для
                                остановки потока при нажатии cancel

    :return: возвращает файлы png, находящиеся в директории проекта python
    """

    # Инициализация классов
    UiUpdate(signals)
    Counter(0)

    print("Reading Nastran files...")
    if not stop.is_set():
        # Отсортированные QUAD4 и TRIA3 элемнты из BDF файла
        shell_elm = reading_bdf(bdf_name)
        phi = get_theta(shell_elm)

        # Словарь с узлами элементов (координаты в глобальной СК)
        elm_grids, elm_nondef_coords = get_nondef_coords(shell_elm, swap)

        # Список собственных векторов из OP2 файла
        eig = reading_op2(op2_name)

    file = open("freq.txt", "w")
    for i in range(len(eig.modes)):
        file.write(f'mode {eig.modes[i]}\t' + f'{round(eig.mode_cycles[i], 3)}\n')
    file.close()

    print("Creating pool process...")
    # Определение числа потоков
    cpu_count = multiprocessing.cpu_count()
    target_cpu = cpu_count // 2 if cpu_count > 1 else cpu_count

    # Создание обратной связи с потоками
    pipes = []
    pipes = [multiprocessing.Pipe() for i in range(len(eig.data))]

    # Подготовка входных аргументов по каждой форме для  get_mode_pict()
    all_args = [(eig.node_gridtype, eig.data[i], elm_grids, elm_nondef_coords,
                 eig.modes[i], (round(eig.mode_cycles[i], 3)), deform_multipler,
                 coef_scale, swap, dpi, pipes[i][1], phi) for i in range(len(eig.data))]

    if not stop.is_set():
        # Создание пула процессов, их разбивка по потокам и асинхронный запуск
        with multiprocessing.Pool(processes=target_cpu) as pool:
            [pool.apply_async(get_mode_pict, args) for args in all_args]

            # Возврат статуса о готовности +
            # механизм остановки при нажатии cancel
            for i in range(len(eig.data)):
                if not stop.is_set():
                    status_bar(len(eig.data), pipes[i][0].recv())
                else:
                    break
                    pool.close()
                    pool.terminate()
                    pool.join()


def get_mode_pict(node_gridtype, mode, elm_grids, elm_nondef_coords, mode_number,
                  m_cycle, deform_multipler, coef_scale, swap, dpi, child_p, phi):
    eig_increments = {}
    print(f"Creating {mode_number}")
    # Формирование списка перемещений узлов для собственной формы
    for i in range(len(mode)):
        eig_increments[node_gridtype[i][0]] = mode[i]
    # Получение координат узлов элементов
    elm_def_coords = get_elm_def_coords(elm_grids, eig_increments, deform_multipler, swap, phi)
    # # Информация для подписи изображения
    eig_description = [mode_number, m_cycle]
    # Отрисовка собственной формы
    plot = plot_elements(elm_def_coords,
                         elm_nondef_coords,
                         eig_description,
                         coef_scale)

    # Сохранение в файл
    f_name = f'mode_№_{mode_number}.png'
    plot.savefig(dpi=dpi, fname=f_name)
    child_p.send(mode_number)


def reading_op2(op2_name):
    op2 = OP2()
    # Чтение OP2 и BDF
    op2.read_op2(op2_name)
    # Обработка OP2 файла: извлечение eigenvectors
    for case_number in op2.eigenvectors:
        eig = op2.eigenvectors[case_number]

    return eig


def reading_bdf(bdf_name):
    bdf = BDF()

    # Чтение BDF
    bdf.read_bdf(bdf_name, xref=True)

    # Обработка BDF файла: отбор QUAD4 и TRIA3 элементов
    shell_elm = shell_sorting(bdf)

    return shell_elm


def get_theta(shell_elm):
    list_theta = {}
    for id in shell_elm:
        for node in shell_elm[id].nodes_ref:
            if node.cd_ref.Type == 'C':
                if node.cp_ref.Type != 'C':
                    temp_cartesian = (node.xyz - node.cd_ref.origin) @ node.cd_ref.global_to_local
                    list_theta[node.nid] = get_angles(temp_cartesian, 'C')
                else:
                    list_theta[node.nid] = node.xyz[1]
            if node.cd_ref.Type == 'S':
                if node.cp_ref.Type != 'S':
                    temp_cartesian = (node.xyz - node.cd_ref.origin) @ node.cd_ref.global_to_local
                    list_theta[node.nid] = get_angles(temp_cartesian, 'S')
                else:
                    list_theta[node.nid] = [node.xyz[0], node.xyz[1]]  ### При надобности подправить индексы

    return list_theta.copy()


def shell_sorting(bdf):
    """
    Перебирает все элементы в модели, оставляя только QUAD4 и TRIA3 элементы.

    :param bdf: BDF объект

    :return quad_tria список элементов QUAD4 и TRIA3

    """
    elements = dict(bdf.elements)
    quad_tria = {}
    for e_key in elements:
        if (bdf.elements[e_key].type == 'CQUAD4'
                or bdf.elements[e_key].type == 'CTRIA3'):
            quad_tria[e_key] = elements[e_key]
    return quad_tria


def get_global_coord(node):
    """
    Преобразует координаты циллиндрических и сферических СК к глобальной
     декартовой СК модели.

    :param node: объект GRID с информацией о узле

    :return: global_coord: координаты узла в глобальной декартовой СК
    """

    # Перевод в декартову систему координат
    if node.cp_ref.Type != 'R':
        cartesian_coord = get_cartesian(node.xyz, node.cp_ref.Type)
    else:
        cartesian_coord = node.xyz
    # Умножение декартовых координат узла на матрицу поворота для
    # перевода в глобальную СК
    transfer_cp_matrix = node.cp_ref.local_to_global
    orig = node.cp_ref.origin
    global_coord = cartesian_coord @ transfer_cp_matrix + orig

    return global_coord


def get_cartesian(coord, type):
    """
    Перевод в декартову систему координат.

    :param coord: массив с координатами
    :param type: информация о типе СК. (node.cp_ref.type)

    :return: cartezian_disp: перемещения узла в  декартовой СК

    """

    if type == 'C':
        tc = SkyCoord(rho=coord[0], phi=coord[1],
                      z=coord[2],
                      unit=('mm', 'deg', 'mm'),
                      representation_type='cylindrical')
        cartezian_disp = tc.cartesian.xyz.value

    elif type == 'S':
        tc = SkyCoord(rho=coord[0], phi=coord[1],
                      z=coord[2],
                      unit=('mm', 'deg', 'deg'),
                      representation_type='cylindrical')
        cartezian_disp = tc.cartesian.xyz.value

    else:
        cartezian_disp = coord

    return cartezian_disp


def get_angles(coord, type):
    tc = SkyCoord(x=coord[0], y=coord[1],
                  z=coord[2],
                  unit=('mm', 'mm', 'mm'),
                  representation_type='cartesian')
    if type == 'C':
        phi = tc.cylindrical.phi.value
    if type == 'S':
        phi = [tc.spherical.lon.value, tc.spherical.lat.value]

    return phi


def get_nondef_coords(shell_elm, swap):
    elm_nondef_coords = []
    elm_coords = dict.fromkeys(shell_elm)
    for e_id in shell_elm:
        nodes = []
        node_coords = []
        for i in range(len(shell_elm[e_id].nodes_ref)):
            node = shell_elm[e_id].nodes_ref[i]
            if node.cp != 0:
                node.xyz = get_global_coord(node)
                node.cp = 0
            nodes.append(node)
            node_coords.append(node.xyz)
        elm_coords[e_id] = nodes
        elm_nondef_coords.append(np.array(node_coords))

    # Замена коодринат XY, XZ, YZ друг на друга
    elm_nondef_coords = replace_coordinates(elm_nondef_coords, swap)

    return elm_coords, elm_nondef_coords


def get_elm_def_coords(elm_coord, nodes_increments, deform_multipler, swap, phi):
    """
    Подготовка массива с координатами узлов деформированных элементов

    :param elm_coord: список из элементов модели
    :param nodes_increments: список с приращениями перемещений узлов
    :param deform_multipler: множитель для масштабирования перемещений
    :param swap: режим поворота изображений (меняет оси местами):
         1 - XY, 2 - XZ, 3 - YZ

    :return: elm_def_coords: итоговый массив np.array с координатами узлов
        деформированных элементов (далее в plot_elements)

    """

    elm_def_coords = []
    for e_id in elm_coord:
        def_coords = []

        # Подготовка приращений перемещений
        for node in elm_coord[e_id]:
            eig_incr = nodes_increments[node.nid][:3]
            transfer_cd_matrix = node.cd_ref.local_to_global

            if node.cd_ref.Type == 'R':
                eig_displacement = eig_incr @ transfer_cd_matrix

            if node.cd_ref.Type == 'C':
                theta = phi[node.nid] * np.pi / 180
                r = np.array([[np.cos(theta), np.sin(theta), 0],
                              [-np.sin(theta), np.cos(theta), 0],
                              [0, 0, 1]])
                r_inv = np.linalg.inv(r)
                rot_eig_incr = r_inv @ eig_incr
                eig_displacement = rot_eig_incr @ transfer_cd_matrix

            if node.cd_ref.Type == 'S':
                theta = phi[node.nid] * np.pi / 180
                rz = np.array([[np.cos(theta), np.sin(theta), 0],
                               [-np.sin(theta), np.cos(theta), 0],
                               [0, 0, 1]])
                rx = np.array([[0, 0, 1],
                               [0, np.cos(theta), np.sin(theta)],
                               [0, -np.sin(theta), np.cos(theta)]])
                r = rz @ rx
                r_inv = np.linalg.inv(r)
                rot_eig_incr = r_inv @ eig_incr
                eig_displacement = rot_eig_incr @ transfer_cd_matrix

            deform_coord = node.xyz + eig_displacement * deform_multipler
            def_coords.append(deform_coord)
        elm_def_coords.append(np.array(def_coords))

    # Замена коодринат XY, XZ, YZ друг на друга
    elm_def_coords = replace_coordinates(elm_def_coords, swap)

    return elm_def_coords


@jit(nopython=True)
def rep_coord(coords, comp1, to_comp2):
    coords.T[comp1], coords.T[to_comp2] = coords.T[to_comp2].copy(), coords.T[comp1].copy()


def replace_coordinates(coords, swap):
    if swap == 1:
        comp, to_comp = 0, 1
    elif swap == 2:
        comp, to_comp = 0, 2
    elif swap == 3:
        comp, to_comp = 1, 2

    for i in coords:
        rep_coord(i, comp, to_comp)

    return coords


def get_scale_values(coords_limits, view_plane, view_ratio):
    """
    Возвращает кортеж для метода autoscale_xyz.

    :param coords_limits: список с минимальными и максимальными значениями
                          координат по всем осям.
           view_plane: список из двух или трех элементов, обозначающий
                       плоскость вида.
           view_ratio: соотношение сторон вида

    :return:
    """
    cl = coords_limits
    xyz2 = [(x[0] + x[1]) / 2 for x in cl]
    maxlen = [(x[1] - x[0]) / 2 for x in cl]

    ml = max(maxlen)
    scale = [[xyz2[0] - ml, xyz2[0] + ml],
             [xyz2[1] - ml, xyz2[1] + ml],
             [xyz2[2] - ml, xyz2[2] + ml]]

    # Подгонка соотношения сторон и масштаба для проекций
    if len(view_plane) == 2:
        vp = view_plane
        model_ratio = maxlen[vp[0]] / maxlen[vp[1]]
        coef = view_ratio / model_ratio
        if view_ratio >= model_ratio:
            scale[vp[0]] = [xyz2[vp[0]] - maxlen[vp[0]] * coef,
                            xyz2[vp[0]] + maxlen[vp[0]] * coef]
            scale[vp[1]] = [xyz2[vp[1]] - maxlen[vp[1]],
                            xyz2[vp[1]] + maxlen[vp[1]]]
        else:
            scale[vp[0]] = [xyz2[vp[0]] - maxlen[vp[0]],
                            xyz2[vp[0]] + maxlen[vp[0]]]
            scale[vp[1]] = [xyz2[vp[1]] - maxlen[vp[1]] / coef,
                            xyz2[vp[1]] + maxlen[vp[1]] / coef]
    return tuple(scale)


def get_coord_limits(lst_np):
    """
    Возвращает список с минимальными и максимальными значениями по трем осям.

    :param lst_np:
    :return:
    """
    all_coords = np.concatenate(lst_np, axis=0)
    maxmin = []
    maxmin.append([all_coords[:, 0].min(), all_coords[:, 0].max()])
    maxmin.append([all_coords[:, 1].min(), all_coords[:, 1].max()])
    maxmin.append([all_coords[:, 2].min(), all_coords[:, 2].max()])
    return maxmin


def get_plot_aratio(fig, grid_spec):
    """
    Возвращает соотношение сторон вида.

    :param fig:
    :param grid_spec:

    :return:
    """
    # Ширина и высота изображения (в inches)
    fig_size = fig.get_size_inches()
    # Отношение сторон (ширины к высоте)
    fig_aratio = fig_size[0] / fig_size[1]

    # Отношение сторон SubSpec к отношению сторон всего изображения
    c_sp = grid_spec.colspan
    r_sp = grid_spec.rowspan
    cols = grid_spec.get_gridspec().ncols
    rows = grid_spec.get_gridspec().nrows

    gs_aratio = ((c_sp.stop - c_sp.start) * rows) / \
                ((r_sp.stop - r_sp.start) * cols)

    # Для старой версии через метод get_rows_columns()
    # gss = grid_spec.get_rows_columns()
    # gs_aratio = ((gss[5] + 1 - gss[4]) * gss[0]) / (
    #         (gss[3] + 1 - gss[2]) * gss[1])

    return fig_aratio * gs_aratio


def plot_subplot(ax, elements, facecolor=None, edgecolor=None, transparency=None, linewidth=None):
    plotted_elm = a3.art3d.Poly3DCollection(elements)
    ax.add_collection3d(plotted_elm)

    if facecolor:
        plotted_elm.set_facecolor(facecolor)
    if edgecolor:
        plotted_elm.set_edgecolor(edgecolor)
    if transparency:
        plotted_elm.set_alpha(transparency)
    if linewidth:
        plotted_elm.set_linewidth(linewidth)


def plot_elements(eig_disp, def_disp, eig_descript, coef_scale):
    """
    Построение рисунков.

    :param coord_elements:

    :return:
    """

    # Параметры текста для подписи
    subtitle_font = {'fontname': 'Times New Roman'}
    height = 5
    width = 9

    # Создание графика с подписью
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    fig.suptitle('Deformations' +
                 '; Mode: ' + str(eig_descript[0]) +
                 '; Freq.: ' + str(eig_descript[1]) + '.',
                 x=0.98,
                 y=0.05,
                 ha='right',
                 va='top',
                 fontsize=12,
                 **subtitle_font)

    # Разбивка графика на сетку grid_spec
    gs = gridspec.GridSpec(ncols=2, nrows=7, figure=fig)
    grid_specs = [gs[:2, 0], gs[:2, 1], gs[2:, 0], gs[2:, 1]]
    view_planes = [[1, 2], [0, 2], [1, 0], [1, 2, 3]]
    m_rotate = [[0, 0], [90, 0], [0, 90], [45, 25]]
    m_dist = [5.5 / coef_scale,
              5.5 / coef_scale,
              5.5 / coef_scale,
              4.5 / coef_scale]

    # Формирование изображений
    for i in range(len(grid_specs)):
        ax = fig.add_subplot(grid_specs[i],
                             projection='3d',
                             azim=m_rotate[i][0],
                             elev=m_rotate[i][1],
                             proj_type='ortho')

        plot_subplot(ax, eig_disp, facecolor='w', edgecolor='k', transparency=None, linewidth=0.1)
        plot_subplot(ax, def_disp, facecolor='dimgray', edgecolor=None, transparency=0.1, linewidth=0.1)

        max_len = get_coord_limits(eig_disp)

        plot_aratio = get_plot_aratio(fig, grid_specs[i])
        ax.auto_scale_xyz(
            *get_scale_values(max_len, view_planes[i], plot_aratio))

        ax.dist = m_dist[i]
        ax.grid(False)
        ax.set_axis_off()

    return fig
