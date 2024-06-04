# -*- coding: utf-8 -*-
import copy
import math
import numpy as np
import open3d as o3
from probreg import cpd, l2dist_regs
from tkinter import *
import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Combobox  
import point_cloud_utils as pcu
import time
from PCA import getPCAtransform, getMainComponents

loaded = None
source = None
target = None
result = None

def RegistrationResultRMSE(target, transformedSource):
    distances, corres = pcu.k_nearest_neighbors(np.asarray(transformedSource.points), np.asarray(target.points), 1)
    return np.sqrt((distances**2).mean())

def applyDensityFromEntry(entry, pointCloud):
    if entry.get() == "":
        percent = 0.1
    else:
        percent = max(0.1, int(entry.get())/100.0)
    numToKeep = int(len(pointCloud.points)*percent)
    points = np.asarray(pointCloud.points)
    indicesToKeep = np.random.choice(len(points), numToKeep, replace=False)
    points = [points[i] for i in indicesToKeep]
    pointCloud.points = o3.utility.Vector3dVector(np.asarray(points))

def calculate_angle(vector1, vector2):
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(a**2 for a in vector1))
    magnitude2 = math.sqrt(sum(a**2 for a in vector2))
    cos_theta = min(dot_product / (magnitude1 * magnitude2), 1)
    radians = math.acos(cos_theta)
    degrees = math.degrees(radians)
    return degrees

def getMaxAngleBetweenMainComponents(pointCloud1, pointCloud2):
    components1 = getMainComponents(pointCloud1)
    components2 = getMainComponents(pointCloud2)
    max_angle = 0
    for vector1, vector2 in zip(components1, components2):
        angle = calculate_angle(vector1, vector2)
        if angle > max_angle:
            max_angle = angle
    return max_angle

def rotatePointCloud(pointCloud, angleX, angleY, angleZ):
    pointCloud.transform(np.array([[np.cos(angleX)*np.cos(angleY), np.cos(angleX)*np.sin(angleY)*np.sin(angleZ)-np.sin(angleX)*np.cos(angleZ), np.cos(angleX)*np.sin(angleY)*np.cos(angleZ)+np.sin(angleX)*np.sin(angleZ), 0.0],
                            [np.sin(angleX)*np.cos(angleY), np.sin(angleX)*np.sin(angleY)*np.sin(angleZ)+np.cos(angleX)*np.cos(angleZ), np.sin(angleX)*np.sin(angleY)*np.cos(angleZ)-np.cos(angleX)*np.sin(angleZ), 0.0],
                            [-np.sin(angleY), np.cos(angleY)*np.sin(angleZ), np.cos(angleY)*np.cos(angleZ), 0.0],
                            [0.0, 0.0, 0.0, 1.0]]))

# Вычисления
def onStart():
    global source
    global target
    global result
    global PCAVar
    global max_dist
    source = copy.deepcopy(loaded)
    target = copy.deepcopy(loaded)

    #Отсечение части облака по коэффициенту перекрытия
    percent = 0.1
    if entryOverlap.get() != "":
        percent = max(int(entryOverlap.get())/100.0, 0.1)
    pointsSortedByX = np.asarray(source.points)
    pointsSortedByX = pointsSortedByX[pointsSortedByX[:, 0].argsort()]
    num_points = len(pointsSortedByX)
    num_points_to_erase = int((num_points - int(num_points * percent)) / 2)
    target.points = o3.utility.Vector3dVector(pointsSortedByX[0:(num_points - num_points_to_erase)])
    source.points = o3.utility.Vector3dVector(pointsSortedByX[num_points_to_erase:])

    # Изменение целевого облака точек
    angleX = np.deg2rad(int(entryXangle.get()))
    angleY = np.deg2rad(int(entryYangle.get()))
    angleZ = np.deg2rad(int(entryZangle.get()))

    rotatePointCloud(target, angleX, angleY, angleZ)
    
    #Изменение плотностей облаков
    applyDensityFromEntry(entrySourceCloudDensity, source)
    applyDensityFromEntry(entryTargetCloudDensity, target)

    noise_coef = 0
    try:
        noise_coef = float(entryNoise.get())
    except ValueError:
        noise_coef = 0
        entryNoise.delete(0, END)
        entryNoise.insert(0, "0")
    noise = np.random.normal(0, max_dist*noise_coef, (len(target.points), 3))
    pointsWithNoise = np.asarray(target.points) + noise
    target.points = o3.utility.Vector3dVector(pointsWithNoise)
    PCAtransformMatrix = None
    tmpTgt = copy.deepcopy(loaded)
    rotatePointCloud(tmpTgt, angleX, angleY, angleZ)
    tmpResult = copy.deepcopy(loaded)
    if PCAVar.get() == True:
        tmpResult.points = o3.utility.Vector3dVector(tmpResult.points - np.mean(source.points, axis=0))
        tmpTgt.points = o3.utility.Vector3dVector(tmpTgt.points - np.mean(target.points, axis=0))
        PCAtransformMatrix = getPCAtransform(source, target)
        tmpResult.transform(PCAtransformMatrix)
        source.transform(PCAtransformMatrix)

    methodName = methodCombobox.get()
    registration = None
    tf_param = None
    conv_criteria = o3.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    start_time = time.time()
    if methodName == "ICP":
        registration = o3.pipelines.registration.registration_icp(
            source, target, max_dist/1.5, np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]), 
            o3.pipelines.registration.TransformationEstimationPointToPoint(),
            conv_criteria
        )
    if methodName == "ICP Point to plane":
        target.estimate_normals()
        registration = o3.pipelines.registration.registration_icp(
            source, target, max_dist/2.0, np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]), 
            o3.pipelines.registration.TransformationEstimationPointToPlane(),
            conv_criteria
        )
    if methodName == "Generalized ICP":
        registration = o3.pipelines.registration.registration_generalized_icp(
            source, target, max_dist/2.0, np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]), 
            o3.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            conv_criteria
        )
    if methodName == "CPD":
        tf_param, _, _ = cpd.registration_cpd(source, target, use_cuda=True)
    if methodName == "GMM":
        tf_param = l2dist_regs.registration_gmmreg(source, target, sigma=1.0, delta=0.95, n_gmm_components=1000, alpha=0.8, beta=0.3)
    labelTime.configure(text=f'Время, с: {time.time() - start_time}')
    result = copy.deepcopy(source)

    if methodName == "CPD" or methodName == "GMM":
        result.points = tf_param.transform(result.points)
        tmpResult.points = tf_param.transform(tmpResult.points)
    else:
        result.transform(registration.transformation)
        tmpResult.transform(registration.transformation)

    dist, corres = pcu.k_nearest_neighbors(np.asarray(result.points), np.asarray(target.points), 1)
    labelErrorMetric.configure(text="Сумма рассстояний между парами ближайших точек: " + str(round(dist.sum(), 5)))
    labelErrorMetricRMSE.configure(text="Среднеквадратичное отклонение расстояний меджу ближайшими точками: " + str(round(RegistrationResultRMSE(target, result), 5)))
    labelAngle.configure(text="Максимальное отклонение угла: " + str(round(getMaxAngleBetweenMainComponents(tmpTgt, tmpResult), 2)) + "°")

    showResultBtn.grid(column=2, row=1, pady=(10, 10), sticky=NW)

def browseFiles():
    global loaded
    global max_dist
    filename = filedialog.askopenfilename(initialdir = "./",
                                          title = "Select a File",
                                          filetypes = (("PCD files", "*.pcd*"), ("PLY files", "*.ply*"), ("all files", "*.*")))
    loaded = o3.io.read_point_cloud(filename)
    loaded.remove_non_finite_points()
    if len(loaded.points) < 1:
        return
    fileLbl.configure(text="Загружено облако точек из файла:\n"+filename)
    startBtn.grid(column=2, row=0, pady=(10,10), padx=(10,10), sticky=NW)
    bounding_box = loaded.get_axis_aligned_bounding_box()
    max_dist = np.linalg.norm(bounding_box.get_min_bound()- bounding_box.get_max_bound())

def clicked():
    vis = o3.visualization.Visualizer()
    vis.create_window(width=1400, height=800)
    
    vis.add_geometry(source)
    vis.add_geometry(target)
    vis.add_geometry(result)

    source.paint_uniform_color([1, 0, 0])
    target.paint_uniform_color([0, 1, 0])
    result.paint_uniform_color([0, 0, 1])
    vis.get_render_option().point_size = 2
    vis.get_render_option().background_color = [0.0,0.0,0.0]
    vis.run()
    vis.destroy_window()

def managePercentEntry(entry):
    text = entry.get()
    if len(text) > 3:
        entry.delete(0, END)
        entry.insert(0, text[:-1])
    if (len(text) > 0 and text[len(text)-1].isdigit() == False) or (len(text)==3 and text != "100"):
        entry.delete(0, END)
        entry.insert(0, text[:-1])

def onPercentInput(a, b, c):
    managePercentEntry(entryOverlap)
    managePercentEntry(entrySourceCloudDensity)
    managePercentEntry(entryTargetCloudDensity)

def createHelpWindow():
    helpWindow = tk.Toplevel(window)
    helpWindow.geometry('800x300')
    helpWindow.title("Справка")
    window.withdraw()
    def closeHelp():
        helpWindow.destroy()
        window.deiconify()
    helpWindow.protocol("WM_DELETE_WINDOW", closeHelp)
    backBtn = Button(helpWindow, text="Назад", font=("Arial Bold", 16), command=closeHelp)  
    backBtn.grid(column=0, row=0, padx=(5,5))  
    label = Label(helpWindow, anchor="w", text="Регистрация облаков точек - процесс сопоставления облаков точек, находящихся\n в одной системе координат и представляющих информацию об одном объекте.\n\nПеред началом работы необходимо загрузить файл с облаком точек (форматы PCD, PLY)\nКнопка \"Начать вычисления\" появится после загрузки облака точек, её нажание запускает выбранный метод регистрации \n с входными данными, обладающими введенными параметрами, с последующим выводов резульатов\n\nПредоставляются следующие результаты работы метода регистрации:\n - Метрики ошибки (сумма расстояний между ближайшими пар точек и среднеквадратическое отклонение расстояний от нуля)\n - Время работы метода регистрации в секундах\n - Визуализация результатов (целевое облако точек - зелёное, исходное - красное, результат преобразования исходного облака - синий)\n - Отклонение от ожидаемого угла в градусах", justify='left')
    label.grid(column=0, row=1, padx=(20,20), sticky=W)  

window = Tk() 
window.geometry('1100x600')
window.title("Бенчмарк методов регистрации облаков точек")  
main_menu = Menu()
main_menu.add_cascade(label="Справка", command=createHelpWindow)
window.config(menu=main_menu)

fileLbl = Label(window, text="")  
fileLbl.grid(column=1, row=0)  

fileBtn = Button(window, text="Открыть файл с облаком точек", command=browseFiles)  
fileBtn.grid(column=0, row=0, pady=(10, 10))  

lblXangle = Label(window, text="Угол между облаками точек в градусах по оси X:")  
lblXangle.grid(column=0, row=1)  
entryXangle = Entry(window, width=4, font=("Arial Bold", 20))
entryXangle.insert(0, "20")
entryXangle.grid(column=1, row=1, pady=(5, 5))  

lblYangle = Label(window, text="Угол между облаками точек в градусах по оси Y:")  
lblYangle.grid(column=0, row=2)  
entryYangle = Entry(window, width=4, font=("Arial Bold", 20))
entryYangle.insert(0, "0")
entryYangle.grid(column=1, row=2, pady=(5, 5))  

lblZangle = Label(window, text="Угол между облаками точек в градусах по оси Z:")  
lblZangle.grid(column=0, row=3)  
entryZangle = Entry(window, width=4, font=("Arial Bold", 20))
entryZangle.insert(0, "0")
entryZangle.grid(column=1, row=3, pady=(5, 5))  

labelOverlap = Label(window, text="Процент перекрытия облаков точек: ")
labelOverlap.grid(column=0, row=4, pady=(10, 10))

entryOverlapVar = StringVar()
entryOverlapVar.set('100')
entryOverlapVar.trace('w', onPercentInput)

entryOverlap = Entry(window, width=4, textvariable=entryOverlapVar, font=("Arial Bold", 20))
entryOverlap.grid(column=1, row=4)  

labelNoise = Label(window, text="Величина среднего отклонения в шуме:\n(доля от максимального расстояния в облаке)")
labelNoise.grid(column=0, row=5, pady=(10, 10))
entryNoise = Entry(window, width=4, font=("Arial Bold", 20))
entryNoise.insert(0, "0")
entryNoise.grid(column=1, row=5)  

entrySourceCloudDensityVar = StringVar()
entrySourceCloudDensityVar.set('100')
entrySourceCloudDensityVar.trace('w', onPercentInput)

labelSourceCloudDensity = Label(window, text="Плотность исходного облака:")
labelSourceCloudDensity.grid(column=0, row=6, pady=(10, 10))
entrySourceCloudDensity = Entry(window, width=4, textvariable=entrySourceCloudDensityVar, font=("Arial Bold", 20))
entrySourceCloudDensity.grid(column=1, row=6)  

entryTargetCloudDensityVar = StringVar()
entryTargetCloudDensityVar.set('100')
entryTargetCloudDensityVar.trace('w', onPercentInput)

labelTargetCloudDensity = Label(window, text="Плотность целевого облака:")
labelTargetCloudDensity.grid(column=0, row=7, pady=(10, 10))
entryTargetCloudDensity = Entry(window, width=4, textvariable=entryTargetCloudDensityVar, font=("Arial Bold", 20))
entryTargetCloudDensity.grid(column=1, row=7)  

labelMethod = labelTargetCloudDensity = Label(window, text="Метод регистрации:")
labelMethod.grid(column=0, row=8, pady=(10, 10))
methodCombobox = Combobox(window, state="readonly", values=["ICP", "Generalized ICP", "ICP Point to plane","CPD", "GMM"])
methodCombobox.current(0)
methodCombobox.grid(column=1, row=8, padx=(5, 5))

labelPCA = Label(window, text='Исппользовать метод главных компонент\nдля вычисления начального преобразования')
labelPCA.grid(column=0, row=9, pady=(10, 10))
PCAVar = BooleanVar()
PCAcheck = Checkbutton(window, variable=PCAVar)
PCAcheck.grid(column=1, row=9)

startBtn = Button(window, text="Начать вычисления", command=onStart)  
showResultBtn = Button(window, text="Визуализация результатов", command=clicked)  

labelErrorMetric = Label(window, text="", font=("Arial Bold", 12))
labelErrorMetric.grid(column=2, row=2, pady=(10, 10), sticky=NW)
labelErrorMetricRMSE = Label(window, text="", font=("Arial Bold", 12))
labelErrorMetricRMSE.grid(column=2, row=3, pady=(10, 10), sticky=NW)
labelTime = Label(window, text="", font=("Arial Bold", 12))
labelTime.grid(column=2, row=4, pady=(10, 10), sticky=NW)
labelAngle = Label(window, text="", font=("Arial Bold", 12))
labelAngle.grid(column=2, row=5, pady=(10, 10), sticky=NW)

window.mainloop()
