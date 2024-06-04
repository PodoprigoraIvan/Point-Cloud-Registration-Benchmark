from numpy import linalg as LA
import numpy as np
import open3d as o3

def compute_rotation_matrix(points1, points2):
    H = points1.T @ points2
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    det = np.linalg.det(R)
    if det < 0:
        Vt[:, 2] *= -1
        R = Vt @ U.T @ np.diag([-1,1,1])
    res = np.eye(4)
    res[:3, :3] = R
    return res

def moveCentroidToZero(eigenvectors):
    x = -eigenvectors[0][0] - eigenvectors[1][0] - eigenvectors[2][0]
    y = -eigenvectors[0][1] - eigenvectors[1][1] - eigenvectors[2][1]
    z = -eigenvectors[0][2] - eigenvectors[1][2] - eigenvectors[2][2]
    return np.append(eigenvectors, [[x,y,z]], axis=0)

def calculatePointCloudCovMatrix(pcl):
    pointsArr = np.asarray(pcl.points)
    coordsLists = np.stack((pointsArr[:, 0], pointsArr[:, 1], pointsArr[:, 2]), axis=0)
    return np.cov(coordsLists)

def sortEig(eigenvalues, eigenvectors):
    sorted_indices = np.argsort(eigenvalues)
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    return eigenvectors_sorted

def getMainComponents(pointCloud):
    covarianceMatrix = calculatePointCloudCovMatrix(pointCloud)
    eigenvectors = sortEig(*LA.eig(covarianceMatrix))
    eigenvectors = eigenvectors.T
    return eigenvectors

def getPCAtransform(source, target):
    source.points = o3.utility.Vector3dVector(source.points - np.mean(source.points, axis=0))
    target.points = o3.utility.Vector3dVector(target.points - np.mean(target.points, axis=0))

    eigenvectorsSrc = getMainComponents(source)
    eigenvectorsTgt = getMainComponents(target)

    tmpPcl1 = o3.geometry.PointCloud()
    tmpPcl2 = o3.geometry.PointCloud()
    eigenvectorsSrc = moveCentroidToZero(eigenvectorsSrc)
    eigenvectorsTgt = moveCentroidToZero(eigenvectorsTgt)
    tmpPcl1.points = o3.utility.Vector3dVector(eigenvectorsSrc)
    tmpPcl2.points = o3.utility.Vector3dVector(eigenvectorsTgt)
    transformMatrix = compute_rotation_matrix(np.asarray(tmpPcl1.points), np.asarray(tmpPcl2.points))
    return transformMatrix