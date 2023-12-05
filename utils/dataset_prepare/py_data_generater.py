#
import trimesh
import pygeodesic.geodesic as geodesic
import numpy as np
import os
import multiprocessing as mp
from threading import Thread
from tqdm import tqdm



PATH_TO_MESH = "./data/tiny/meshes/"
PATH_TO_OUTPUT_NPZ = "./data/tiny/npz/"
PATH_TO_OUTPUT_FILELIST = "./data/tiny/filelist/"

TRAINING_SPLIT_RATIO = 0.8


def visualize_ssad(vertices: np.ndarray, triangles: np.ndarray, source_index: int):
    # Initialise the PyGeodesicAlgorithmExact class instance
    geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, triangles)

    # Define the source and target point ids with respect to the points array
    source_indices = np.array([source_index])
    target_indices = None
    distances, best_source = geoalg.geodesicDistances(source_indices, target_indices)
    return distances


def visualize_two_pts(vertices: np.ndarray, triangles: np.ndarray, source_index: int, dest_index: int):
    # Initialise the PyGeodesicAlgorithmExact class instance
    geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, triangles)

    # Define the source and target point ids with respect to the points array
    source_indices = np.array([source_index])
    target_indices = np.array([dest_index])
    distances, best_source = geoalg.geodesicDistance(source_indices, target_indices)
    return distances


def data_prepare_ssad(object_file: str, output_path: str, source_index: int):
    vertices = []
    triangles = []

    with open(object_file, "r") as f:
        lines = f.readlines()
        for each in lines:
            if len(each) < 2:
                continue
            if each[0:2] == "v ":
                temp = each.split()
                vertices.append([float(temp[1]), float(temp[2]), float(temp[3])])
            if each[0:2] == "f ":
                temp = each.split()
                # 
                temp[3] = temp[3].split("/")[0]
                temp[1] = temp[1].split("/")[0]
                temp[2] = temp[2].split("/")[0]
                triangles.append([int(temp[1]) - 1, int(temp[2]) - 1, int(temp[3]) - 1])
    vertices = np.array(vertices)
    triangles = np.array(triangles)


    # Initialise the PyGeodesicAlgorithmExact class instance
    geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, triangles)

    # Define the source and target point ids with respect to the points array
    source_indices = np.array([source_index])
    target_indices = None
    distances, best_source = geoalg.geodesicDistances(source_indices, target_indices)


def data_prepare_gen_dataset(object_file: str, output_path: str, num_sources, num_each_dest, tqdm_on=True):
    vertices = []
    triangles = []

    with open(object_file, "r") as f:
        lines = f.readlines()
        for each in lines:
            if len(each) < 2:
                continue
            if each[0:2] == "v ":
                temp = each.split()
                vertices.append([float(temp[1]), float(temp[2]), float(temp[3])])
            if each[0:2] == "f ":
                temp = each.split()
                # 
                temp[3] = temp[3].split("/")[0]
                temp[1] = temp[1].split("/")[0]
                temp[2] = temp[2].split("/")[0]
                triangles.append([int(temp[1]) - 1, int(temp[2]) - 1, int(temp[3]) - 1])
    vertices = np.array(vertices)
    triangles = np.array(triangles)


    # Initialise the PyGeodesicAlgorithmExact class instance
    geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, triangles)

    result = np.array([[0,0,0]])

    sources = np.random.randint(low=0, high=len(vertices), size=[num_sources])

    # iterate
    # only the process on process #0 will be displayed
    # this should not be problematic or confusing on most homogeneous CPUs
    it = tqdm(range(num_sources)) if tqdm_on else range(num_sources)
    for i in it:
        source_indices = np.array([sources[i]])
        target_indices = np.random.randint(low=0, high=len(vertices), size=[num_each_dest])
        if source_indices.max() >= len(vertices):
            print("!!!!!!!", source_indices.max(), len(vertices))
        if target_indices.max() >= len(vertices):
            print("!!!!!!!", target_indices.max(), len(vertices))

        distances, best_source = geoalg.geodesicDistances(source_indices, target_indices)

        a = source_indices.repeat([num_each_dest]).reshape([-1,1])
        b = target_indices.reshape([-1,1])
        c = distances.reshape([-1,1])
        new = np.concatenate([a, b, c], -1)
        result = np.concatenate([result, new])

    np.savetxt(output_path, result)


#############################################
def computation_thread(filename, object_name, a, b, c, d, idx=None):
    assert idx != None, "an idx has to be given"
    tqdm_on = False
    if idx == 0:
        tqdm_on = True
    print(filename, object_name)
    data_prepare_gen_dataset(filename, object_name + "_train_" + str(idx), a, b, tqdm_on=tqdm_on)
   # data_prepare_gen_dataset(filename, "utils/dataset_prepare/outputs/" + object_name + "_test_" + str(idx), c, d, tqdm_on=tqdm_on)


##############################################




if __name__ == "__main__":
    '''
    
    
    a: on training set, how many sources to randomly sample.
    b: on training set, for each source, how many dest to randomly sample.
    c: on testing set, how many sources to randomly sample.
    d: on testing set, for each source, how many dest to randomly sample.
    threads: how many threads to use. 0 means all cores.
    
    '''

    #############################################################
    object_name = None 
    a = 300
    b = 800
    c = 400
    d = 60
    file_size_threshold = 12_048_576     # a threshold to filter out large meshes
    threads = 1
    #############################################################

    assert threads >= 0 and type(threads) == int
    if threads == 0:
        threads = mp.cpu_count()
        print(f"Automatically utilize all CPU cores ({threads})")
    else:
        print(f"{threads} CPU cores are utilized!")

    # make dirs, if not exist
    if os.path.exists(PATH_TO_OUTPUT_NPZ) == False:
        os.mkdir(PATH_TO_OUTPUT_NPZ)
    if os.path.exists(PATH_TO_OUTPUT_FILELIST) == False:
        os.mkdir(PATH_TO_OUTPUT_FILELIST)
            
    all_files = []
    for mesh in os.listdir(PATH_TO_MESH):
        # check if the file is too large
        if os.path.getsize(PATH_TO_MESH + mesh) < file_size_threshold:
            all_files.append(PATH_TO_MESH + mesh)


    
    #all_files = os.listdir("./inputs")

    object_names = all_files
    for i in range(len(object_names)):
        if object_names[i][-4:] == ".obj":
            object_names[i] = object_names[i][:-4]

    
    print(f"Current dir: {os.getcwd()}, object to be processed: {len(object_names)}")
   
   
    # handle the case when the output file already exists
    for i in tqdm(range(len(object_names))):
        object_name = object_names[i]
        if object_name.split("/")[-1][0] == ".":
            continue        # not an obj file
        
        filename_out = PATH_TO_OUTPUT_NPZ + object_name + ".npz"
        if os.path.exists(filename_out):
            continue
        
        filename = object_name + ".obj"
       

        train_data_filename_list = []
        test_data_filename_list = []


        pool = []

        for t in range(threads):
            task = mp.Process(target=computation_thread, args=(filename, object_name, a//threads,b,c//threads,d, t,))
            task.start()
            pool.append(task)
        for t, task in enumerate(pool):
            task.join()
            train_data_filename_list.append(object_name + "_train_" + str(t))
            #test_data_filename_list.append("./utils/dataset_prepare/outputs/" + object_name + "_test_" + str(t))
        #breakpoint()
        # 整合多线程的结果到一起
        #print(object_name)
        try:
            for i in range(len(train_data_filename_list)):
                # train data
                with open(object_name + "_train_" + str(i), "r") as f:
                    data = f.read()
                with open(object_name + "_train", "a") as f:
                    f.write(data)
        except:
            #print("Error on " + object_name + ", this is mostly due to non-manifold (failed to initialise the PyGeodesicAlgorithmExact class instance)")
            continue


        # 清理掉中间过程文件
        #print("qqqqqq")
        #breakpoint()
        for each in (train_data_filename_list + test_data_filename_list):
            os.remove(each)

        filename_in = object_name + ".obj"
        dist_in = object_name + "_train"
        filename_out = PATH_TO_OUTPUT_NPZ + object_name.split("/")[-1] + ".npz"
        try:
            # 由于下述问题，有时trimesh loader会返回错误的顶点数量（并在训练时导致数组越界）
            # https://github.com/mikedh/trimesh/issues/489
            # 参考sanity check部分，对于出现这种错误的mesh，我们采取最直接的解决方案：放弃该mesh
            # 当使用QEM方法简化网格的时候，似乎更可能出现这个问题，出现率估计约百分之一到千分之一。
            # 下面的mesh就是一个例子：(1.3 remeshing + 0.85 QEM)
            # /mnt/sdb/pangbo/gnn-dist/utils/dataset_prepare/simplified_obj/02828884/26f583c91e815e8fcb2a965e75be701c.obj
            mesh = trimesh.load_mesh(filename_in)
            dist = np.loadtxt(dist_in)
        except Exception:
            print(f"load {filename_in} or {dist_in} failed...")
            continue
        
        # delete the dist_in
        os.remove(dist_in)

        # 额外保存图的拓扑结构（edges）
        # trimesh 的 .edge_unique 可以得到该网格的所有边（不重复）
        aa = mesh.edges_unique
        # 我们需要双向边，所以还需一点点额外处理
        bb = np.concatenate([aa[:, 1:], aa[:, :1]], 1)
        cc = np.concatenate([aa, bb])
        
       # breakpoint()
        # sanity check
        vertices = mesh.vertices
        if dist.max() > 100000000:
            print("inf encountered!!")
        elif ((dist.astype(np.float32).max()) >= vertices.shape[0]):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(object_name, "encountered trimesh loading error!!")
            continue

        np.savez(filename_out,
                edges=cc,
                vertices=mesh.vertices.astype(np.float32),
                normals=mesh.vertex_normals.astype(np.float32),
                faces=mesh.faces.astype(np.float32),
                dist_val=dist[:, 2:].astype(np.float32),
                dist_idx=dist[:, :2].astype(np.uint16),
        )



        


    print("\nnpz data generation finished. Now generating filelist...\n")
    # 最后生成 filelist
    lines = []
    breakpoint()
    for each in tqdm(object_names):
        filename_out = PATH_TO_OUTPUT_NPZ + each.split("/")[-1] + ".npz"
        try:
            dist = np.load(filename_out)
            # sanity check
            if dist['dist_val'].max() != np.inf and dist['dist_val'].max() < 1000000000:
                lines.append(filename_out + "\n")
            else:
                print(f"{filename_out} not good, contains inf!")
                continue
        except Exception:
            print(f"load {filename_out} failed for unknown reason.")
            continue
    
    import random
    random.shuffle(lines)
    # split
    train_num = int(len(lines) * TRAINING_SPLIT_RATIO)
    test_num  = len(lines) - train_num
    train_lines = lines[:train_num]
    test_lines  = lines[train_num:]
    
    with open(PATH_TO_OUTPUT_FILELIST + 'filelist_train.txt', 'w') as f:
        f.writelines(train_lines)
        
    with open(PATH_TO_OUTPUT_FILELIST + 'filelist_test.txt', 'w') as f:
        f.writelines(test_lines)