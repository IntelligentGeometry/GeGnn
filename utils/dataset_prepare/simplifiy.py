
import os 
import sys 
import pymeshlab 
from tqdm import tqdm
import multiprocessing as mp
from threading import Thread
 
def walk_dir(dir): 
    full_path = [] 
    filename = [] 
    for root, dirs, files in os.walk(dir): 
        for file in files: 
            if file.endswith(".obj"): 
                full_path.append(os.path.join(root, file)) 
                filename.append(file) 
    return [full_path, filename] 
 
 
def simplify_mesh(input_mesh, output_mesh):
    if os.path.exists(input_mesh) == False:
        return
    if os.path.exists(output_mesh):
        return

    ms = pymeshlab.MeshSet() 
    ms.load_new_mesh(input_mesh) 
    m = ms.current_mesh()
    num_v = m.vertex_matrix().shape[0]
    num_f = m.face_matrix().shape[0]

    # 
    simplify_method = "combined"
    
    if simplify_method == "QEM":
        # not recommended. QEM may produce very strange topology/degenerate triangles, which is not good for training
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=2000+num_f//4) 
    elif simplify_method == "remesh":
        # the distribution of triangles is uniform. Not so good since we want to make the network "exposed" to more complex triangulations
        ms.meshing_isotropic_explicit_remeshing(iterations=6, targetlen=pymeshlab.Percentage(1.5)) 
    elif simplify_method == "combined":
        # used in our paper
        ms.meshing_isotropic_explicit_remeshing(iterations=6, targetlen=pymeshlab.Percentage(1.35)) 
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(num_f*0.85)) 
    else:
        raise NotImplementedError
    
    ms.save_current_mesh(output_mesh)
  #  try:
  #      pass
  #      
  #  except Exception:
  #      print(f"Mesh {input_mesh} failed")
 
 
 
if __name__ == "__main__": 
    
    threads = 4
    dir = "path/to/your/dataset"
    output_dir = "path/to/your/output/directory"
    

    [full_path, filename] = walk_dir(dir) 
    
    try:
        os.mkdir(output_dir)
    except:
        pass
    
    for i in tqdm(range(len(full_path))): 
        r = i % threads
        pool = []
        
        input_mesh = full_path[i] 
        output_mesh = output_dir+filename[i] 
        task = mp.Process(target=simplify_mesh, args=(input_mesh, output_mesh))
        task.start()
        pool.append(task)
        
        if r == 0:        
            for t, task in enumerate(pool):
                task.join()
            

