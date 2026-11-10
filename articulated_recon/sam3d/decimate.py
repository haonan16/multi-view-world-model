"""decimate SAM 3D meshes to ~20k faces (open3d quadric) -> <name>_20k.ply; loops until no new meshes for 40 min"""
import glob, os, time
import open3d as o3d
idle = 0
while idle < 40:
    new = 0
    for f in glob.glob("/n/lab_storage/ydu_lab/Lab/akiruga/articulated_recon/real/v2/*/sam3d/*.ply"):
        if f.endswith(("_20k.ply", "_cam.ply", "_gs.ply")) or os.path.exists(f[:-4] + "_20k.ply"):
            continue
        if time.time() - os.path.getmtime(f) < 20 or not os.path.exists(f[:-4] + ".json"):
            continue
        m = o3d.io.read_triangle_mesh(f)
        m = m.simplify_quadric_decimation(20000)
        m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()
        o3d.io.write_triangle_mesh(f[:-4] + "_20k.ply", m)
        print(f, len(m.triangles), flush=True); new += 1
    idle = 0 if new else idle + 1
    time.sleep(60)
