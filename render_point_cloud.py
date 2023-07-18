import pyvista as pv
import numpy as np

pv.set_plot_theme('dark')

number = '013'

point_cloud = pv.read('./test_data/'+number+' cleaned.ply')

point_cloud = point_cloud.rotate_z(90)
# point_cloud = point_cloud.rotate_y(180)

pl = pv.Plotter(off_screen=True)
pl.add_mesh(point_cloud)
pl.enable_eye_dome_lighting()
pl.camera_position = 'xz'
pl.remove_legend()
pl.show(screenshot=number+'_4.png')

point_cloud = point_cloud.rotate_z(180)

pl = pv.Plotter(off_screen=True)
pl.add_mesh(point_cloud)
pl.enable_eye_dome_lighting()
pl.camera_position = 'xz'
pl.remove_legend()
pl.show(screenshot=number+'_3.png')

point_cloud = point_cloud.rotate_z(90)
point_cloud = point_cloud.rotate_y(90)
point_cloud = point_cloud.rotate_z(90)

pl = pv.Plotter(off_screen=True)
pl.add_mesh(point_cloud)
pl.enable_eye_dome_lighting()
pl.camera_position = 'xz'
pl.remove_legend()
pl.show(screenshot=number+'_2.png')

point_cloud = point_cloud.rotate_z(180)

pl = pv.Plotter(off_screen=True)
pl.add_mesh(point_cloud)
pl.enable_eye_dome_lighting()
pl.camera_position = 'xz'
pl.remove_legend()
pl.show(screenshot=number+'_1.png')

