import numpy as np
import matplotlib.pyplot as plt

def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance



width = 300
height = 200

max_depth = 2
soft_shadows = True

camera = np.array([0, 0, 1])
ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio) # left, top, right, bottom
sky_color = np.array([0.35, 0.65, 1])


light = { 'center': np.array([5, 5, 5]), 'radius': 2, 'n_checks': 1, 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }

objects = [
    { 'center': np.array([-0.5, 0, -1]), 'radius': 0.7, 'ambient': np.array([0.1, 0, 0]), 'diffuse': np.array([0.7, 0, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.4 },
    { 'center': np.array([0.4, 0, -0.3]), 'radius': 0.2, 'ambient': np.array([0.1, 0, 0.1]), 'diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.2 },
    { 'center': np.array([-0.2, 0.3, -0.1]), 'radius': 0.15, 'ambient': np.array([0, 0.1, 0]), 'diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.6 },
    { 'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.9, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([0, 0, 0]), 'shininess': 100, 'reflection': 0 }
]
#    { 'center': np.array([0, 0, -50000]), 'radius': 49900, 'ambient': np.array([0.35, 0.65, 1]), 'diffuse': np.array([0, 0, 0]), 'specular': np.array([0, 0, 0]), 'shininess': 0, 'reflection': 0.1 }



image = np.zeros((height, width, 3))
for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    illumination = np.zeros((3))
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
        pixel = np.array([x, y, 0])
        origin = camera
        direction = normalize(pixel - origin)

        color = np.zeros((3))
        reflection = 1
        n_shadowed = 0

        if soft_shadows:
            for l in range(-2, 2 +1):
                for m in range(-2, 2 +1):

                    light_center = [light['center'][0] + l/5, light['center'][1] + m/5, light['center'][2]]

                    nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
                    if nearest_object is None:
                        break

                    # compute intersection point between ray and nearest object
                    intersection = origin + min_distance * direction

                    normal_to_surface = normalize(intersection - nearest_object['center'])
                    shifted_point = intersection + 1e-5 * normal_to_surface
                    intersection_to_light = normalize(light_center - shifted_point)

                    _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)
                    intersection_to_light_distance = np.linalg.norm(light_center - intersection)
                    is_shadowed = min_distance < intersection_to_light_distance

                    if is_shadowed:
                        n_shadowed += 1 
        
        reflection = 1 - n_shadowed/25

        for k in range(max_depth):
            

            nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)

            if nearest_object is None:
                if k == 0:
                    color = sky_color
                break

            
            # compute intersection point between ray and nearest object
            intersection = origin + min_distance * direction

            normal_to_surface = normalize(intersection - nearest_object['center'])
            shifted_point = intersection + 1e-5 * normal_to_surface
            intersection_to_light = normalize(light['center'] - shifted_point)

            if soft_shadows == False:
                _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)
                intersection_to_light_distance = np.linalg.norm(light['center'] - intersection)
                is_shadowed = min_distance < intersection_to_light_distance
                

                if is_shadowed:
                    break       


            # RGB
            illumination = np.zeros((3))

            # ambiant
            illumination += nearest_object['ambient'] * light['ambient']

            # diffuse
            illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light, normal_to_surface)

            # specular
            intersection_to_camera = normalize(camera - intersection)
            H = normalize(intersection_to_light + intersection_to_camera)
            illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4)

            # reflection
            color += reflection * illumination 
            reflection *= nearest_object['reflection']

            origin = shifted_point
            direction = reflected(direction, normal_to_surface) 
            #print("k:", k)      


        image[i, j] = np.clip(color, 0, 1)
        
        # image[i, j] = ...
        #print("progress h:%d/%d" % (i + 1, height), "w:%d/%d" % (j+1, width))
        #print(nearest_object, "h: %d/%d" % (i + 1, height), "w: %d/%d" % (j+1, width))
    
    print("%d/%d" % (i + 1, height))


plt.imsave('image.png', image)
