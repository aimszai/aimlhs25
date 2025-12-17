import numpy as np

def input_vectors():
    collector = {}
    csv = open('prüfungsvorbereitung/vektor.csv','r')
    lines = csv.readlines()
    csv.close()
    for line in lines[1:]:
        key, value = line.strip().split(',')
        collector[key] = value
    return collector

def parse_vector(collector):
    for key, vector in collector.items():
        if len(vector) > 0:
            vector = vector.split(' ')
            vector = [float(x.strip()) for x in vector]
            vector = np.array(vector)
            collector[key] = vector
        else:
            print("Empty vector input")
    return collector

def calculate_degrees(a, c):
    # Calculate angle between direction vectors a and c
    dot_product = np.dot(a, c)
    norm_a = np.linalg.norm(a)
    norm_c = np.linalg.norm(c)
    cos_phi = np.abs(dot_product)/(norm_a * norm_c)
    cos_phi = np.clip(cos_phi, -1.0, 1.0)# Ensure within valid range for arccos
    phi_rad = np.arccos(cos_phi)
    phi_deg = np.degrees(phi_rad)
    print("\n################# Angles ###############\n")
    print(f"Angle between direction vectors (in radians): {phi_rad:.4f}")
    print(f"Angle between direction vectors (in degrees): {phi_deg:.4f}")


def check(distance):
    tolerance = 1e-9
    if distance < tolerance:
        print(f"Distance between points of intersection: {distance:.4f}")
    else:
        print(f"Warning: Points of intersection do not match within tolerance. Distance: {distance:.4f}")

def main():
    collector = input_vectors()
    collector = parse_vector(collector)
    try:
        a = collector['vektor_a']
        b = collector['vektor_b']
        c = collector['vektor_c']
        d = collector['vektor_d']
        e = collector['vektor_e']
    except:
        print("Fehler bei der Umwandlung der Vektoren")
        return
    
    is_line_line = len(e) == 0

    print("\n################# Calculation ###############\n");

    RS = d - b
    if is_line_line:
        A = np.column_stack((a, -c))
    else:
        A = np.column_stack((a, e,-c))
    
    solution, residuals, rank, s = np.linalg.lstsq(A, RS, rcond=None)

    if is_line_line:
        lamda_, miu = solution
        P_1 = b + lamda_ * a
        P_2 = d + miu * c
        distance = np.linalg.norm(P_1 - P_2)
    else:
        lamda_, epsilon, miu = solution
        P_1 = b + lamda_ * a + epsilon * e
        P_2 = d + miu * c
        # p_1 is point on the plane x = b + lamda * a + epsilon * e
        distance = np.linalg.norm(P_1 - P_2)# P_2 is point on the line d + miu * c

    check(distance)
    
    calculate_degrees(a,c)

    print("\n################# Results ###############\n")
    print(f"Problem Type: {'Line-Line' if is_line_line else 'Line-Plane'}")
    print(f"Parameters found: \n    - λ = {lamda_:.4f} \n    - μ = {miu:.4f} \n Point of Intersection:")
    if np.allclose(P_1, P_2):
        print(f"    - P1/2 = {P_1} ")
    else:
        print(f"    - P1 = {P_1} \n    - P2 = {P_2} \n")
    print(f"Residuals: \n    {residuals}")

if __name__ == '__main__':
    main()