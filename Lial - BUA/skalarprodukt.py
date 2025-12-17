import numpy as np

def input_vectors():
    collector = {}
    x = input("csv(1) or manual?(2)")
    if x == '1':
        csv = open('prüfungsvorbereitung/vektor.csv','r')
        lines = csv.readlines()
        csv.close()
        for line in lines[1:]:
            key, value = line.strip().split(',')
            collector[key] = value
    else:
        vector_a = input("Enter vector a (space-separated numbers): ")
        vector_b = input("Enter vector b (space-separated numbers): ")
        collector['vektor_a'] = vector_a
        collector['vektor_b'] = vector_b
    return collector

def parse_vector(collector):
    for key, vector in collector.items():
        if len(vector) > 0:
            vector = vector.split(' ')
            vector = [float(x.strip()) for x in vector]
            vector = np.array(vector)
            collector[key] = vector
        else:
            print(f"Empty vector input {key}")
    return collector

def calculate_scalar_product(a, b):
    print(a.shape)
    if a.shape != b.shape:
        print("Error: Vectors must be of the same dimension.")
        exit(1)
    result = 0
    for i in range(a.shape[0]):
        tmp = a[i]*b[i]
        result += tmp
    return result

def main():
    collector = input_vectors()
    collector = parse_vector(collector)
    try:
        a = collector['vektor_a']
        b = collector['vektor_b']
    except:
        print("Fehler bei der Umwandlung der Vektoren")
        return

    result = calculate_scalar_product(a, b)
    print(f"Skalarprodukt ist: {result}")

if __name__ == "__main__":
    main()