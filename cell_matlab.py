def cell(a, b):
    outer = []
    for i in range(1,a+1):
        for j in range(1,b+1):
            inner = []
            inner.append(i)
            inner.append(j)
            outer.append(inner)
    for i in outer:
        i.clear()
    return outer

if __name__ == "__main__":
    A = cell(1,10)
    print(A)

